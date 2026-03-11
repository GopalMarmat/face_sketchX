import os, glob, numpy as np, tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from src.model import build_unet_generator, build_patchgan_discriminator

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 200
LR_G = 2e-4
LR_D = 2e-4
CHECKPOINT = 'models/best_generator.h5'
SKETCH_GLOB = 'data/raw_data/sketches/*.jpg'
REAL_GLOB = 'data/raw_data/photos/*.jpg'

AUTOTUNE = tf.data.AUTOTUNE

def load_image_pair(sk_path, real_path):
    sk = tf.io.read_file(sk_path)
    sk = tf.image.decode_image(sk, channels=1)
    sk = tf.image.convert_image_dtype(sk, tf.float32)
    sk = tf.image.resize(sk, [IMG_SIZE, IMG_SIZE])
    real = tf.io.read_file(real_path)
    real = tf.image.decode_image(real, channels=3)
    real = tf.image.convert_image_dtype(real, tf.float32)
    real = tf.image.resize(real, [IMG_SIZE, IMG_SIZE])
    sk = sk * 2.0 - 1.0
    real = real * 2.0 - 1.0
    sk = tf.image.grayscale_to_rgb(sk)
    return sk, real

def build_dataset(sk_glob, re_glob, batch):
    sk_files = sorted(glob.glob(sk_glob))
    re_files = sorted(glob.glob(re_glob))
    assert len(sk_files) == len(re_files), "sketch and real counts must match"
    ds = tf.data.Dataset.from_tensor_slices((sk_files, re_files))
    def _py(a,b):
        return tf.py_function(func=load_image_pair, inp=[a,b], Tout=[tf.float32, tf.float32])
    ds = ds.map(_py, num_parallel_calls=AUTOTUNE)
    def augment(sk, real):
        if tf.random.uniform(()) > 0.5:
            sk = tf.image.flip_left_right(sk); real = tf.image.flip_left_right(real)
        return sk, real
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

vgg = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,3))
vgg.trainable = False
content_layers = ['block3_conv3','block4_conv2']
feat_extractor = tf.keras.Model(vgg.input, [vgg.get_layer(l).output for l in content_layers])

bce = BinaryCrossentropy(from_logits=True)
mae = MeanAbsoluteError()

class Pix2PixTrainer:
    def __init__(self, gen, disc, lambda_l1=100.0, lambda_perc=1.0):
        self.gen = gen; self.disc = disc
        self.lambda_l1 = lambda_l1; self.lambda_perc = lambda_perc
        self.opt_g = Adam(LR_G, beta_1=0.5); self.opt_d = Adam(LR_D, beta_1=0.5)

    @tf.function
    def train_step(self, sk, real):
        with tf.GradientTape() as g_t, tf.GradientTape() as d_t:
            fake = self.gen(sk, training=True)
            d_real = self.disc([sk, real], training=True)
            d_fake = self.disc([sk, fake], training=True)
            d_loss = 0.5 * (bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake))
            g_gan = bce(tf.ones_like(d_fake), d_fake)
            g_l1 = mae(real, fake) * self.lambda_l1
            real_pp = tf.keras.applications.vgg19.preprocess_input((real + 1.0) * 127.5)
            fake_pp = tf.keras.applications.vgg19.preprocess_input((fake + 1.0) * 127.5)
            f_real = feat_extractor(real_pp); f_fake = feat_extractor(fake_pp)
            perc = 0.0
            for a,b in zip(f_real, f_fake): perc += tf.reduce_mean(tf.abs(a-b))
            g_total = g_gan + g_l1 + perc * self.lambda_perc
        grads_g = g_t.gradient(g_total, self.gen.trainable_variables)
        grads_d = d_t.gradient(d_loss, self.disc.trainable_variables)
        self.opt_g.apply_gradients(zip(grads_g, self.gen.trainable_variables))
        self.opt_d.apply_gradients(zip(grads_d, self.disc.trainable_variables))
        return g_total, d_loss

def train():
    gen = build_unet_generator(IMG_SIZE)
    disc = build_patchgan_discriminator(IMG_SIZE)
    trainer = Pix2PixTrainer(gen, disc, lambda_l1=100.0, lambda_perc=1.0)
    ds = build_dataset(SKETCH_GLOB, REAL_GLOB, BATCH_SIZE)
    best = 1e9
    for epoch in range(1, EPOCHS+1):
        print(f'Epoch {epoch}/{EPOCHS}')
        losses = []
        for step, (sk, real) in enumerate(ds):
            g_loss, d_loss = trainer.train_step(sk, real)
            losses.append(float(g_loss))
            if step % 50 == 0:
                print(f' step {step} g_loss={g_loss:.4f} d_loss={d_loss:.4f}')
        mean_loss = np.mean(losses) if losses else 0.0
        print(' mean gen loss:', mean_loss)
        if mean_loss < best:
            best = mean_loss
            print(' Saving generator to', CHECKPOINT)
            os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
            gen.save(CHECKPOINT, include_optimizer=False)
    print('Training finished. Best model saved at', CHECKPOINT)

if __name__=='__main__':
    if len(glob.glob(SKETCH_GLOB))==0:
        print('No training data found. Place images in data/raw_data/sketches and data/raw_data/reals')
    else:
        train()
