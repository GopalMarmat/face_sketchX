import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters, kernel=4, strides=2, batchnorm=True):
    x = layers.Conv2D(filters, kernel, strides=strides, padding='same', use_bias=not batchnorm)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def deconv_block(x, skip, filters, kernel=4, dropout=False):
    x = layers.Conv2DTranspose(filters, kernel, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Concatenate()([x, skip])
    return x

def build_unet_generator(img_size=512):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    e1 = conv_block(inputs, 64, batchnorm=False)
    e2 = conv_block(e1, 128)
    e3 = conv_block(e2, 256)
    e4 = conv_block(e3, 512)
    e5 = conv_block(e4, 512)
    e6 = conv_block(e5, 512)
    e7 = conv_block(e6, 512)
    e8 = conv_block(e7, 512, batchnorm=False)
    d1 = deconv_block(e8, e7, 512, dropout=True)
    d2 = deconv_block(d1, e6, 512, dropout=True)
    d3 = deconv_block(d2, e5, 512, dropout=True)
    d4 = deconv_block(d3, e4, 512)
    d5 = deconv_block(d4, e3, 256)
    d6 = deconv_block(d5, e2, 128)
    d7 = deconv_block(d6, e1, 64)
    d8 = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(d7)
    return Model(inputs, d8, name='UNet_Generator')

def build_patchgan_discriminator(img_size=512):
    inp = layers.Input(shape=(img_size, img_size, 3))
    tar = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Concatenate()([inp, tar])
    x = conv_block(x, 64, batchnorm=False)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512, strides=1)
    x = layers.Conv2D(1, 4, strides=1, padding='same')(x)
    return Model([inp, tar], x, name='PatchGAN_Discriminator')
