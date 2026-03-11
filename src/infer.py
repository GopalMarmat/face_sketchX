import os, glob, numpy as np, tensorflow as tf
from PIL import Image, ImageFilter
from src.model import build_unet_generator

IMG_SIZE = 512
MODEL_PATH = 'models/best_generator.h5'
INPUT_GLOB = 'data/raw_data/sketches/*.png'

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess(path):
    img = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = (np.array(img).astype('float32') / 255.0)
    arr = np.stack([arr,arr,arr], axis=-1)
    arr = arr * 2.0 - 1.0
    return arr

def postprocess(pred):
    img = ((pred + 1.0) * 127.5).astype('uint8')
    pil = Image.fromarray(img)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=2))
    return pil

def run():
    model = build_unet_generator(IMG_SIZE)
    model.load_weights(MODEL_PATH)
    for i, p in enumerate(sorted(glob.glob(INPUT_GLOB))):
        x = preprocess(p)
        pred = model.predict(x[None,...])[0]
        out = postprocess(pred)
        out.save(os.path.join(OUTPUT_DIR, f'out_{i:04d}.png'))
        print('saved', i)

if __name__=='__main__':
    run()
