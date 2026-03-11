import os, numpy as np, cv2, glob
from src.utils.utils import sorted_alphanumeric, load_and_preprocess_image, augment_image

def build_npy(photo_dir='data/raw_data/photos', sketch_dir='data/raw_data/sketches', size=512, out_dir='data/processed'):
    os.makedirs(out_dir, exist_ok=True)
    imgs, sks = [], []
    photos = sorted_alphanumeric([f for f in os.listdir(photo_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    sketches = sorted_alphanumeric([f for f in os.listdir(sketch_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    for p in photos:
        img = load_and_preprocess_image(os.path.join(photo_dir,p), size=size)
        if img is None: continue
        for a in augment_image(img): imgs.append(a)
    for s in sketches:
        sk = load_and_preprocess_image(os.path.join(sketch_dir,s), size=size, gray=True)
        if sk is None: continue
        for a in augment_image(sk): sks.append(a)
    assert len(imgs) == len(sks), f'Counts mismatch: {len(imgs)} vs {len(sks)}'
    np.save(os.path.join(out_dir,'real_images.npy'), np.array(imgs))
    np.save(os.path.join(out_dir,'sketch_images.npy'), np.array(sks))
    print('Saved processed arrays to', out_dir)

if __name__=='__main__':
    build_npy()
