import re, cv2, numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def load_and_preprocess_image(filepath, size=512, gray=False):
    img = cv2.imread(filepath)
    if img is None: return None
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (size, size))
        img = img.astype('float32')/255.0
        img = np.stack([img, img, img], axis=-1)
        return img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype('float32')/255.0
    return img

def augment_image(image):
    out = []
    out.append(image)
    out.append(cv2.flip(image, 1))
    out.append(cv2.flip(image, 0))
    out.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    out.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    return [img_to_array(x) for x in out]
