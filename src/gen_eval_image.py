import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import os
from PIL import Image

from src.model import build_unet_generator

# ================= MediaPipe =================
from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

# ================= CONFIG =================
IMG_SIZE = 512
GEN_MODEL_PATH = "models/best_generator.h5"
FACE_MODEL_PATH = "face_landmarker.task"

# ================= PREPROCESS =================
def preprocess(img):
    img = img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.stack([arr, arr, arr], axis=-1)
    return arr * 2.0 - 1.0

def postprocess(pred):
    img = ((pred + 1.0) * 127.5).astype("uint8")
    pil = Image.fromarray(img)
    return pil.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=2))

# ================= FACE LANDMARK UTILS =================
def extract_landmarks(pil_img):
    img = np.array(pil_img)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        num_faces=1
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_img = MPImage(image_format=ImageFormat.SRGB, data=img)
        result = landmarker.detect(mp_img)

        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

def draw_face_structure(pil_img):
    img = np.array(pil_img)
    lm = extract_landmarks(pil_img)

    if lm is None:
        return pil_img

    h, w, _ = img.shape
    for p in lm:
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(img, (x, y), 1, (0, 255, 255), -1)

    return Image.fromarray(img)

# ================= METRIC HELPERS =================
def compute_psnr(gt, pred):
    gt = np.array(gt).astype("float32")
    pred = np.array(pred).astype("float32")
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def landmark_distance(lm1, lm2):
    if lm1 is None or lm2 is None:
        return float("inf")
    return np.mean([
        np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        for p1, p2 in zip(lm1, lm2)
    ])

def landmark_accuracy(ref_lm, gen_lm):
    if ref_lm is None or gen_lm is None:
        return 0.0
    dist = landmark_distance(ref_lm, gen_lm)
    acc = max(0.0, 1.0 - dist * 8)  # empirical scaling
    return round(acc * 100, 2)

def ssim_score(gt, pred):
    gt_gray = cv2.cvtColor(np.array(gt), cv2.COLOR_RGB2GRAY)
    pred_gray = cv2.cvtColor(np.array(pred), cv2.COLOR_RGB2GRAY)
    return round(ssim(gt_gray, pred_gray) * 100, 2)

def edge_ssim_score(sketch, generated):
    sketch_gray = np.array(sketch.convert("L").resize((IMG_SIZE, IMG_SIZE)))
    gen_gray = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2GRAY)

    edges_gen = cv2.Canny(gen_gray, 100, 200)
    edges_sketch = cv2.Canny(sketch_gray, 100, 200)

    return round(ssim(edges_sketch, edges_gen) * 100, 2)

def face_validity(pil_img):
    return 100.0 if extract_landmarks(pil_img) else 0.0

# ================= GENERATION =================
def generate_images(model, sketch, n=3):
    x = preprocess(sketch)
    images = []

    for _ in range(n):
        noise = np.random.normal(0, 0.02, x.shape)
        pred = model.predict((x + noise)[None, ...], verbose=0)[0]
        images.append(postprocess(pred))

    return images

def select_best_image(ref_lm, images):
    scores = [landmark_distance(ref_lm, extract_landmarks(img)) for img in images]
    best_idx = int(np.argmin(scores))
    return images[best_idx]

# ================= FINAL EVALUATION =================
def evaluate(sketch, best_img, gt_img=None):
    ref_lm = extract_landmarks(sketch)
    gen_lm = extract_landmarks(best_img)

    lm_acc = landmark_accuracy(ref_lm, gen_lm)
    edge_val = edge_ssim_score(sketch, best_img)
    face_acc = face_validity(best_img)

    metrics = {
        "Landmark Accuracy (%)": lm_acc,
        "Edge SSIM (%)": edge_val,
        "Face Validity (%)": face_acc,
    }

    if gt_img is not None:
        psnr_val = compute_psnr(gt_img, best_img)
        ssim_val = ssim_score(gt_img, best_img)

        metrics["PSNR (dB)"] = round(psnr_val, 2)
        metrics["SSIM (%)"] = ssim_val

        final = (
            0.30 * lm_acc +
            0.25 * edge_val +
            0.20 * ssim_val +
            0.15 * face_acc +
            0.10 * min(psnr_val, 40) * 2.5
        )
    else:
        final = (
            0.45 * lm_acc +
            0.35 * edge_val +
            0.20 * face_acc
        )

    metrics["Final Score (%)"] = round(final, 2)
    return metrics

def make_json_safe(metrics: dict):
    safe_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            safe_metrics[k] = float(v)
        else:
            safe_metrics[k] = v
    return safe_metrics

def check_criminal_match(metrics):
    psnr = None
    ssim = None

    for k, v in metrics.items():
        if "PSNR" in k:
            psnr = float(v)
        if "SSIM" in k and "%" in k:
            ssim = float(v)

    if psnr is not None and ssim is not None:
        print(f"PSNR:{psnr} and SSIM:{ssim}")
        if psnr > 13 and ssim > 50:
            return True, psnr

    return False, psnr



def check_criminal_db(sketch,best_img):
    folder_path = "test/photos/"

    data_found = False
    img_dict={}
    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)

            img = Image.open(img_path)

            # processing
            print("Processing:", file)
            #gt_img = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            #img_path = os.path.join(img_dir, file)

            gt_img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                # -------- EVALUATION --------
            metrics = evaluate(sketch=sketch,best_img=best_img,gt_img=gt_img)
            
            is_match,img_psrn = check_criminal_match(metrics)
            if is_match:
                data_found=True
                img_dict[img_path]=img_psrn
    print(img_dict)
    best_psnr_img=None
    if data_found:
        best_psnr_img = max(img_dict, key=img_dict.get)
    return data_found, best_psnr_img, metrics