import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

from src.model import build_unet_generator
from src.gen_eval_image import (
    draw_face_structure,
    generate_images,
    select_best_image,
    evaluate,
    extract_landmarks,
    make_json_safe,
    check_criminal_match,
    check_criminal_db
)

# ================= CONFIG =================
IMG_SIZE = 512
GEN_MODEL_PATH = "models/best_generator.h5"

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FaceSketchX",
    page_icon="🧠",
    layout="wide",
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 FaceSketchX</h1>
    <p style='text-align:center; font-size:18px;'>
        Landmark-Aware Sketch-to-Face Generation & Evaluation
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("Upload inputs & check criminal record")

    sketch_file = st.file_uploader(
        "✏️ Upload Sketch Image",
        ["png", "jpg", "jpeg"],
    )

    gt_file = st.button(
        "🖼️ Check Criminal Record DB",
        use_container_width=True
    )

    #generate_btn = st.button("🚀 Show DB Image", use_container_width=True)

    st.markdown("---")
    st.caption("© FaceSketchX | Research Demo")

# ================= LOAD MODEL =================
if "generator" not in st.session_state:
    with st.spinner("Loading model..."):
        model = build_unet_generator(IMG_SIZE)
        model.load_weights(GEN_MODEL_PATH)
        st.session_state.generator = model
    st.success("Model loaded")

# ================= RADAR CHART =================
def plot_radar(metrics):
    labels = list(metrics.keys())
    values = list(metrics.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values += values[:1]
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 100)
    ax.set_title("Evaluation Radar Chart", pad=20)

    return fig

# ================= MAIN =================
if sketch_file:
    sketch = Image.open(sketch_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    # -------- INPUT SECTION --------
    st.subheader("📥 Input Overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Sketch Input**")
        st.image(sketch, use_container_width=True)

    with c2:
        st.markdown("**Reference Landmarks**")
        st.image(draw_face_structure(sketch), use_container_width=True)

    # -------- GENERATION --------
    st.markdown("---")
    st.subheader("🎨 Candidate Generation")

    with st.spinner("Generating faces..."):
        images = generate_images(st.session_state.generator, sketch)

    cols = st.columns(3)
    for i, img in enumerate(images):
        cols[i].image(img, caption=f"Candidate {i+1}", use_container_width=True)

    # -------- BEST IMAGE --------
    st.markdown("---")
    st.subheader("🏆 Best Image Selection")

    best_img = select_best_image(extract_landmarks(sketch), images)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Final Output**")
        st.image(best_img, use_container_width=True)

    with cB:
        st.markdown("**Facial Landmarks**")
        st.image(draw_face_structure(best_img), use_container_width=True)


    st.markdown("---")
    st.subheader("📊 Criminal Image Result")

    # -------- CRIMINAL MATCH CHECK --------

    if gt_file:
        data_found, criminal_image, metrics=check_criminal_db(sketch=sketch, best_img=best_img)


        if data_found:
            st.success(
                "🚨 **CRIMINAL IMAGE MATCHED**\n\n"
                "The generated face does match the provided criminal image.\n\n"
                "**Reason:**\n"
                "- PSNR > 14 dB\n"
                "- SSIM > 50%\n\n"
            )
                #st.stop()  # ⛔ Stop further processing (radar, downloads, etc.)


            # Display metrics
            st.subheader("📊 Sketch Match result with DB")


            mcols = st.columns(len(metrics))
            for col, (k, v) in zip(mcols, metrics.items()):
                col.metric(k, v)

            # -------- RADAR CHART --------
            st.markdown("### 📈 Metric Radar Visualization")
            radar_metrics = {
                k.replace("(%)", "").replace("(dB)", ""): min(v, 100)
                for k, v in metrics.items()
                if "%" in k
            }

            st.pyplot(plot_radar(radar_metrics))

            # -------- DOWNLOADS --------
            st.markdown("---")
            st.subheader("⬇️ Downloads")

            img = Image.open(criminal_image)
            img_bytes = img.tobytes()
            st.image(img)

            st.download_button(
                "Download Generated Image",
                data=img_bytes,
                file_name="facesketchx_output.jpg",
                mime="image/jpg",
            )

            json_safe_metrics = make_json_safe(metrics)

            st.download_button(
                "Download Metrics (JSON)",
                data=json.dumps(json_safe_metrics, indent=4),
                file_name="facesketchx_metrics.json",
                mime="application/json",
            )


            st.success("Process completed successfully")
        else:
            st.success(
                "**CRIMINAL IMAGE NOT MATCHED**\n\n"
                "The generated face does not match the provided criminal image.\n\n"
                "**Reason:**\n"
                "- PSNR < 14 dB\n"
                "- SSIM < 50%\n\n"
                "⚠️ Please verify the sketch or provide a clearer reference image."
            )

else:
    st.info("⬅️ Upload a sketch image from the sidebar to begin.")
