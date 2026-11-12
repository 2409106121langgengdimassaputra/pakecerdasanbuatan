import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (128, 128)  # harus sama dengan ukuran saat training

st.set_page_config(page_title="Klasifikasi Beras", page_icon="🌾", layout="centered")


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("rice_cnn.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


def predict(model, image: Image.Image):
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    arr = tf.image.resize(arr, IMG_SIZE)  # sama seperti pipeline training (piksel 0-255)
    arr = tf.expand_dims(arr, 0)
    return model.predict(arr, verbose=0)[0]


st.title("🌾 Klasifikasi Citra Butiran Beras")
st.write(
    "Unggah foto butiran beras, lalu model **Custom CNN** akan memprediksi jenisnya: "
    "Arborio, Basmati, Ipsala, Jasmine, atau Karacadag."
)

model, class_names = load_model_and_classes()

file = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    with st.spinner("Memproses..."):
        prob = predict(model, image)

    top = int(np.argmax(prob))
    st.success(f"Prediksi: **{class_names[top]}** (keyakinan {prob[top] * 100:.2f}%)")

    st.subheader("Probabilitas tiap kelas")
    st.bar_chart({k: float(p) for k, p in zip(class_names, prob)})
    for k, p in sorted(zip(class_names, prob), key=lambda x: -x[1]):
        st.write(f"{k}: {p * 100:.2f}%")

    if prob[top] < 0.6:
        st.warning(
            "Keyakinan model rendah. Pastikan foto berisi satu butir beras dengan "
            "latar polos dan pencahayaan cukup, mirip data latih."
        )
else:
    st.info("Belum ada gambar yang diunggah.")

st.caption("PA Kecerdasan Buatan - Kelas C Kelompok 2 | Dataset: Rice Image Dataset (Kaggle)")
