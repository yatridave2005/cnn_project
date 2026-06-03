import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Currency Detector")

st.title("💸 Real vs Fake Currency Detection")

model = tf.keras.models.load_model("currency_detector.keras")


uploaded_file = st.file_uploader(
    "Upload currency image (jpg / png only)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)

img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    score = float(pred[0][0])

    st.subheader("Result")

   if score > 0.5:
    st.success("✅ Real Currency Note")
   else:
    st.error("❌ Fake Currency Note")

    st.write("Confidence:", round(score, 3))
