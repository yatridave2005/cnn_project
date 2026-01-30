import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Currency Detector")

st.title("💸 Real vs Fake Currency Detection")

# Load model
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()




IMG_SIZE = (1,224, 224,3)

uploaded_file = st.file_uploader(
    "Upload currency image (jpg / png only)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.astype(np.float32)


    img_array = np.expand_dims(img_array, axis=0)
   



    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    st.write("Prediction:", pred)


    score = float(pred[0][0])

    st.subheader("Result")

    if score > 0.5:
        st.error("❌ Fake Currency Note")
    else:
        st.success("✅ Real Currency Note")

    st.write("Confidence:", round(score, 3))
