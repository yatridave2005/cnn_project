import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Currency Detector")

st.title("💸 Real vs Fake Currency Detection")

# Load model
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


IMG_SIZE = (224, 224)

uploaded_file = st.file_uploader(
    "Upload currency image (jpg / png only)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    score = float(pred[0][0])

    st.subheader("Result")

    if score > 0.5:
        st.error("❌ Fake Currency Note")
    else:
        st.success("✅ Real Currency Note")

    st.write("Confidence:", round(score, 3))
