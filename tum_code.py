import streamlit as st
from PIL import Image
import numpy as np

st.title("Automatic Image Detection")

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(
        "https://raw.githubusercontent.com/LarrissaSilva/imagedetection/main/yolo26n.pt"
    )

model = load_model()

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    img = Image.open(uploaded_file)
    st.image(img)

    if st.button("Detect"):

        import cv2

        img_cv = cv2.cvtColor(
            np.array(img),
            cv2.COLOR_RGB2BGR
        )

        results = model.predict(img_cv)

        plotted = results[0].plot()[:, :, ::-1]
        st.image(plotted)
