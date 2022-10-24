import pandas as pd
import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import inference

ccd = {4: 16711680, 0: 255, 2: (255, 153, 0), 3: (255, 0, 0), 1: (255, 255, 0)}
st.sidebar.image("misc/LOGO.png", width=100)
st.sidebar.title("Cell classifier")

image = None
masks = None
if image is None:
    image = st.file_uploader(label="Регион интереса")
location = st.empty()

if image:
    view_select = st.sidebar.selectbox(
        "ВАРИАНТЫ ПРОСМОТРА", ["STANDARD", "MASK", "HEATMAP"]
    )
    location.image(image)

    image = np.fromstring(image.getvalue(), np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    masks = inference(image)
    image_mask = np.zeros_like(image)
    blur = np.zeros(image_mask.shape[:2])
    for detection in masks:
        label = detection["label"]
        mask = detection["detection"]
        mask = np.array([mask], dtype=np.int64)
        cv.drawContours(image_mask, mask, -1, ccd[label], -1)
        x, y, w, h = cv.boundingRect(mask)
        blur[y + h // 2, x + w // 2] = label

    blur_rendered = blur.copy()
    for i in range(100):
        blur_rendered += (blur + blur_rendered) / 2
        blur_rendered = cv.GaussianBlur(blur_rendered, (15, 15), 3, cv.BORDER_DEFAULT)

    if view_select == "MASK":
        location.image(cv.addWeighted(image, 0.9, image_mask, 0.5, 0.0))

    elif view_select == "HEATMAP":
        blur_rendered *= 255 / blur_rendered.max((0, 1))
        blur_rendered = cv.applyColorMap(
            blur_rendered.astype(np.uint8), cv.COLORMAP_JET
        )
        location.image(cv.addWeighted(image, 0.9, blur_rendered, 0.9, 0.0))
    else:
        location.image(image)

#
if masks is not None:
    hist = st.checkbox("HIST")
    if hist:
        masks = pd.DataFrame(masks).dropna()
        fig, ax = plt.subplots()

        bins = np.arange(0, 5, 1) - 0.5
        ax.hist(masks["label"].values, bins)

        ax.set_xticks(bins + 0.5)
        st.pyplot(fig)
st.button("Re-run")
