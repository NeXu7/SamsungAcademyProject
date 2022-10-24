import os

import streamlit as st
import cv2 as cv
import numpy as np

import json

from utils.utils import inference

# from stqdm import stqdm
# import pandas as pd
# import numpy as np
# import utils.utils as utils
# import json


ccd = {4: 16711680, 0: 255, 2: (255, 153, 0), 3: (255, 0, 0), 1: (255, 255, 0)}
st.sidebar.image("misc/LOGO.png", width=100)
st.sidebar.title("Cell classifier")

# image = location.file_uploader(label="Регион интереса")
image = None
if image is None:
    image = st.file_uploader(label="Регион интереса")
location = st.empty()
# with open("test_mask.json", "r") as f:
#     masks = json.load(f)

if image:
    model_select = st.sidebar.selectbox(
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

    if model_select == "MASK":
        location.image(cv.addWeighted(image, 0.9, image_mask, 0.5, 0.0))

    elif model_select == "HEATMAP":
        blur_rendered *= 255 / blur_rendered.max((0, 1))
        blur_rendered = cv.applyColorMap(
            blur_rendered.astype(np.uint8), cv.COLORMAP_JET
        )
        location.image(cv.addWeighted(image, 0.9, blur_rendered, 0.9, 0.0))
    else:
        location.image(image)

    #
    #

    #
    # fig, ax = plt.subplots()
    # plt.imshow(blur_rendered, cmap="hot")
    # st.write(fig)
    #
    # st.image(cv.addWeighted(image, 0.9, blur_rendered, 0.9, 0.0))

st.button("Re-run")

#     print(inference(image))

# print(type(cv.()))
# model_select = st.selectbox(
#     "Choose model architecture:",
#     ["", "densenet121", "densenet201", "efficientnet_b0"]
# )
# if len(model_select) > 1:
#     model = utils.get_model(model_select)
#     model.eval()
# finish = False
# uploaded_file = None
# load_file = st.checkbox("Classification from geojson?", key=0)
# # download_prob = st.checkbox("Result as probability?", key=1)
# if load_file:
#     uploaded_file = st.file_uploader("Choose a file")
#
# if uploaded_file is not None:
#     uploaded_file = pd.read_json(uploaded_file)
#     download_mask = uploaded_file.copy()
#
#
# if len(model_select) > 1:
#     if uploaded_file is not None:
#         uploaded_file = utils.contour_ext(uploaded_file)
#         st.write(f"{model_select}-classification in progress")
#         progress_bar = st.sidebar.progress(0)
#         status_text = st.sidebar.empty()
#
#         data = utils.get_data(slide_path="slides/5189/TCGA-BP-5189-01Z-00-DX1.cfffe175-da76-4edd-9187-0570a877127b.svs",
#                               contour_data=uploaded_file)
#         result = utils.classify_cell(model, data)
#         st.write(f"Готово!")
#         finish = True
#     elif not load_file:
#         st.write(f"{model_select}-classification in progress")
#         progress_bar = st.sidebar.progress(0)
#         status_text = st.sidebar.empty()
#         data = utils.get_data(img_folder_path="slides/test_cell_imgs/cell_img")
#         result = utils.classify_cell(model, data)
#         st.write(f"Готово!")
#         finish = True
#
#     if finish:
#         if uploaded_file is not None:
#             file_name = st.text_input("Input file name")
#             new_data = utils.make_predict_file(download_mask, result)
#             new_data = [new_data.iloc[i].to_dict() for i in range(len(new_data))]
#             with open(f'{file_name}.json', 'w') as f:
#                 json.dump(new_data, f)
#         else:
#             pd.DataFrame([os.listdir("slides/test_cell_imgs/cell_img"),
#                           result.argmax(0)], columns=['file_name', 'predict'])
#         # elif download_prob:
#         #     st.download_button("Download result", result)
