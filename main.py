import os

import streamlit as st
from stqdm import stqdm
import pandas as pd
import numpy as np
import utils.utils as utils
import json

st.title("Cell classifier")
model_select = st.selectbox(
    "Choose model architecture:",
    ["", "densenet121", "densenet201", "efficientnet_b0"]
)
model = utils.get_model(model_select)
model.eval()
finish = False
uploaded_file = None
load_file = st.checkbox("Classification from geojson?", key=0)
# download_prob = st.checkbox("Result as probability?", key=1)
if load_file:
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    uploaded_file = pd.read_json(uploaded_file)
    download_mask = uploaded_file.copy()


if len(model_select) > 1:
    if uploaded_file is not None:
        uploaded_file = utils.contour_ext(uploaded_file)
        st.write(f"{model_select}-classification in progress")
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        data = utils.get_data(slide_path="slides/5189/TCGA-BP-5189-01Z-00-DX1.cfffe175-da76-4edd-9187-0570a877127b.svs",
                              contour_data=uploaded_file)
        result = utils.classify_cell(model, data)
        st.write(f"Готово!")
        finish = True
    elif not load_file:
        st.write(f"{model_select}-classification in progress")
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        data = utils.get_data(img_folder_path="slides/test_cell_imgs/cell_img")
        result = utils.classify_cell(model, data)
        st.write(f"Готово!")
        finish = True

    if finish:
        if uploaded_file is not None:
            file_name = st.text_input("Input file name")
            new_data = utils.make_predict_file(download_mask, result)
            new_data = [new_data.iloc[i].to_dict() for i in range(len(new_data))]
            with open(f'{file_name}.json', 'w') as f:
                json.dump(new_data, f)
        else:
            pd.DataFrame([os.listdir("slides/test_cell_imgs/cell_img"),
                          result.argmax(0)], columns=['file_name', 'predict'])
        # elif download_prob:
        #     st.download_button("Download result", result)
st.button("Re-run")
