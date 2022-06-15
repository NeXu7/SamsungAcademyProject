import os
import streamlit as st
from stqdm import stqdm
import time
import pandas as pd
import numpy as np
import utils

st.title("Cell classifier")
model_select = st.selectbox(
    "Выберите модель классификатора",
    ["", "densenet121", "densenet201", "efficientnet_b0"]
)
if len(model_select) > 1:
    st.write(f"Производится классификация {model_select}")
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    model = utils.get_model(model_select)
    data = utils.get_data(img_folder_path="slides/test_cell_imgs/cell_img")
    result = utils.classify_cell(model, data)
    st.write(f"Готово!")
