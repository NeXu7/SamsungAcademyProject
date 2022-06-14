import os
import streamlit as st
import time
import pandas as pd
import numpy as np
import utils

st.title("Cell classifier")
model = st.selectbox(
    "Выберите модель классификатора",
    ["", "densenet121", "densenet201", "efficientnet_b0"]
)
if len(model) > 1:
    st.write(f"Производится классификация {model}")
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    m = utils.get_model(model)
    for i in range(1, 101):

        status_text.text("%i%% Complete" % i)
        progress_bar.progress(i)
        time.sleep(0.05)
    st.write(f"Готово!")
