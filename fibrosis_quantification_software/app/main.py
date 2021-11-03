# -*- coding: utf-8 -*-

import fibrosis_quantification
import streamlit as st
from prefect import Flow

st.title("Fibrosis Quantification Software")
st.header("Created by Sergei Issaev, with the supervision of Dr. Roger Tam and Dr. Fabio Rossi")


uploaded_file = st.file_uploader("Upload your histology image (PSR stained, 10x microscopy)", type=["png", "jpg"])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    radio = st.radio("Patch or whole slide image?", ["WSI", "Patch"])
    clicked = st.button("Calculate")
    if clicked:
        with Flow("fibrosis-quant-flow") as flow:
            text = fibrosis_quantification.hello_world()
        state_1 = flow.run()
        task_ref = flow.get_tasks()[0]
        st.write(state_1.result[task_ref].result)
