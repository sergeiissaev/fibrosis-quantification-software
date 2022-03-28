# -*- coding: utf-8 -*-
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import streamlit as st
from art import tprint

from fibrosis_quantification_software_code.app.flow import create_flow
from fibrosis_quantification_software_code.app.footer import footer


def main():
    st.title("Fibrosis Quantification Software")
    st.header("Created by Sergei Issaev, with the supervision of Dr. Roger Tam and Dr. Fabio Rossi")

    footer()

    uploaded_file = st.file_uploader("Upload your histology image (PSR stained, 10x microscopy)", type=["png", "jpg"])

    if uploaded_file is not None:
        radio = st.radio("Patch or whole slide image?", ["WSI", "Patch"])
        clicked = st.button("Calculate")
        if clicked:
            st.markdown("<h5 style='text-align: center;'>Original image</h1>", unsafe_allow_html=True)
            st.image(uploaded_file)
            flow, patchwise_thresholded_tissue_nontissue = create_flow(patch_or_wsi=radio, uploaded_file=uploaded_file)
            state_1 = flow.run()
            if state_1.message == "All reference tasks succeeded.":
                tprint("Success")


if __name__ == "__main__":
    main()
