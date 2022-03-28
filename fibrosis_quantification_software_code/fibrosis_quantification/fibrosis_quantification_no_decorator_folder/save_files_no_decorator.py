# -*- coding: utf-8 -*-
import base64
import logging
from zipfile import ZipFile

import numpy as np
import streamlit as st
from PIL import Image


def create_zip(patchwise_thresholded_tissue_nontissue: np.ndarray):
    logging.info(type(patchwise_thresholded_tissue_nontissue))
    img = Image.fromarray(patchwise_thresholded_tissue_nontissue)
    img.save("test.jpg")
    zipObj = ZipFile("sample.zip", "w")
    # Add multiple files to the zip
    zipObj.write("test.jpg")
    # close the Zip File
    zipObj.close()
    ZipfileDotZip = "sample.zip"
    with open(ZipfileDotZip, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
            Click last model weights\
        </a>"
    st.sidebar.markdown(href, unsafe_allow_html=True)
