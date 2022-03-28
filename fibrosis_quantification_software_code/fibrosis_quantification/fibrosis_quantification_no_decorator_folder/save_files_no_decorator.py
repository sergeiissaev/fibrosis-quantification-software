# -*- coding: utf-8 -*-
import base64
import logging
from zipfile import ZipFile

import numpy as np
import streamlit as st
from PIL import Image


def create_zip(*args) -> None:
    """Create zip file containing images and csv summary"""
    list_of_im_names = list()
    name_list = ["thresholded image.jpg", "original_image.jpg"]
    ZipfileDotZip = "sample.zip"
    for idx in range(len(args)):
        image = args[idx]
        im_name = f"{name_list[idx]}.jpg"
        logging.info(f"Zipping up image {im_name}")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, st.uploaded_file_manager.UploadedFile):
            image = Image.open(image)
        image.save(im_name)
        list_of_im_names.append(im_name)

    zipObj = ZipFile("sample.zip", "w")
    for im_name in list_of_im_names:
        zipObj.write(im_name)
    zipObj.close()

    with open(ZipfileDotZip, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
            Click last model weights\
        </a>"
    st.sidebar.markdown(href, unsafe_allow_html=True)
