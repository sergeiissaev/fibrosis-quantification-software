# -*- coding: utf-8 -*-
import base64
import logging
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def create_dataframe(tissue_final: float, fibrosis_final: float, csv_filename: str) -> None:
    data = [[tissue_final, fibrosis_final]]
    df = pd.DataFrame(data, columns=["tissue_percentage", "fibrosis_percentage"])
    df.to_csv(csv_filename, index=False)


def create_zip(*args, tissue_final: float, fibrosis_final: float) -> None:
    """Create zip file containing images and csv summary"""
    list_of_im_names = list()
    csv_filename = "fibrosis_results.csv"
    create_dataframe(tissue_final=tissue_final, fibrosis_final=fibrosis_final, csv_filename=csv_filename)
    list_of_im_names.append(csv_filename)
    name_list = ["tissue_vs_background", "original_image", "AI_output", "fibrotic_vs_nonfibrotic"]
    for idx in range(len(args)):
        image = args[idx]
        im_name = f"{name_list[idx]}.jpg"
        logging.info(f"Zipping up image {im_name}")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, st.uploaded_file_manager.UploadedFile):
            ZipfileDotZip = f"{Path(image.name).stem}.zip"
            image = Image.open(image)
        image.save(im_name)
        list_of_im_names.append(im_name)
    zipObj = ZipFile(ZipfileDotZip, "w")
    for file in list_of_im_names:
        zipObj.write(file)
    zipObj.close()

    with open(ZipfileDotZip, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}'>\
            Download results\
        </a>"
    st.sidebar.markdown(href, unsafe_allow_html=True)
