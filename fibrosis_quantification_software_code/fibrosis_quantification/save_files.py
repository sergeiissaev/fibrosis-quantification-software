# -*- coding: utf-8 -*-
from datetime import timedelta

import numpy as np
import streamlit as st
from prefect import task

import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator_folder.save_files_no_decorator as save_files_no_decorator


@task(name="Saving as zip file", max_retries=3, retry_delay=timedelta(seconds=10), nout=1)
def create_zip(
    patchwise_thresholded_tissue_nontissue: np.ndarray,
    uploaded_file: st.uploaded_file_manager.UploadedFile,
    generated_image: np.ndarray,
    clean_thresholded_fibrosis_nonfibrosis: np.ndarray,
    tissue_final: float,
    fibrosis_final: float,
):
    save_files_no_decorator.create_zip(
        patchwise_thresholded_tissue_nontissue,
        uploaded_file,
        generated_image,
        clean_thresholded_fibrosis_nonfibrosis,
        tissue_final=tissue_final,
        fibrosis_final=fibrosis_final,
    )
