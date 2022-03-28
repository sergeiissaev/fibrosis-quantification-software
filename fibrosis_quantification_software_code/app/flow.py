# -*- coding: utf-8 -*-
import streamlit as st
from prefect import Flow

import fibrosis_quantification_software_code.fibrosis_quantification.blob_removal as blob_removal
import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification as fibrosis_quantification
import fibrosis_quantification_software_code.fibrosis_quantification.save_files as save_files


def create_flow(patch_or_wsi: str, uploaded_file: st.uploaded_file_manager.UploadedFile):
    with Flow("fibrosis-quant-flow") as flow:
        model = fibrosis_quantification.import_model()
        (
            num_samples,
            im1_preprocess_blocks,
            img_preprocess_blocks_255,
            width,
            height,
            patchwise_thresholded_tissue_nontissue,
        ) = fibrosis_quantification.preliminary_preprocessing(uploaded_file, patch_or_wsi)
        grid2d, threshgenner, thresh_tissue = fibrosis_quantification.apply_gan(
            num_samples, model, im1_preprocess_blocks, img_preprocess_blocks_255, width, height
        )
        clean_thresholded_fibrosis_nonfibrosis, remove = fibrosis_quantification.clean_images(
            width, height, grid2d, threshgenner, thresh_tissue
        )
        patchwise_thresholded_tissue_nontissue = blob_removal.blob_removal(
            patch_or_wsi, patchwise_thresholded_tissue_nontissue, remove, num_samples, height, width
        )
        tissue_final, fibrosis_final = fibrosis_quantification.report_fibrosis(
            patchwise_thresholded_tissue_nontissue, patch_or_wsi, clean_thresholded_fibrosis_nonfibrosis
        )
        save_files.create_zip(
            patchwise_thresholded_tissue_nontissue=patchwise_thresholded_tissue_nontissue, uploaded_file=uploaded_file
        )

    return flow, patchwise_thresholded_tissue_nontissue


if __name__ == "__main__":
    flow, patchwise_thresholded_tissue_nontissue = create_flow(patch_or_wsi="patch", uploaded_file=0)
    flow.visualize()
