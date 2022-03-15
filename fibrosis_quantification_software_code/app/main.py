# -*- coding: utf-8 -*-
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import streamlit as st
from art import tprint
from prefect import Flow

import fibrosis_quantification_software_code.fibrosis_quantification.blob_removal as blob_removal
import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification as fibrosis_quantification
from fibrosis_quantification_software_code.app.footer import footer


def main():
    st.title("Fibrosis Quantification Software")
    st.header("Created by Sergei Issaev, with the supervision of Dr. Roger Tam and Dr. Fabio Rossi")

    footer()

    uploaded_file = st.file_uploader("Upload your histology image (PSR stained, 10x microscopy)", type=["png", "jpg"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        radio = st.radio("Patch or whole slide image?", ["WSI", "Patch"])
        print(file_details, radio)
        clicked = st.button("Calculate")
        if clicked:
            st.markdown("<h5 style='text-align: center;'>Original image</h1>", unsafe_allow_html=True)
            st.image(uploaded_file)
            with Flow("fibrosis-quant-flow") as flow:
                model = fibrosis_quantification.import_model()
                (
                    num_samples,
                    im1_preprocess_blocks,
                    img_preprocess_blocks_255,
                    width,
                    height,
                    patchwise_thresholded_tissue_nontissue,
                ) = fibrosis_quantification.preliminary_preprocessing(uploaded_file, radio)
                grid2d, threshgenner, thresh_tissue = fibrosis_quantification.apply_gan(
                    num_samples, model, im1_preprocess_blocks, img_preprocess_blocks_255, width, height
                )
                clean_thresholded_fibrosis_nonfibrosis, remove = fibrosis_quantification.clean_images(
                    width, height, grid2d, threshgenner, thresh_tissue
                )
                patchwise_thresholded_tissue_nontissue = blob_removal.blob_removal(
                    radio, patchwise_thresholded_tissue_nontissue, remove, num_samples, height, width
                )
                tissue_final, fibrosis_final = fibrosis_quantification.report_fibrosis(
                    patchwise_thresholded_tissue_nontissue, radio, clean_thresholded_fibrosis_nonfibrosis
                )

            state_1 = flow.run()
            # print(f"state 1 is {state_1}")
            # for task in flow.get_tasks():
            #     if task.name == "preliminary preprocessing":
            #         processing_task = task
            # task_1_source_image, task_1_result_image = state_1.result[processing_task].result

            if state_1.message == "All reference tasks succeeded.":
                tprint("Success")


if __name__ == "__main__":
    main()
