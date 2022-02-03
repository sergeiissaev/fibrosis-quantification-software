# -*- coding: utf-8 -*-
from prefect import Flow

import fibrosis_quantification_software_code.fibrosis_quantification.blob_removal as blob_removal
import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification as fibrosis_quantification

uploaded_file = 0
radio = 0


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
    clean_thresholded_fibrosis_nonfibrosis = fibrosis_quantification.clean_images(
        width, height, grid2d, threshgenner, thresh_tissue
    )
    tissue_final, fibrosis_final = fibrosis_quantification.report_fibrosis(
        patchwise_thresholded_tissue_nontissue, radio, clean_thresholded_fibrosis_nonfibrosis
    )
    output = blob_removal.blob_removal(radio)
flow.visualize()
state_1 = flow.run()
