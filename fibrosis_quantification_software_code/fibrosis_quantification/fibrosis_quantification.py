# -*- coding: utf-8 -*-
from datetime import timedelta

from prefect import task

import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator_folder.fibrosis_quantification_no_decorator as fibrosis_quantification_no_decorator


@task(name="load model", max_retries=3, retry_delay=timedelta(seconds=10), nout=1)
def import_model():
    model = fibrosis_quantification_no_decorator.import_model()
    return model


@task(name="preliminary preprocessing", max_retries=3, retry_delay=timedelta(seconds=10), nout=6)
def preliminary_preprocessing(source_file, radio):
    (
        num_samples,
        im1_preprocess_blocks,
        img_preprocess_blocks_255,
        width,
        height,
        patchwise_thresholded_tissue_nontissue,
    ) = fibrosis_quantification_no_decorator.preliminary_preprocessing(source_file, radio)
    return (
        num_samples,
        im1_preprocess_blocks,
        img_preprocess_blocks_255,
        width,
        height,
        patchwise_thresholded_tissue_nontissue,
    )


@task(name="Applying GAN", max_retries=3, retry_delay=timedelta(seconds=10), nout=3)
def apply_gan(num_samples, model, im1_preprocess_blocks, img_preprocess_blocks_255, width, height):
    grid2d, threshgenner, thresh_tissue = fibrosis_quantification_no_decorator.apply_gan(
        num_samples,
        model,
        im1_preprocess_blocks,
        img_preprocess_blocks_255,
        width,
        height,
    )
    return grid2d, threshgenner, thresh_tissue


@task(name="Cleaning generated images", max_retries=3, retry_delay=timedelta(seconds=10), nout=1)
def clean_images(width: int, height: int, grid2d: list, threshgenner, thresh_tissue):
    clean_thresholded_fibrosis_nonfibrosis = fibrosis_quantification_no_decorator.clean_images(
        width, height, grid2d, threshgenner, thresh_tissue
    )
    return clean_thresholded_fibrosis_nonfibrosis


@task(name="Reporting fibrosis", max_retries=3, retry_delay=timedelta(seconds=10), nout=2)
def report_fibrosis(patchwise_thresholded_tissue_nontissue, radio, clean_thresholded_fibrosis_nonfibrosis):
    tissue_final, fibrosis_final = fibrosis_quantification_no_decorator.report_fibrosis(
        patchwise_thresholded_tissue_nontissue, radio, clean_thresholded_fibrosis_nonfibrosis
    )
    return tissue_final, fibrosis_final
