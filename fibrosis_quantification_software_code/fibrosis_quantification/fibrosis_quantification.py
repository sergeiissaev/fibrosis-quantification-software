# -*- coding: utf-8 -*-
from datetime import timedelta

from prefect import task

import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator.fibrosis_quantification_no_decorator as fibrosis_quantification_no_decorator


@task(name="load model", max_retries=3, retry_delay=timedelta(seconds=10), nout=1)
def import_model():
    model = fibrosis_quantification_no_decorator.import_model()
    return model


@task(name="preliminary preprocessing", max_retries=3, retry_delay=timedelta(seconds=10), nout=3)
def preliminary_preprocessing(source_file):
    (
        num_samples,
        im1_preprocess_blocks,
        img_preprocess_blocks_255,
    ) = fibrosis_quantification_no_decorator.preliminary_preprocessing(source_file)
    return num_samples, im1_preprocess_blocks, img_preprocess_blocks_255


@task(name="Applying GAN", max_retries=3, retry_delay=timedelta(seconds=10), nout=1)
def apply_gan(num_samples, model, im1_preprocess_blocks, img_preprocess_blocks_255):
    _ = fibrosis_quantification_no_decorator.apply_gan(
        num_samples, model, im1_preprocess_blocks, img_preprocess_blocks_255
    )
    return None
