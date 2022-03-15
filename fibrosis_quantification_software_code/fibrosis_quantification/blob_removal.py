# -*- coding: utf-8 -*-
from datetime import timedelta

from prefect import task

import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator_folder.blob_removal_no_decorator as blob_removal_no_decorator


@task(name="Removing blobs", max_retries=3, retry_delay=timedelta(seconds=10), nout=1)
def blob_removal(radio, patchwise_thresholded_tissue_nontissue, remove, samples, height, width):
    (patchwise_thresholded_tissue_nontissue) = blob_removal_no_decorator.blob_removal(
        radio, patchwise_thresholded_tissue_nontissue, remove, samples, height, width
    )
    return patchwise_thresholded_tissue_nontissue
