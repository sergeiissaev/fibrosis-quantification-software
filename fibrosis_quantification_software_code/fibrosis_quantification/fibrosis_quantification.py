# -*- coding: utf-8 -*-
from datetime import timedelta

from prefect import task

import fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator.fibrosis_quantification_no_decorator as fibrosis_quantification_no_decorator


@task(name="preliminary preprocessing", max_retries=3, retry_delay=timedelta(seconds=10), nout=2)
def preliminary_preprocessing(source_file, model):
    source_file, img_array = fibrosis_quantification_no_decorator.preliminary_preprocessing(source_file, model)
    return source_file, img_array


@task(name="load model", max_retries=3, retry_delay=timedelta(seconds=10), nout=2)
def import_model():
    model = fibrosis_quantification_no_decorator.import_model()
    return model
