# -*- coding: utf-8 -*-
from datetime import timedelta

from prefect import task


@task(name="callback", max_retries=3, retry_delay=timedelta(seconds=10), nout=2)
def callback(uploaded_file, radio):

    return uploaded_file, radio
