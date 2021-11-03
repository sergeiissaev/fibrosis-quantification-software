# -*- coding: utf-8 -*-
from datetime import timedelta

from prefect import task


@task(name="Hello World", max_retries=3, retry_delay=timedelta(seconds=10))
def hello_world():
    return "100% Fibrosis"
