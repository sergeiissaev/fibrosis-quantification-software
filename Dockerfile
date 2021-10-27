FROM mambaorg/micromamba:0.15.3
USER root

RUN mkdir /opt/fibrosis_quantification_software
RUN chmod -R 777 /opt/fibrosis_quantification_software
WORKDIR /opt/fibrosis_quantification_software
USER micromamba

COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes

COPY setup.py setup.py
COPY tests tests
COPY fibrosis_quantification_software fibrosis_quantification_software
RUN pip install -e .

CMD python fibrosis_quantification_software/app/main.py