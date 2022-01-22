FROM mambaorg/micromamba:0.15.3
USER root
RUN mkdir /opt/fibrosis_quantification_software
RUN chmod -R 777 /opt/fibrosis_quantification_software
WORKDIR /opt/fibrosis_quantification_software
EXPOSE 8501
EXPOSE 80
USER micromamba

COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes
COPY setup.py setup.py
COPY tests tests
COPY models models
COPY fibrosis_quantification_software_code fibrosis_quantification_software_code
RUN pip install -e .
#CMD streamlit run fibrosis_quantification_software_code/app/main.py --server.port $PORT
CMD streamlit run fibrosis_quantification_software_code/app/main.py
