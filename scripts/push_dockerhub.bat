@ECHO OFF
docker build -f Dockerfile_local -t sergei740/fibrosis_quantification_image:latest .
docker push sergei740/fibrosis_quantification_image:latest
