@ECHO OFF
docker build -f Dockerfile_local -t test:v1 .
docker run --rm -p 8880:8501 test:v1
