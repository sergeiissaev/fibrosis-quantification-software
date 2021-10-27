#!/usr/bin/env bash
PROJECT=fibrosis_quantification_software

# Default commands
build: docker_build

docker_build:
	docker build . -t $(PROJECT)

docker_run:
	docker run -it --network host -v $(shell pwd):/opt/fibrosis_quantification_software $(PROJECT)

docker_bash:
	docker run -it --network host -v $(shell pwd):/opt/fibrosis_quantification_software $(PROJECT) bash

docker_test:
	docker run -it --network host -v $(shell pwd):/opt/fibrosis_quantification_software $(PROJECT) pytest --cov=fibrosis_quantification_software


deploy:
	docker-compose up -d --build

docker_down:
	docker-compose down