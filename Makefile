USER_NAME := $(shell whoami)
IMAGE_NAME := viper
TAG_NAME := v1.0.0
CONTAINER_NAME := $(IMAGE_NAME)_container
GPU_ID := 0

UID := $(shell id -u)
GID := $(shell id -g)

WANDB_API_KEY := $(shell echo $$WANDB_API_KEY)
PATH_TO_NUSCENES := $(shell echo $$PATH_TO_NUSCENES)
PATH_TO_CITYSCAPES := $(shell echo $$PATH_TO_CITYSCAPES)
PATH_TO_URBANSYN := $(shell echo $$PATH_TO_URBANSYN)

define run_docker
	@docker run -it --rm \
		--net host \
		--gpus '"device=$(GPU_ID)"' \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(CONTAINER_NAME) \
		-u $(USER_NAME) \
		-v ./:/workspace \
		-v $(PATH_TO_URBANSYN):/data/UrbanSyn \
		-v $(PATH_TO_CITYSCAPES):/data/cityscapes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e TERM=xterm-256color \
		$(IMAGE_NAME):$(TAG_NAME) \
		/bin/bash -c $(1)
endef

.PHONY: build run attach jupyter clear
build:
	docker build . -t $(IMAGE_NAME):$(TAG_NAME) --build-arg USER=$(USER_NAME) --build-arg UID=$(UID) --build-arg GID=$(GID)
	@echo "\nBuild complete!"
	@echo "Run 'make run' to start the container."

run:
	$(call run_docker, "source entrypoint.sh && bash")

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash -c bash

jupyter:
	$(call run_docker, "jupyter notebook")

clear:
	@rm -rf .venv/
	@rm -rf viper_dev.egg-info/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleaned up the project directory."