IMAGE_NAME=katib_example
IMAGE_VERSION=0.1
CONTAINER_NAME=

build_image:
	s2i build . seldonio/seldon-core-s2i-python3:0.10 ${IMAGE_NAME}:${IMAGE_VERSION}

push_image:
	docker push $(IMAGE_NAME):$(IMAGE_VERSION)
