IMAGE_NAME=recommender
IMAGE_VERSION=0.2
CONTAINER_NAME=myRec

build_image:
	s2i build . seldonio/seldon-core-s2i-python3:0.10 ${IMAGE_NAME}:${IMAGE_VERSION}

push_image:
	docker push $(IMAGE_NAME):$(IMAGE_VERSION)

exec:
	docker exec -it ${CONTAINER_NAME}_${IMAGE_VERSION} /bin/bash


update:
	docker stop ${CONTAINER_NAME}_${IMAGE_VERSION}
	docker rmi ${IMAGE_NAME}:${IMAGE_VERSION}
	make build_image
	make run

delete:
	docker rmi ${IMAGE_NAME}:${IMAGE_VERSION}

run:
	docker run --name ${CONTAINER_NAME}_${IMAGE_VERSION} -d --rm -p 5000:5000 ${IMAGE_NAME}:${IMAGE_VERSION}
	seldon-core-tester contract.json 0.0.0.0 5000 -p


.PHONY: train
train:
	python train.py

test:
	python test.py
.PHONY: clean
clean:
	rm -rf lastfm_data
