#!/bin/bash -e


image_name=docker-registry.linecorp.com/lp60409/tfrec
image_tag=0.3

full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"

docker build -t "${full_image_name}" .
docker push "$full_image_name"
