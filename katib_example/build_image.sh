#!/bin/bash -e

image_name=docker-registry.linecorp.com/lp60409/katib_example
image_tag=latest

full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")" 

sudo docker build -t "${full_image_name}" .
sudo docker push "$full_image_name"

