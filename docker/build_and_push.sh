#!/bin/bash

# Stop the script if any command fails
set -e

# === Argument Parsing ===
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: $0 <dockerhub-user> <image-name> <image-tag>"
    echo "Example: $0 myusername 3dgsworkspace latest"
    exit 1
fi

DOCKERHUB_USER="$1"
IMAGE_NAME="$2"
IMAGE_TAG="$3"

# Define the Dockerfile path based on the image name
cd ${IMAGE_NAME}
DOCKERFILE_PATH="Dockerfile"

# Check if the specified Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "ERROR: Dockerfile not found."
    echo "Based on the image name '$IMAGE_NAME', this script expected to find a file named: $DOCKERFILE_PATH"
    exit 1
fi

# Define the local and remote image tags
LOCAL_IMAGE_TAG="${IMAGE_NAME}-build:${IMAGE_TAG}"
REMOTE_IMAGE_TAG="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

# 1. Build the Docker image
echo "Building Docker image from: $DOCKERFILE_PATH"
# We tag it with a local name first
docker build -t $LOCAL_IMAGE_TAG -f $DOCKERFILE_PATH .
echo "Build complete."

# 2. Tag the image for Docker Hub
echo "Tagging image for Docker Hub as: $REMOTE_IMAGE_TAG"
docker tag $LOCAL_IMAGE_TAG $REMOTE_IMAGE_TAG

# 3. Login to Docker Hub
echo ""
echo "Please log in to Docker Hub..."
docker login -u $DOCKERHUB_USER

# 4. Push the image
echo ""
echo "Pushing image to Docker Hub..."
docker push $REMOTE_IMAGE_TAG

echo ""
echo "Successfully pushed! Image is available at: $REMOTE_IMAGE_TAG"
