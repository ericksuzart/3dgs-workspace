#!/usr/bin/env bash
set -euo pipefail

# Build and push Docker images to DockerHub.
#
# USAGE:
#   ./build_and_push.sh <dockerhub-user> <image-name> <image-tag>
#
# EXAMPLES:
#   ./build_and_push.sh myusername 3dgsworkspace latest
#   ./build_and_push.sh myusername 3dgs-colmap v1.0.0
#
# ENVIRONMENT:
#   DOCKERHUB_TOKEN  DockerHub access token (required for CI).
#                    If unset, interactive login is used.

# Argument Parsing
if [[ $# -lt 3 ]]; then
    echo "ERROR: Missing arguments."
    echo "Usage: $0 <dockerhub-user> <image-name> <image-tag>"
    echo "Example: $0 myusername 3dgsworkspace latest"
    exit 1
fi

DOCKERHUB_USER="$1"
IMAGE_NAME="$2"
IMAGE_TAG="$3"

# Validate that the image directory exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="${SCRIPT_DIR}/${IMAGE_NAME}"

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "ERROR: Image directory '${IMAGE_DIR}' does not exist."
    # List available image directories (exclude *.sh and pre-processing)
    local_dirs=()
    for d in "${SCRIPT_DIR}"/*/; do
        base="$(basename "$d")"
        [[ "$base" == "pre-processing" ]] && continue
        local_dirs+=("$base")
    done
    echo "Available images: ${local_dirs[*]}"
    exit 1
fi

cd "$IMAGE_DIR"

# Define the local and remote image tags
LOCAL_IMAGE_TAG="${IMAGE_NAME}-build:${IMAGE_TAG}"
REMOTE_IMAGE_TAG="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build the Docker image
echo "Building Docker image from: ${IMAGE_DIR}/Dockerfile"
docker build -m "${CONTAINER_MEMORY:-12g}" --platform linux/amd64 -t "$LOCAL_IMAGE_TAG" -f Dockerfile .
echo "Build complete."

# Tag the image for Docker Hub
echo "Tagging image for Docker Hub as: $REMOTE_IMAGE_TAG"
docker tag "$LOCAL_IMAGE_TAG" "$REMOTE_IMAGE_TAG"

# Login to Docker Hub
echo ""
if [[ -n "${DOCKERHUB_TOKEN:-}" ]]; then
    echo "Logging in to Docker Hub (via token)..."
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin
else
    echo "DOCKERHUB_TOKEN not set — falling back to interactive login."
    echo "For CI/CD, set DOCKERHUB_TOKEN to a DockerHub access token."
    docker login -u "$DOCKERHUB_USER"
fi

# Push the image
echo ""
echo "Pushing image to Docker Hub..."
docker push "$REMOTE_IMAGE_TAG"

echo ""
echo "Successfully pushed! Image is available at: $REMOTE_IMAGE_TAG"
