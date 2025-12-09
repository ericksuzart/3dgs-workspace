docker run --rm --name 3dgs-colmap \
  --gpus all -v "$DATASET_PATH":/dataset:rw \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  ericksuzart/3dgs-colmap convert.sh -s /dataset --resize --magick_executable ""
