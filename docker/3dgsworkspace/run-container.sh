docker run --rm --name 3dgs-workspace --gpus all \
  -v "$DATASET_PATH":/dataset:rw -w /dataset \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  ericksuzart/3dgsworkspace /bin/bash \
  -c "python3 \$GS_PATH/train.py -s /dataset"
