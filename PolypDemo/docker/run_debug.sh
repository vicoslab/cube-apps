#!/bin/bash

# without code override
#docker run --gpus=all --rm -v ./examples:/opt/data poco-vicos-demo:ubuntu18.04-cuda11.2.0-cudnn8 \
#            --model /opt/poco_model.hdf5 --image-ex "png" --image-folder /opt/data/ --out-folder /opt/data/out

# with code override
docker run --gpus=all --rm -v ./examples:/opt/data -v ./scripts/run_main.py:/opt/run_main.py -v ./scripts/utils.py:/opt/utils.py poco-vicos-demo:ubuntu18.04-cuda11.2.0-cudnn8 \
            --model /opt/poco_model.hdf5 --image-ex "png" --image-folder /opt/data/ --out-folder /opt/data/out