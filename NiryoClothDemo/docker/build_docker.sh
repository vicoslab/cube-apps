MODEL_VERSION="no_depth_base_aug=hflip+vflip+rot+jitter_epoch50" 

docker build -t niryo-cloth-demo --build-arg CEDIRNET_VERSION=$(date +%s) --build-arg MODEL_VERSION=${MODEL_VERSION} .
