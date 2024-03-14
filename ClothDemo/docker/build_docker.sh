# download model from:
# depth_large -> /storage/group/RTFM/household_set_v1+v2+v3+folded/3dof/backbone=tu-convnext_large_resize_factor=1_size=768x768/num_train_epoch=50/depth=True_hardneg=0/multitask_weight=uw_w_orient=1/weakly_supervised=True
# depth_base -> /storage/group/RTFM/household_set_v1+v2+folded/3dof/backbone=tu-convnext_base_resize_factor=1_size=768x768/num_train_epoch=50/depth=True_hardneg=0/multitask_weight=uw_w_orient=1/weakly_supervised=True
# no_depth_large -> /storage/group/RTFM/household_set_v1+v2+v3+folded/3dof/backbone=tu-convnext_large_resize_factor=1_size=768x768/num_train_epoch=50/depth=False_hardneg=0/multitask_weight=uw_w_orient=1/weakly_supervised=True
# no_depth_base -> /storage/group/RTFM/household_set_v1+v2/backbone=tu-convnext_base_resize_factor=1_size=768x768/num_train_epoch=50/depth=False/multitask_weight=uw_w_orient=1/weakly_supervised=True

# download center_model
# /storage/group/DIVID/pretrained-center-models/learnable-center-dilated-kernels-w_fg_cent=100_nn6.7.8.1_k=3_ch=16-loss=l1_lr=1e-4_hard_negs=4_on_occluded_augm_head_only_fully_syn=5k_no_edge_with_replacement
# save it as center_model.pth

# For this demo, only no_depth models work!
#MODEL_VERSION="no_depth_large"
#MODEL_VERSION="no_depth_base_aug=hflip+vflip+rot+jitter"
MODEL_VERSION="no_depth_base_aug=hflip+vflip+rot+jitter_epoch50" 

docker build -t cloth-demo --build-arg CEDIRNET_VERSION=$(date +%s) --build-arg MODEL_VERSION=${MODEL_VERSION} .

# Use ViCoSDemoRun software to run this cloth-demo image
