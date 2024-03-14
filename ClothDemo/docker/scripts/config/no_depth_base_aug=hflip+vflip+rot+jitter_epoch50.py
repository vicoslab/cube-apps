import copy


ENABLE_6DOF = False
NUM_FIELDS = 3 + (6 if ENABLE_6DOF else 2)

#SIZE = 1024
SIZE = 768
# SIZE = 384
# SIZE = 256
BACKBONE='tu-convnext_base'

USE_DEPTH = False
#USE_DEPTH = True

#CAMERA = "kinect" # "allied"
CAMERA = "kinect_azure"

# model from /home/domen/Projects/center-vector-exp/rtfm/robot_application/backbone=tu-convnext_base_resize_factor=1_gpu=1_size=768x768/aug=hflip+vflip+rot+jitter/num_train_epoch=50/depth=False/checkpoint_045.pth

args = dict(
    threshold=0.75,
	use_depth = USE_DEPTH,
	size=SIZE,
	camera=CAMERA,
	cuda=True,

	model=dict(
		name='fpn',
		kwargs={
			"pretrained": False,
			'backbone': BACKBONE,
			'num_classes': [NUM_FIELDS, 1],
			'use_custom_fpn': True,
			'add_output_exp': False,
			'use_depth': USE_DEPTH,
			'fpn_args': {
				# 'decoder_dropout':0
				'decoder_segmentation_head_channels': 64,
				'classes_grouping': [(0, 1, 2, 5), (3, 4)],
				'depth_mean': 0.96, 'depth_std': 0.075,
			},
			'init_decoder_gain': 0.1
		},
	),
	center_model=dict(
		name='CenterOrientationEstimator',
		use_learnable_center_estimation=True,

		kwargs=dict(
			use_centerdir_radii = False,
			
			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=True,
			# thresholds for conv2d processing
			local_max_thr=0.01,
			mask_thr=0.01, hough_thr=5000, local_max_thr_use_abs=True,
			suppression_by_mask=False,
			### dilated neural net as head for center detection
			ignore_centerdir_magnitude=True,
			ignore_cls_prediction=True,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				return_sigmoid=False,
				# single scale version (nn6)
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				use_centerdir_radii=False,
				use_centerdir_magnitude=False,
				use_cls_mask=False
			),
			scale_r=1.0,  # 1024
			scale_r_gt=1,  # 1
			use_log_r=False,
			use_log_r_base='10',
			enable_6dof=ENABLE_6DOF,
		),

	),
	num_vector_fields=NUM_FIELDS,
)

def get_args():
	return copy.deepcopy(args)
