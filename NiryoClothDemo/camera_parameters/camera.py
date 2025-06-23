from pyk4a import PyK4A, Config, ColorResolution, DepthMode, CalibrationType
from config import config

# Set up the configuration parameters
config = Config(color_resolution=ColorResolution.RES_1080P, depth_mode=DepthMode.NFOV_UNBINNED)

# Initialize the camera with the configuration
k4a = PyK4A(config)

# Start the camera
k4a.start()

# Get calibration data
calibration = k4a.calibration
color_camera_matrix = calibration.get_distortion_coefficients(1)
print(color_camera_matrix)


# Access the color camera intrinsics
#color_intrinsics = calibration.color.intrinsics

# Print intrinsics values
#print("fx:", color_intrinsics.fx)
#print("fy:", color_intrinsics.fy)
#print("cx:", color_intrinsics.cx)
#print("cy:", color_intrinsics.cy)

# Stop and close the camera
k4a.stop()