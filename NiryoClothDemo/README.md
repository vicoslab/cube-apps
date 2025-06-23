# Prijemanje tekstila
The NiryoOne robot grasps one of the detected corner points of the cloth.

The NiryoOne script requires Python 3.6, whereas the rest of the codebase runs on a newer Python version. To bridge this compatibility gap, an intermediate_connector is used.
NiryoOne IP:port == 169.254.200.200:49152. In order for connection to be established both container and robot need to be on the same subnet.

The run_main.py script initialises an instance of the echolib_wrapper, which handles the corner prediction process and publishes the results to both the display and the echolib_intermediate_wrapper.py. The echolib_intermediate_wrapper.py, created within the intermediate_connector.py, sends the detected corner coordinates and angle to the robot arm via a TCP connection. Upon receiving this data, the robot executes the grasp action.

## Installation

Compile it using: 
```bash
cd /cube-apps/NiryoClothDemo/docker
sudo ./build_docker.sh
```
> **Note:** Machine which runs docker must be connected to NiryoOne robot arm and Azure Kinect DK camera.

## Usage

1. Run cube-main GUI environment
2. Turn on the robot and wait for its lights to turn blue
3. Turn on the demo cell main light
4. Click on option "Prijemanje tekstila"
5. Calibrate robot arm with button "Umeri"
6. Activate cloth corner detection with button "Vključi detekcijo"
7. Grasp a corner with button "Zgrabi krpo"