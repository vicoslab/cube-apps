import echolib
from echolib.camera import Frame, FrameSubscriber, FramePublisher
import time
from threading import Thread
import numpy as np
import cv2
import skimage as ski
import string

class Command:
    DISABLE = 0
    ENABLE = 1

    CALIBRATE = 41
    GRAB = 42

class EcholibWrapper:
    """
    Processes incoming data from the EchoLib client and handles the communication with the NiryoOne robot.
    """

    def __init__(self, client_wrapper):
        
        self.loop = echolib.IOLoop()
        self.client = echolib.Client()
        self.loop.add_handler(self.client)

        #self.n_processed_frames = 0
        #self.n_depth_frame = 0
        
        self.docker_ready      = echolib.Publisher(self.client, "containerReady", "int")

        # Subscribe to the cloth position output (x, y, z)
        self.cloth_position_sub = echolib.Subscriber(self.client, "cloth_position_output", "str", self._cloth_position_callback)

        # Subscribe to cloth demo Frame output
        self.docker_demo_sub = FrameSubscriber(self.client, "docker_demo_output", self._docker_demo_callback)
        self.depth_frame_sub = FrameSubscriber(self.client, "azure_kinect_depth", self._read_depth_callback)
        
        
        # Subscribe to the processed image stream (if needed)
        self.docker_command_in = echolib.Subscriber(self.client, "docker_demo_command_input", "int", self._docker_command_callback)

        self.cloth_positions = None
        self.image_frame = None

        self.closing = False

        # Variables needed for pyNiryo robot workflow
        self.client_wrapper = client_wrapper
        self.grab = False
        self.calibrate = False
        
        self.image_height = 1080
        self.image_width = 1920


        self.marker_positions = None

        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250), cv2.aruco.DetectorParameters())
        self.base_depth = 785 # 0
        
        self.depth_image = None
        
        self.frame_out_new = False

        self.frame_out = None
        self.x_gripper = None
        self.y_gripper = None
        self.x_gripper_final = None
        self.y_gripper_final = None
        
        
    def _read_depth_callback(self, message):
        """
        Handles incoming depth image frames.
        
        Args:
            message (echolib.Message): The incoming message containing the depth image.
        """
        self.depth_image = message.image
        #self.n_depth_frame += 1
        #print(f"Received new processed depth frame {self.n_depth_frame}.")
    
    def _docker_demo_callback(self, message):
        """
        Handles incoming processed image frames.
        This is where the marker detection and position calculation occurs.
        
        Args:
            message (echolib.Message): The incoming message containing the processed image.
        """
        
        self.image_frame = message.image
        
        # Detect marker
        gray = cv2.cvtColor(self.image_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
    
        if ids is not None:
            for marker_id in ids.flatten():
                if marker_id == 40:
                    marker_corners = corners[0][0]
                    center_x = int(np.mean(marker_corners[:, 0]))
                    center_y = int(np.mean(marker_corners[:, 1]))
            
                    if self.depth_image is not None:
                        z = self.depth_image[center_y, center_x]
                        if self.base_depth > 0:
                            #print(f"\n[!] Calculated depth {self.base_depth - z}\n")
                            z = np.int16(self.base_depth) - np.int16(z)
                            self.marker_positions = [center_x, center_y, z]


    def _cloth_position_callback(self, message):
        """
        Handles incoming cloth position data.
        This is where the positions are parsed and stored.
        
        Args:
            message (echolib.Message): The incoming message containing the cloth position data.
        """
        position_str = echolib.MessageReader(message).readString()
        if position_str != "":
            position_values = position_str.split(", ")

            positions = []
            
            # Iterate over all positions in chunuks of 4 (X, Y, Z, Angle)
            for i in range(0, len(position_values), 3):
                x     = np.float32(position_values[i])
                y     = np.float32(position_values[i + 1])
                safe_angle = ''
                for c in position_values[i + 2]:
                    if c in string.printable:
                        safe_angle += c
                    else:
                        break
                
                angle = np.float32(safe_angle)

                # Get square around the marker position
                matrix = np.zeros((32, 32), dtype=np.float32)
                for i in range(-16, 16):
                    for j in range(-16, 16):
                        #if (x + i) < 0 or (x + i) >= self.depth_width or (y + j) < 0 or (y + j) >= self.depth_height:
                        #    continue
                        z = self.depth_image[int(y + j), int(x + i)]
                        matrix[i+16, j+16] = int(self.base_depth) - int(z)
                # Zero values must be ignored because they are usualy majority and wrong depth
                nonzero_vals = matrix[matrix > 0]


                otsu_val = None
                # In case this fails depth of the cloth must be 0
                if nonzero_vals.size > 0:
                    # Perform tresholding to remove noise
                    otsu_thresh = ski.filters.threshold_otsu(nonzero_vals)
                    # Cloth corner is most % of the time higher than everything in front of it
                    otsu_val = nonzero_vals[nonzero_vals >= otsu_thresh]
                #else:
                    #print("[!] No non-zero values found in the matrix.")
                
                if otsu_val is not None:    
                    z = np.median(otsu_val)
                else:
                    z = 0.0
                    #print("[!] No otsu value found.")
                
                # Append the combination touple
                positions.append((x, y, z, angle))
                
                #print(f"Mean otsu: {z}")
                #print(f"Received cloth position: X={x}, Y={y}, Z={z_normalised}, Angle={angle}")
            self.cloth_positions = positions


    def _docker_command_callback(self, message):
        """
        Handles incoming command messages from the Docker container.
        
        Args:
            message (echolib.Message): The incoming message containing the command.
        """
        msg = echolib.MessageReader(message).readInt()
        #self.image_frame = message.image
        #print(f"Received new message from Docker {msg}.")

        if msg == Command.CALIBRATE:
            self.calibrate = True
        elif msg == Command.GRAB:
            self.grab = True
            

    def process(self):
        """
        Main processing loop for handling incoming data.
        """
        while not self.closing:
            if self.calibrate:
                self.calibrate = False
                outcome = self.client_wrapper.reset_camera_calibration_cords()
                if self.depth_image is None:
                    print("[!] Calibration failed, depth image is None.")
                    return
                self.base_depth = self.depth_image[int(self.image_height/2), int(self.image_width/2)]
                print(f"Got depth: {self.base_depth}")
                if outcome:
                    calibration_length = self.client_wrapper.get_calibration_length()
                    for i in range(calibration_length):
                        moved = self.client_wrapper.move(i)
                        time.sleep(1.5)
                        if moved:
                            if self.marker_positions is None:
                                print("[!] Marker not detected, please turn on the light.")
                            self.client_wrapper.write_marker(self.marker_positions)
                            self.client_wrapper.move_home()
                            time.sleep(0.5)
                    self.client_wrapper.sleep()
                    self.client_wrapper.make_homographic_matrix()
                    
            elif self.grab:
                self.grab = False
                if self.cloth_positions:
                    responce = False
                    for position in self.cloth_positions:
                        if not responce:
                            responce = self.client_wrapper.grab_cloth(position[0], position[1], position[2], position[3]) # X, Y, Z, Angle
                            if responce:
                                if len(responce) == 4:
                                    self.x_gripper = float(responce[0])
                                    self.y_gripper = float(responce[1])
                                    self.x_gripper_final = float(responce[2])
                                    self.y_gripper_final = float(responce[3])
                            
                                    print(f"Grabbing cloth at: X={self.x_gripper_final}, Y={self.y_gripper_final}, X1={self.x_gripper_final}, Y1={self.y_gripper_final}")
                    if not responce:
                        self.client_wrapper.non_grab_cloth()
                    
        time.sleep(0.01)

    def run(self, wait_sec=10, sleep_sec=0):
        for i in range(0,10):
            self.loop.wait(10)

            writer = echolib.MessageWriter()
            writer.writeInt(1)
            self.docker_ready.send(writer)

        thread = Thread(target = self.process)
        thread.start()

        print("Starting intermediate wrapper...")

        while self.loop.wait(1):
            continue
            #if self.frame_out_new:
                
            #    self.docker_frame_out.send(Frame(image = self.image_frame))
            #    self.frame_out_new = False 

        print("Stop intermediate wrapper...")

        thread.join()