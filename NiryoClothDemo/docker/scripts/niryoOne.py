import socket
import threading
import numpy as np
import cv2
from pyniryo import NiryoRobot
import math
from typing import List

class Command:
    """
    Command constants for the robot controller.
    """
    
    RESET_CAMERA_CALIBRATION_COORDINATES = 43
    MOVE = 44
    WRITE_MARKER = 45
    MAKE_HOMOGRAPHIC_MATRIX = 46
    GRAB_CLOTH = 47
    GO_HOME = 48
    SLEEP = 49
    GET_CALIBRATION_LENGTH = 50
    FAILED_GRAB_CLOTH = 53
    GET_GRIPPER_POSITION = 54
    
    OK = "51"
    ERROR = "52"

class RobotController:
    # https://archive-docs.niryo.com/dev/pyniryo/v1.1.1/en/_modules/api/tcp_client.html
    def __init__(self):
        self.home_pose_left = [0, 0.5, -1.25, math.pi/4, 0, 0]
        self.home_pose_right = [0, 0.5, -1.25, -math.pi/4, 0, 0]
        self.home_pose = [0, 0.5, -1.25, 0, 0, 0]
        self.drop_joints = [0.0, 0.0, 0.6, 0, -1.0, 0.0] 
        try:
            self.robot = NiryoRobot('169.254.200.200')
            self.robot.calibrate_auto()
            self.robot.update_tool()
            self.robot.release_with_tool()
        except Exception as e:
            print(f"Error occured during robot initialization.\n{e}")
            exit()


        self.camera_matrix = np.array([[908.31616211, 0.0, 956.25305176],
                                       [0.0, 908.07641602, 552.98254395],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        
        self.dist_coeffs = np.array([3.25172871e-01, -2.44627404e+00,  9.61361584e-05, -1.74064207e-05,
                                     1.42736828e+00,  2.04096168e-01, -2.26242542e+00,  1.35025930e+00])

        

        self.X_MIN, self.X_MAX = 0.14, 0.36
        self.Y_MIN, self.Y_MAX = -0.151, 0.151
        self.Z_MIN, self.Z_MAX = 0.049, 0.31
        
        self.robot_calibration_cords = np.array([
            [0.15, -0.150, 0.05],
            [0.15, 0.150, 0.05],
            [0.35, -0.150, 0.05], # 0.40
            [0.35, 0.150, 0.05], # 0.40
            [0.25, 0.0, 0.05], # For z range
            [0.25, 0.0, 0.2] # For z range
        ], dtype=np.float32)
        
        self.image_height = 1080
        self.image_width = 1920
        
        # Ranges for mapping cloth coordinates to robot range coordinates
        self.robot_range = {"min_X": 0.15, "max_X": 0.35, # 0.4
                            "min_Y": -0.150, "max_Y": 0.150,
                            "min_Z": 0.05, "max_Z": 0.3}
        
        #self.camera_range = {}
        self.camera_range = {'min_X': 277.22598, 'max_X': 520.05096, 'min_Y': 736.84875, 'max_Y': 1101.083, 'min_Z': 52.0, 'max_Z': 196.0}

        self.camera_calibration_cords = []
        #self.H = None  # Homography matrix
        
        # Robot doesnt like to turn within this range
        self.max_angle = 4.5378560552 # 260 angle in radians
        self.min_angle = 1.745329252 # 100 angle in radians
        self.angle_345 = (23*math.pi)/12
        self.angle_15 = math.pi/12 # 15 angle in radians
        
        self.angle_45 = 0.7853981634 # 45 angle in radians
        
        self.gripper_length = 90.0 # 9cm
        self.gripper_angle_offset = ((math.sqrt(2) * self.gripper_length)/2)/1000
        self.marker_offset_front = 0.020 # 0.020
        self.interpolation_offset = 0.075 # 0.075
        self.pitch = math.pi/6 #math.pi/4
        self.roll = math.pi/2
        #self.bottom_angle_range = 0.2792526803 # 16 degrees in radians
        #self.top_angle_range = self.min_angle
        
        
    def is_within_limits(self, x: float, y: float, z: float, angle: np.float32 = np.float32(0)):
        """
        Check if target position is within the defined robot arm limits.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            angle (np.float32): Angle in radians
        
        Returns:
            bool: True if within limits, False otherwise
        """
        if angle < 0:
            angle += 2*math.pi
        not_good_angle = (self.min_angle) <= angle <=  (self.max_angle)
        return (self.X_MIN <= x <= self.X_MAX) and (self.Y_MIN <= y <= self.Y_MAX) and (self.Z_MIN <= z <= self.Z_MAX) and not not_good_angle

    def move(self, position_number: int):
        """
        Move robot to the specified position index from the calibration coordinates.
        """
        cord = self.robot_calibration_cords[position_number]
        if self.is_within_limits(cord[0], cord[1], cord[2]):
            self.robot.move_pose([cord[0], cord[1], cord[2], 0, 0, 0])
        else:
            print(f"Robot is out of working bounds: X:{cord[0]}, Y:{cord[1]}, Z:{cord[2]}")

    def move_home(self):
        """
        Move robot to the home position.
        """
        self.robot.move_joints(self.home_pose)
    
    def failed_grab_cloth(self):
        """
        Move robot to show that it failed to grab the cloth.
        """
        self.robot.move_joints(self.home_pose_left)
        self.robot.move_joints(self.home_pose_right)
        self.robot.move_joints(self.home_pose)
        self.robot.set_learning_mode(True)
    
    def sleep(self):
        """
        Put the robot in learning mode.
        """
        self.robot.set_learning_mode(True)
    
    def write_marker(self, marker_positions: List[float]):
        """
        Write marker positions to the camera calibration coordinates.
        
        Args:
            marker_positions (List[float]): List of marker positions [x, y, z]
        """
        if marker_positions is not None:
            #print(f"\nOriginal input: X: {marker_positions[0]}, Y: {marker_positions[1]}")
            undistorted_pts = cv2.undistortPoints(np.array([[marker_positions[0], marker_positions[1]]], dtype=np.float32), self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            marker_positions[0] = undistorted_pts[0][0][0]
            marker_positions[1] = undistorted_pts[0][0][1]
            #print(f"Undistorted pixel coordinates: {undistorted_pts}\n")
            
            self.camera_calibration_cords.append(marker_positions)
            #print(f"ROBOT wrote marker position: {marker_positions}")

    def make_homographic_matrix(self):
        """
        Create homographic matrix from camera calibration coordinates and robot calibration coordinates.
        """
        if len(self.camera_calibration_cords) == len(self.robot_calibration_cords):
            # First write min and max values for 3D coordinates of camera calibration
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            min_z, max_z = float('inf'), float('-inf')
            
            for point in self.camera_calibration_cords:
                x, y, z = point
                
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            
            min_z = self.camera_calibration_cords[-2][-1]
            max_z = self.camera_calibration_cords[-1][-1]
            
            self.camera_range["min_X"] = min_x
            self.camera_range["max_X"] = max_x
            
            self.camera_range["min_Y"] = min_y
            self.camera_range["max_Y"] = max_y
            
            self.camera_range["min_Z"] = min_z
            self.camera_range["max_Z"] = max_z

            print(f"Calibration complete. Ranges: {self.camera_range}")

    def calculate_robot_position(self, x: float, y: float, z: float):
        """
        Calculate robot position based on camera coordinates.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
        
        Returns:
            tuple: Robot coordinates (x, y, z)
        """
        if self.camera_range["min_X"] <= x <= self.camera_range["max_X"] and self.camera_range["min_Y"] <= y <= self.camera_range["max_Y"]:
            robot_x = ((x - self.camera_range["min_X"])/(self.camera_range["max_X"] - self.camera_range["min_X"])) * (self.robot_range["max_X"] - self.robot_range["min_X"]) + self.robot_range["min_X"]
            robot_y = ((y - self.camera_range["min_Y"])/(self.camera_range["max_Y"] - self.camera_range["min_Y"])) * (self.robot_range["max_Y"] - self.robot_range["min_Y"]) + self.robot_range["min_Y"]
            robot_z = ((z - self.camera_range["min_Z"])/(self.camera_range["max_Z"] - self.camera_range["min_Z"])) * (self.robot_range["max_Z"] - self.robot_range["min_Z"]) + self.robot_range["min_Z"]
        else:
            robot_x, robot_y, robot_z = 0, 0, 0
        return (robot_x, robot_y, robot_z)

    def adjust_z(self, z: float):
        """
        Adjust Z coordinate based on camera coordinates.
        
        Args:
            z (float): Z coordinate
        
        Returns:
            float: Adjusted Z coordinate
        """
        robot_z = ((z - self.camera_range["min_Z"])/(self.camera_range["max_Z"] - self.camera_range["min_Z"])) * (self.robot_range["max_Z"] - self.robot_range["min_Z"]) + self.robot_range["min_Z"]
        return robot_z
    
    def adjust_x_y(self, angle: np.float32, x: float, y: float, offset: float):
        """
        Check if X and Y coordinates need to be adjusted based on the angle. Because of the marker height on the gripper.
        
        Args:
            angle (np.float32): Angle in radians
            x (float): X coordinate
            y (float): Y coordinate
        
        Returns:
            tuple: Adjusted X and Y coordinates
        """
        if angle < 0:
            measure_angle = angle + 2*math.pi
        else:
            measure_angle = angle
        
        if self.angle_45 < measure_angle < self.min_angle:
            #print("From 45 to 100 degrees, downwards")
            x += (0.03 * (math.sin(measure_angle)))
        elif self.angle_15 < measure_angle <= self.angle_45:
            #print("From 0 to 45 degrees, downwards")
            x += (0.025 * (math.sin(measure_angle)))
            y -= (0.007 * (math.cos(measure_angle)))
        elif 7*math.pi/4 < measure_angle < self.angle_345:
            #print("From 315 to 360 degrees, upwards")
            #x += (0.025 * (math.sin(measure_angle)))
            y += (0.01 * (math.cos(measure_angle)))
        elif self.max_angle < measure_angle < 7*math.pi/4:
            #print("From 260 to 315 degrees, upwards")
            x -= (0.025 * (math.sin(measure_angle)))
            #y += (0.007 * (math.cos(measure_angle)))
        
        if offset == self.interpolation_offset:
            x -= (offset * math.cos(angle))
            y -= (offset * math.sin(angle))
        
        return (x, y)
        
    
    def grab_cloth(self, x: float, y: float, z: float, angle: np.float32):
        """
        Move to detected cloth position and grab it.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            angle (np.float32): Angle in radians
        
        Returns:
            str: Command status (OK or ERROR)
        """
        #print(f"IMAGE COORDINATES: X:{y}, Y:{x}, Z:{z}")
        try:
            # Transform angle from camera angle to robot angle
            angle = angle * -1
            angle -= math.pi/2
            
            #print(f"\nINPUT: X:{x}, Y:{y}, Z:{z}")
            #test_x, test_y, test_z = self.calculate_robot_position(x, y, z)
            
            undistorted_pts = cv2.undistortPoints(np.array([[x, y]], dtype=np.float32), self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            #print(f"Undistorted pixel: {undistorted_pts}")
            #print(f"Undistorted: X:{y}, Y:{x}, Z:{z}")
            
            x, y, z = self.calculate_robot_position(undistorted_pts[0][0][0], undistorted_pts[0][0][1], z)
            
            #print(f"\nORIGINAL: X:{test_x}, Y:{test_y}, Z:{test_z}; UNDISTORTED: X:{x}, Y:{y}, Z:{z}")
            
            #print(f"Before adjusting: X:{x}, Y:{y}, Z:{z}, Roll:{self.roll}, Pitch{self.pitch}, Yaw:{angle}")

            # Adjust X and Y coordinates based on the angle
            inter_x, inter_y = self.adjust_x_y(angle, x, y, self.interpolation_offset)
            x, y = self.adjust_x_y(angle, x, y, self.marker_offset_front)

            # Adjust Z coordinate based on the camera coordinates
            if z < 0:
                #print(f"Z is less than 0: {z}")
                z = self.gripper_angle_offset
            else:
                #print(f"Z is greater than 0: {z}")
                z += self.gripper_angle_offset
           
            #print(f"After adjusting: X:{x}, Y:{y}, Z:{z}, Roll:{self.roll}, Pitch{self.pitch}, Yaw:{angle}")
            
            if self.is_within_limits(x, y, z, angle) and self.is_within_limits(inter_x, inter_y, z, angle):
                #print(f"\nRobot intermediate MOVED TO: X:{inter_x}, Y:{inter_y}, Z:{z}, Roll:{self.roll}, Pitch{self.pitch}, Yaw:{angle}")
                print(f"Robot MOVED TO: X:{x}, Y:{y}, Z:{z}, Roll:{self.roll}, Pitch{self.pitch}, Yaw:{angle}\n")
                self.robot.move_pose([inter_x, inter_y, z, self.roll, self.pitch, angle])
                self.robot.move_pose([x, y, z, self.roll, self.pitch, angle])
                self.robot.grasp_with_tool()
                self.robot.move_joints(self.drop_joints)
                self.robot.release_with_tool()
                self.robot.move_joints(self.home_pose)
                self.robot.set_learning_mode(True)
                return Command.OK
            else:
                print(f"Robot is out of working bounds: X:{x}, Y:{y}, Z:{z}, Roll:{self.roll}, Pitch{self.pitch}, Yaw:{angle}")
                return Command.ERROR
        except Exception as e:
            print(f"Error moving robot: {e}")
            if "Goal has been aborted" in str(e):
                self.robot.move_joints(self.home_pose)
                self.robot.set_learning_mode(True)
            return Command.ERROR

def handle_robot(client_socket: socket.socket, robot_controller: RobotController):
    """
    Handles communication with the client and sends status of the executed functions.
    
    Args:
        client_socket (socket.socket): Client socket
        robot_controller (RobotController): Robot controller instance
    """
    try:
        while True:
            request = client_socket.recv(1024).decode('utf-8').strip()
            if not request:
                break # Client disconected

            #print(f"[*] Request from robot: {request}")

            # Determine which function is called
            if " " in request:
                function_arguments = request.split(" ")
                if int(function_arguments[0]) == Command.MOVE:
                    robot_controller.move(int(function_arguments[1]))
                    response = Command.OK
                elif int(function_arguments[0]) == Command.WRITE_MARKER:
                    # X and Y in camera axis are inverted in the robot axis
                    marker_position = [float(function_arguments[2]), float(function_arguments[1]), float(function_arguments[3])]
                    robot_controller.write_marker(marker_position)
                    response = Command.OK
                elif int(function_arguments[0]) == Command.GRAB_CLOTH:
                    response = robot_controller.grab_cloth(float(function_arguments[2]), float(function_arguments[1]), float(function_arguments[3]), np.float32(function_arguments[4]))
                else:
                    response = Command.ERROR
            else:
                if int(request) == Command.RESET_CAMERA_CALIBRATION_COORDINATES:
                    robot_controller.camera_calibration_cords = []
                    
                    response = Command.OK
                elif int(request) == Command.MAKE_HOMOGRAPHIC_MATRIX:
                    robot_controller.make_homographic_matrix()
                    response = Command.OK
                elif int(request) == Command.GO_HOME:
                    robot_controller.move_home()
                    response = Command.OK
                elif int(request) == Command.SLEEP:
                    robot_controller.sleep()
                    response = Command.OK
                elif int(request) == Command.GET_CALIBRATION_LENGTH:
                    response = str(len(robot_controller.robot_calibration_cords))
                elif int(request) == Command.FAILED_GRAB_CLOTH:
                    robot_controller.failed_grab_cloth()
                    response = Command.OK
                elif int(request) == Command.GET_GRIPPER_POSITION:
                    pose = robot_controller.robot.get_pose_quat()
                    response = str(pose[0]) + ", " + str(pose[1])
                else:
                    response = Command.ERROR

            # Send the response back to the client
            client_socket.sendall(response.encode('utf-8'))

    except Exception as e:
        print(f"[!] Error handling client: {e}")
        client_socket.sendall(Command.ERROR.encode('utf-8'))
    finally:
        client_socket.close()
        #print(f"[*] Connection closed.")

def main():
    # Make instance of the robot
    robot_controller = RobotController()

    # Start server
    server_ip = 'localhost'
    server_port = 49152

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(5)

    try:
        while True:
            client_socket, _ = server_socket.accept()
            #print(f"[*] Client connected.")
            client_thread = threading.Thread(target=handle_robot, args=(client_socket, robot_controller))
            client_thread.start()
    except KeyboardInterrupt:
        print("[!] Shutting down server.")
    finally:
        robot_controller.robot.set_learning_mode(True)
        robot_controller.robot.close_connection()
        server_socket.close()

if __name__=='__main__':
    main()

