import socket
from echolib_wrapper_intermediate import EcholibWrapper
from typing import List
import numpy as np

class Command:
    """
    Command constants for the robot controller.
    """
    
    RESET_CAMERA_CALIBRATION_COORDINATES = "43"
    MOVE = "44"
    WRITE_MARKER = "45"
    MAKE_HOMOGRAPHIC_MATRIX = "46"
    GRAB_CLOTH = "47"
    GO_HOME = "48"
    SLEEP = "49"
    GET_CALIBRATION_LENGTH = "50"
    FAILED_GRAB_CLOTH = "53"
    GET_GRIPPER_POSITION = "54"
    
    OK = 51
    ERROR = 52

class ClientWrapper():
    def __init__(self, server_ip='localhost', server_port=49152):
        self.server_ip = server_ip
        self.server_port = server_port

    def reset_camera_calibration_cords(self):
        """
        Reset camera calibration coordiantes inside niryoOne script.
        
        Returns:
            True if the reset was successful, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                sock.sendall(Command.RESET_CAMERA_CALIBRATION_COORDINATES.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')
                if int(response.strip()) == Command.OK:
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Error connecting to NiryoOne server. \nException: {e}")
            return False

    def move(self, position: int):
        """
        Move the robot to a specific position.
        
        Args:
            position: The position to move to (0-len(robots_calibration_coordinates)).
        
        Returns:
            True if the move was successful, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.MOVE + " " + str(position)
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')

                if int(response.strip()) == Command.OK:
                    return True
                else:
                    return False
        except Exception as e:
            print(e)
            return False
    
    def write_marker(self, marker_position: List[float]):
        """
        Write marker position to the server.
        
        Args:
            marker_position: The position of the marker in 3D space to write to the server.
        
        Returns:
            True if the write was successful, False otherwise.
        """
        try:
            marker_position = [str(point) for point in marker_position]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.WRITE_MARKER + " " + " ".join(marker_position)
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')

                if int(response.strip()) == Command.OK:
                    return True
                else:
                    #print(f"Unexpected response from server: {response}")
                    return False
        except Exception as e:
            print(e)
            return False

    def make_homographic_matrix(self):
        """
        Make homographic matrix inside niryoOne script.
        
        Returns:
            True if the matrix was created successfully, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                sock.sendall(Command.MAKE_HOMOGRAPHIC_MATRIX.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')

                if int(response.strip()) == Command.OK:
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Error connecting to NiryoOne server.")
            return False

    def grab_cloth(self, x: float, y: float, z: float, angle: np.float32):
        """
        Grab cloth with the robot.
        
        Args:
            x: The x-coordinate of the cloth in 3D space.
            y: The y-coordinate of the cloth in 3D space.
            z: The z-coordinate of the cloth in 3D space.
            angle: The angle at which to grab the cloth.
        
        Returns:
            True if the grab was successful, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.GRAB_CLOTH + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(angle)
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')

                if response.strip() == str(Command.OK):
                    return response.strip().split(", ")
                else:
                    print(f"Unexpected response from server: {response}")
                    return False
        except Exception as e:
            print(e)
            return False

    def move_home(self):
        """
        Move the robot to the home position.
        
        Returns:
            True if the move was successful, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.GO_HOME
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')

                if int(response.strip()) == Command.OK:
                    return True
                else:
                    #print(f"Unexpected response from server: {response}")
                    return False
        except Exception as e:
            print(e)
            return False
    
    def sleep(self):
        """
        Set robot to learning mode.
        
        Returns:
            True if the sleep was successful, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.SLEEP
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')

                if int(response.strip()) == Command.OK:
                    return True
                else:
                    #print(f"Unexpected response from server: {response}")
                    return False
        except Exception as e:
            print(e)
            return False
    
    def non_grab_cloth(self):
        """
        Moves robot to indicate failed grab.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.FAILED_GRAB_CLOTH
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')

                if int(response.strip()) == Command.OK:
                    return True
                else:
                    #print(f"Unexpected response from server: {response}")
                    return False
        except Exception as e:
            print(e)
            return False
    
    def get_calibration_length(self):
        """
        Get the length of the calibration coordinates.
        
        Returns:
            The length of the calibration coordinates.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.GET_CALIBRATION_LENGTH
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')
                if int(response.strip()) == Command.ERROR:
                    return 0
                return int(response.strip())
        except Exception as e:
            print(e)
            return 0
    
    def get_gripper_position(self):
        """
        Get the current position of the gripper.
        
        Returns:
            The current position of the gripper.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                message = Command.GET_GRIPPER_POSITION
                sock.sendall(message.encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')
                if response == str(Command.ERROR):
                    return 0.0, 0.0
                return response.strip()
        except Exception as e:
            print(e)
            return 0
    
def main():
    demo = EcholibWrapper(ClientWrapper())

    try:
        demo.run()
    except KeyboardInterrupt:
        pass

if __name__=='__main__':
    main()