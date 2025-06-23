import time
import cv2

import echolib
from echolib.camera import Frame, FramePublisher, FrameSubscriber
import numpy as np
from threading import Thread

class Command:
    
    DISABLE = 0
    ENABLE = 1
    
    CAMERA_STREAM_DEFAULT = 10
    CAMERA_STREAM_KINECT_AZURE = 11
    CAMERA_STREAM_KINECT_V2 = 12

class EcholibWrapper:
    """
    EcholibWrapper class that handles the communication with the Echolib server and processes the incoming frames.
    It subscribes to the camera stream and processes the frames using the specified detection method.
    It also handles the incoming docker commands and sends the processed frames and cloth positions to the appropriate channels.
    """
    
    CAMERA_STREAMS = {
        Command.CAMERA_STREAM_DEFAULT: dict(rgb="camera_stream_0"),
        Command.CAMERA_STREAM_KINECT_AZURE: dict(rgb="azure_kinect_rgb", depth="azure_kinect_depth"),
        Command.CAMERA_STREAM_KINECT_V2: dict(rgb="kinect_rgb", depth="kinect_depth")
    }

    def __init__(self, detection_method):

        self.loop   = echolib.IOLoop()
        self.client = echolib.Client()
        self.loop.add_handler(self.client)

        self.enabled = False

        self.docker_ready       = echolib.Publisher(self.client, "containerReady", "int")
        self.cloth_position_out = echolib.Publisher(self.client, "cloth_position_output", "str")
        self.docker_command_in  = echolib.Subscriber(self.client, "docker_demo_command_input", "int", self._docker_command_callback)
        
        self.depth_frame_sub = FrameSubscriber(self.client, "azure_kinect_depth", self._depth_stream_callback) # DELETE ME 
        
        self.camera_stream = FrameSubscriber(self.client, self.CAMERA_STREAMS[Command.CAMERA_STREAM_KINECT_AZURE]['rgb'], self._camera_stream_callback)
        self.camera_stream_depth    = None

        self.docker_frame_out = FramePublisher(self.client, "docker_demo_output")

        self.detection_method = detection_method

        self.frame_in    = None
        self.frame_in_new = False

        self.frame_out    = None
        self.frame_out_new = False 

        self.closing = False

        self.n_frames = 0
        self.depth_n_frames = 0
        
        self.position_out_new = True
        self.cloth_pos_with_depth = ""
        
        self.depth_frame_in = None
        self.depth_frame_in_new = False
        
    def _docker_command_callback(self, message):
        """
        Callback function to handle incoming docker commands like camera switching and enabling/disabling them.
        
        Args:
            message (echolib.Message): The incoming message containing the command.
        """
        
        print("parsing docker command callback")
        
        msg = echolib.MessageReader(message).readInt()
                
        if msg in self.CAMERA_STREAMS:
            stream_properties = self.CAMERA_STREAMS[msg]
            print("Switch streaming to {}".format(stream_properties['rgb']))
            
            self.camera_stream = FrameSubscriber(self.client, stream_properties['rgb'], self._camera_stream_callback)

            if 'depth' in stream_properties:
                self.camera_stream_depth = FrameSubscriber(self.client, stream_properties['depth'], self._depth_stream_callback)
            else:
                self.camera_stream_depth = None

        elif msg in [Command.ENABLE,Command.DISABLE]:
            self.enabled = msg != 0
        
        print("Docker demo: got command {}".format(msg))    

    def _camera_stream_callback(self, message):
        """
        Handles incoming camera stream frames.
        
        Args:
            message (echolib.Message): The incoming message containing the camera frame.
            It is a 3D array of RGB values.
        """
        
        self.frame_in    = message.image
        self.frame_in_new = True

        self.n_frames += 1

        #print("Docker demo: reading camera stream {}".format(self.n_frames))

    def _depth_stream_callback(self, message):
        """
        Handles incoming depth stream frames.
        
        Args:
            message (echolib.Message): The incoming message containing the depth frame.
            It is 1D array of depth values.
        """
        
        self.depth_frame_in    = message.image
        self.depth_frame_in_new = True

        self.depth_n_frames += 1

        #print("Docker demo: reading depth stream {}".format(self.depth_n_frames))

    def process(self):
        """
        Main processing loop that handles the incoming frames and applies the detection method.
        It runs in a separate thread to avoid blocking the main loop.
        Formats the output and prepears them for the appropriate channels.
        """
        
        while not self.closing:
            frame = None
            cloth_positions = None

            if self.frame_in_new:
                frame = self.frame_in
                self.frame_in_new = False

                if self.enabled:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame, cloth_positions = self.detection_method.predict(frame)

            if frame is not None:
                self.frame_out = frame
                self.frame_out_new = True 
            if cloth_positions:
                self.cloth_pos_with_depth = ', '.join(f"{x}, {y}, {angle}" for x, y, angle in cloth_positions)
                self.position_out_new = True
                    
            time.sleep(0.01)
         
    def run(self, wait_sec=10, sleep_sec=0):
        """
        Main run method that initializes the loop and starts the processing thread.
        It also handles the incoming messages and sends the processed frames and cloth positions to the appropriate channels.
        """
        
        for i in range(0,10):
            self.loop.wait(10)

            writer = echolib.MessageWriter()
            writer.writeInt(1)
            self.docker_ready.send(writer)

        thread = Thread(target = self.process)
        thread.start()

        print("Starting...")

        while self.loop.wait(1):

            #print("In loop...")

            if self.frame_out_new:
                
                self.docker_frame_out.send(Frame(image = self.frame_out))
                self.frame_out_new = False 

            if self.position_out_new:
                
                writer = echolib.MessageWriter()
                writer.writeString(self.cloth_pos_with_depth)
                self.cloth_position_out.send(writer) # Send [x, y, z] to the "cloth_position_output" channel
                self.position_out_new = False

        print("Stop")

        thread.join()