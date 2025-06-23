import time
import numpy as np

import echolib
from echolib.camera import Frame, FramePublisher, FrameSubscriber
from echolib.array import TensorSubscriber

from threading import Thread

class Command:
    
    DISABLE = 0
    ENABLE = 1
    
    CAMERA_STREAM_DEFAULT = 10
    CAMERA_STREAM_KINECT_AZURE = 11
    CAMERA_STREAM_AXIS_PTZ = 12

class EcholibWrapper:
    
    CAMERA_STREAMS = {
        Command.CAMERA_STREAM_DEFAULT: "camera_stream_0",
        Command.CAMERA_STREAM_KINECT_AZURE: "azure_kinect_rgb",
        Command.CAMERA_STREAM_AXIS_PTZ: "camera_stream_ptz",
    }

    def __init__(self, method, args):

        self.loop   = echolib.IOLoop()
        self.client = echolib.Client()
        self.loop.add_handler(self.client)

        self.enabled = False

        self.docker_ready      = echolib.Publisher(self.client, "containerReady", "int")
        self.docker_command_in = echolib.Subscriber(self.client, "docker_demo_command_input", "int", self._docker_command_callback)
        
        self.camera_stream    = FrameSubscriber(self.client, "camera_stream_0", self._camera_stream_callback)
        self.docker_frame_out = FramePublisher(self.client, "docker_demo_output")
        
        self.threshold = echolib.Subscriber(self.client, "counting_threshold", "float", self._threshold_callback)
        self.threshold_data = 0.75
        self.bounding_boxes = TensorSubscriber(self.client, "counting_bboxes", self._bboxes_callback)
        self.bounding_boxes_data = np.array([[0,0,0,0]], dtype=np.float32)

        self.detection_method = method(args)

        self.frame_in    = None
        self.frame_in_new = False
        self.aspect_ratio = 2012/1518

        self.frame_out    = None
        self.frame_out_new = False 

        self.closing = False

        self.n_frames = 0

    def _docker_command_callback(self, message):

        msg = echolib.MessageReader(message).readInt()
        print("Docker demo: got command {}".format(msg))    
        
        if msg == Command.DISABLE:
            self.enabled = False
        elif msg == Command.ENABLE:
            self.enabled = True
        elif msg in self.CAMERA_STREAMS:
            stream = self.CAMERA_STREAMS[msg]
            print(f"Switch streaming to {stream}")
            
            self.camera_stream = FrameSubscriber(self.client, stream, self._camera_stream_callback)

            
    def _camera_stream_callback(self, message):

        self.frame_in    = message.image
        self.frame_in_new = True

        self.n_frames += 1

        print("Docker demo: reading camera stream {}".format(self.n_frames))
        
    def _threshold_callback(self, message):
        msg = echolib.MessageReader(message).readFloat()
        print("Got new threshold", msg)
        self.threshold_data = msg
        
    def _bboxes_callback(self, message):
        self.bounding_boxes_data = message

    def process(self):
        
        while not self.closing:

            frame = None

            if self.frame_in_new:

                frame = self.frame_in
                self.frame_in_new = False

                expected_aspect_ratio = self.aspect_ratio
                img_aspect_ratio = frame.shape[1] / frame.shape[0]

                if expected_aspect_ratio > img_aspect_ratio:
                    crop = np.abs(int((frame.shape[1] / expected_aspect_ratio - frame.shape[0]) / 2))
                    frame = np.array(frame[crop:-crop,:], dtype=np.uint8)
                elif expected_aspect_ratio < img_aspect_ratio:
                    crop = np.abs(int((frame.shape[0] * expected_aspect_ratio - frame.shape[1]) / 2))
                    frame = np.array(frame[:,crop:-crop], dtype=np.uint8)
            
                bboxes = self.bounding_boxes_data.view(np.float32)
                if self.enabled and (bboxes.shape[0] > 1 or np.abs(bboxes).sum() > 0):
                    frame = self.detection_method.predict(frame, bboxes, self.threshold_data)

            if frame is not None:

                self.frame_out    = frame
                self.frame_out_new = True 
            
            time.sleep(0.01)
            
    def run(self, wait_sec=10, sleep_sec=0):

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

        print("Stop")

        thread.join()

