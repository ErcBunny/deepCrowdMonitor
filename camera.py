import os
import cv2
import detect
from base_camera import BaseCamera
import numpy as np
import time

imgSrc = None
stream_started = False
boxes = None

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        global imgSrc
        global boxes
        while boxes is None:
            time.sleep(1)
        while True:
            yield cv2.imencode('.jpg', boxes)[1].tobytes()
