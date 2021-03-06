import cv2
from utils.object_tracking import ObjectTracking
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog
import numpy as np

class Controller(object):

    def __init__(self, view):
        self.view = view
        self.object_tracking = ObjectTracking()
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.webcam_processing)
        self.capture = cv2.VideoCapture(0)
        self.webcam_enabled = False

    def load_image(self, image_path):
        self.object_tracking.change_reference_image(image_path)

    def toggle_webcam_timer(self):
        if self.webcam_enabled:
            self.webcam_timer.stop()
            blank = np.zeros([500,500,3])
            self.view.updateView(blank)
            self.capture.release()
        else:
            self.capture.open(0)
            self.webcam_timer.start(33)

        self.webcam_enabled = not self.webcam_enabled

    def webcam_processing(self):
        ret, image = self.capture.read()
        if ret:
            result_image = self.object_tracking.get_tracking_result_image(image)
            self.view.updateView(result_image)

    def __del__(self):
        self.capture.release()
        self.webcam_timer.stop()