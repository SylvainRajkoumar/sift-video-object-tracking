import os
from controller import Controller
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap
import cv2


class View(QMainWindow):

    def __init__(self):
        super(View, self).__init__()
        self.control = None
        self.webcam_enabled = False
        loadUi('viewui.ui',self)
        self.load_file_action.triggered.connect(self.get_image_path)
        self.start_webcam_action.triggered.connect(self.toggle_webcam)

    def toggle_webcam(self):
        self.control.toggle_webcam_timer()

    def setControl(self, c):
        self.control = c

    @pyqtSlot()
    def get_image_path(self):
        filename = QFileDialog.getOpenFileName(self,"Ouvrir une image", "./", "Image Files (*.png *.jpg *.bmp)")[0]
        if(os.path.exists(filename)):
            self.control.image_face_detection(filename)
    
    def updateView(self, image):
        height, width, channel = image.shape
        bytesPerLine = channel * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qImg))