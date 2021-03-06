from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QMainWindow, QPushButton, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QImage
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QMutex
import numpy as np
import datetime as dt
from image_segmentation import ImageSegmentation 
import pygame
import time

class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()       

        self.video_name = None
        self.image_name = None
        self.camera_id = 0

        self.image_segmentation = ImageSegmentation()

        self.central_widget = QWidget()
        self.button_frame = QPushButton('Start zdjecie', self.central_widget)
        self.button_movie = QPushButton('Start nagranie', self.central_widget)
        self.button_live = QPushButton('Start na zywo', self.central_widget)
        self.button_stop = QPushButton('Stop', self.central_widget)
        self.select_video = QPushButton('Wybierz nagranie', self.central_widget)
        self.select_image = QPushButton('Wybierz zdjecie', self.central_widget)
        self.camera_label = QLabel()
        self.camera_combo = QComboBox()
        self.image_view = QLabel()

        self.camera_label.setText('Wybierz swoje urzadzenie nagrywajace')

        index = 0        
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                self.camera_combo.addItem(str(index))
                cap.release()
            index += 1
            i -= 1
        

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_live)
        self.layout.addWidget(self.button_frame)
        self.layout.addWidget(self.button_movie)
        self.layout.addWidget(self.image_view)
        self.layout.addWidget(self.button_stop)
        self.layout.addWidget(self.select_video)
        self.layout.addWidget(self.select_image)
        self.layout.addWidget(self.camera_label)
        self.layout.addWidget(self.camera_combo)        
        self.setCentralWidget(self.central_widget)

        self.button_frame.clicked.connect(self.start_image)
        self.button_movie.clicked.connect(self.start_movie)
        self.button_live.clicked.connect(self.start_live)
        self.button_stop.clicked.connect(self.stop)
        self.select_image.clicked.connect(self.openFileDialogImage)
        self.select_video.clicked.connect(self.openFileDialogVideo)
        self.camera_combo.currentIndexChanged.connect(self.selection_change)
        

    def selection_change(self):
        print(self.camera_combo.currentText())
        self.camera_id = int(self.camera_combo.currentText())      

    def openFileDialogImage(self):
        self.openFileNameDialog(0)
        self.show()

    def openFileDialogVideo(self):
        self.openFileNameDialog(1)
        self.show()

    def openFileNameDialog(self, fileType):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileType == 0:
            self.image_name = fileName
        if fileType == 1:
            self.video_name = fileName

    def start_image(self):
        self.image = cv2.imread(self.image_name)
        _, self.image = self.image_segmentation.analize_photo(self.image, 0)
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image = self.image.scaled(800, 700)       
        self.image_view.setPixmap(QtGui.QPixmap.fromImage(self.image))


    def start_movie(self):
        self.thread = VideoThread(self.video_name)
        self.thread.change_pixmap_signal.connect(self.update_Video)
        self.thread.start()

    def start_live(self):
        self.thread = LiveThread(self.camera_id)
        self.thread.change_pixmap_signal.connect(self.update_Live)
        self.thread.start()

    def stop(self):
        self.thread.stop()   


    @pyqtSlot(np.ndarray)
    def update_Video(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img, transform=1)
        self.image_view.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_Live(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img, transform=1)        
        self.image_view.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, transform):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(700, 800)

        t = QtGui.QTransform()
        t.rotate(90)

        if transform:
            p = p.transformed(t)

        return QPixmap.fromImage(p)


class LiveThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_id):
        super().__init__()

        self._run_flag = True
        self.camera_id = camera_id
        self.image_segmentation = ImageSegmentation()
        self.last_time = dt.datetime.now()

        self.mutex = QMutex()

    def play_warning(self):
        self.mutex.lock()

        current_time = dt.datetime.now()

        if  (current_time - self.last_time).seconds > 2:                        
            pygame.mixer.init()        
            pygame.mixer.music.load("sound.ogg")
            pygame.mixer.music.play()           

            self.last_time = dt.datetime.now()

        self.mutex.unlock()      

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                check, cv_img = self.image_segmentation.analize_video(cv_img, 0)
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)                
                self.change_pixmap_signal.emit(cv_img)

                if not check:
                    self.play_warning()

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_name):
        super().__init__()        
        self.video_name = video_name        

        self._run_flag = True
        self.image_segmentation = ImageSegmentation()
        self.last_time = dt.datetime.now()
        
        self.not_check_counter = 0
        
        self.mutex = QMutex()

    def play_warning(self):
        self.mutex.lock()

        current_time = dt.datetime.now()

        if  (current_time - self.last_time).seconds > 2:                        
            pygame.mixer.init()        
            pygame.mixer.music.load("sound.ogg")
            pygame.mixer.music.play()           

            self.last_time = dt.datetime.now()

        self.mutex.unlock()        


    def run(self):
        cap = cv2.VideoCapture(self.video_name)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                check, cv_img = self.image_segmentation.analize_video(cv_img, 0)
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                self.change_pixmap_signal.emit(cv_img)

                if not check:
                    self.not_check_counter += 1

                    if self.not_check_counter >= 15:                    
                        self.play_warning()
                else:
                    self.not_check_counter = 0

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())