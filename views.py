from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QMainWindow, QPushButton
from PyQt5.QtGui import QPixmap, QImage
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget()
        self.button_frame = QPushButton('Start zdjecie', self.central_widget)
        self.button_movie = QPushButton('Start nagranie', self.central_widget)
        self.button_live = QPushButton('Start na zywo', self.central_widget)
        self.button_stop = QPushButton('Stop', self.central_widget)
        self.image_view = QLabel()


        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_live)
        self.layout.addWidget(self.button_frame)
        self.layout.addWidget(self.button_movie)
        self.layout.addWidget(self.image_view)
        self.layout.addWidget(self.button_stop)
        self.setCentralWidget(self.central_widget)

        self.button_frame.clicked.connect(self.start_image)
        self.button_movie.clicked.connect(self.start_movie)
        self.button_live.clicked.connect(self.start_live)
        self.button_stop.clicked.connect(self.stop)

    def start_image(self):
        self.image = cv2.imread('zdjecie.png')
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image = self.image.scaled(640, 350, Qt.KeepAspectRatio)
        self.image_view.setPixmap(QtGui.QPixmap.fromImage(self.image))


    def start_movie(self):
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_Video)
        self.thread.start()

    def start_live(self):
        self.thread = LiveThread()
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
        qt_img = self.convert_cv_qt(cv_img, transform=0)
        self.image_view.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, transform):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 350, Qt.KeepAspectRatio)

        t = QtGui.QTransform()
        t.rotate(90)

        if transform:
            p = p.transformed(t)

        return QPixmap.fromImage(p)


class LiveThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture('nagranie.mp4')
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())