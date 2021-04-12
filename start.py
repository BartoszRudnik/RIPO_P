from PyQt5.QtWidgets import QApplication

from models import Camera
from views import StartWindow

app = QApplication([])
start_window = StartWindow()
start_window.show()
app.exit(app.exec_())