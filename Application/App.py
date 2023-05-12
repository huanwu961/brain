import sys
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        self.setGeometry(100, 100, 600, 300)
        self.setStyleSheet("background-color: #222222;")

        # Create container on left side of main GUI
        container = QWidget()
        container.setStyleSheet("background-color: #111111;")
        container.setFixedWidth(150)
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        # Create video button inside container
        video_button = QPushButton("Video")
        video_button.setStyleSheet("background-color: none; color: white; border: none;")
        video_button.clicked.connect(self.show_camera)
        container_layout.addWidget(video_button)

        # Add container to main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(container)
        main_layout.addStretch()

        # Create widget to display camera feed
        self.camera_widget = QLabel()
        self.camera_widget.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.camera_widget)

        # Create main widget and set layout
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create VideoCapture object for camera feed
        self.cap = cv2.VideoCapture(0)

        # Create QTimer object for camera feed update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(10)

    def show_camera(self):
        # Open camera window
        camera_window = QMainWindow()
        camera_window.setWindowTitle("Camera Feed")

        # Create widget to display camera feed
        camera_widget = QLabel()
        camera_widget.setAlignment(Qt.AlignCenter)
        camera_window.setCentralWidget(camera_widget)

        # Show camera window
        camera_window.show()

    def update_camera(self):
        # Read frame from camera
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to QImage and display in widget
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_widget.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())