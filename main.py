from PySide6.QtQuickControls2 import QQuickStyle
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickImageProvider
from PySide6.QtQuickControls2 import QQuickStyle
from backend import EmotionDetectorBackend

class CameraImageProvider(QQuickImageProvider):
    """Image provider for camera frames"""
    def __init__(self):
        super().__init__(QQuickImageProvider.Pixmap)
        self.current_frame = QImage(640, 480, QImage.Format_RGB888)
        self.current_frame.fill(Qt.gray)
        self.frame_counter = 0

    def requestPixmap(self, id, size, requestedSize):
        """Return the current camera frame as pixmap"""
        if self.current_frame.isNull():
            self.current_frame = QImage(640, 480, QImage.Format_RGB888)
            self.current_frame.fill(Qt.gray)

        pixmap = QPixmap.fromImage(self.current_frame)

        if size:
            size.setWidth(pixmap.width())
            size.setHeight(pixmap.height())

        return pixmap

    def update_frame(self, frame):
        """Update the current frame"""
        if not frame.isNull():
            self.current_frame = frame.copy()
            self.frame_counter += 1

def main():
    # Create application
    app = QApplication(sys.argv)

    # Set application details
    app.setApplicationName("BUKEmotionAI Studio Pro (2 Emotions)")
    app.setOrganizationName("BUK AILab")
    app.setApplicationVersion("3.0")

    # Set Material style
    QQuickStyle.setStyle("Material")

    # Create backend
    backend = EmotionDetectorBackend()

    # Create image provider for camera feed
    image_provider = CameraImageProvider()

    # Create QML engine
    engine = QQmlApplicationEngine()

    # Register image provider
    engine.addImageProvider("camera", image_provider)

    # Connect backend frame signals to image provider
    backend.frame_ready.connect(image_provider.update_frame)

    # Expose backend to QML
    engine.rootContext().setContextProperty("backend", backend)

    # Load QML
    qml_file = Path(__file__).parent / "main.qml"

    if not qml_file.exists():
        print(f"Error: QML file not found at {qml_file}")
        sys.exit(-1)

    print(f"Loading QML from: {qml_file}")
    engine.load(QUrl.fromLocalFile(str(qml_file)))

    if not engine.rootObjects():
        print("Error: No QML root objects loaded")
        sys.exit(-1)

    print("QML loaded successfully")

    # Initialize backend
    backend.initialize()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
