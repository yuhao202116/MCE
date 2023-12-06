import os
import sys

import numpy as np
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, \
    QGroupBox
from PyQt5.QtWidgets import QLabel

from predict import predict


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.Oripixmap = None
        self.setWindowTitle('Lung Diagnose')
        self.setGeometry(100, 100, 600, 400)
        self.Ori_imagepath = ['./5.png', './6.png', './7.png', './8.png']
        self.CAM_imagepath = ['./1.png', './2.png', './3.png', './4.png']
        self.res_classify = ['COVID', 'Lung Opacity', 'Viral Pneumonia', 'Normal']
        self.image_index = 0
        self.model_weightpath = './MCE_model.tflite'
        self.initUI()

    def initUI(self):
        # 创建一个按钮
        self.camimage_label = QLabel(self)
        self.Oriimage_label = QLabel(self)

        self.Oripixmap = QPixmap(self.Ori_imagepath[self.image_index])
        self.Oriimage_label.setPixmap(self.Oripixmap.scaled(700, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.button2 = QPushButton('Previous', self)
        self.button2.clicked.connect(self.on_click4)
        self.button3 = QPushButton('Next', self)
        self.button3.clicked.connect(self.on_click3)

        self.button = QPushButton('模型诊断', self)
        self.button.clicked.connect(self.on_click1)
        self.button1 = QPushButton('生成CAM', self)
        self.button1.clicked.connect(self.on_click2)

        # 创建一个文本框
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setMinimumHeight(300)  # 设置最小高度为200

        # 创建布局并添加控件
        layout_image = QHBoxLayout()
        layout_image.addWidget(self.Oriimage_label)
        layout_image.addWidget(self.camimage_label)

        layout_buttons1 = QHBoxLayout()
        layout_buttons1.addWidget(self.button2)
        layout_buttons1.addWidget(self.button3)

        layout_buttons = QVBoxLayout()
        layout_buttons.addWidget(self.button)
        layout_buttons.addWidget(self.button1)
        layout_buttons.addWidget(self.text_edit)

        # 分组储存
        image_box = QGroupBox()
        image_box.setLayout(layout_image)

        button_box = QGroupBox()
        button_box.setLayout(layout_buttons)

        button_box1 = QGroupBox()
        button_box1.setLayout(layout_buttons1)

        container = QVBoxLayout()
        container.addWidget(image_box)
        container.addWidget(button_box)
        container.addWidget(button_box1)

        # 创建一个中心控件设置并布局
        central_widget = QWidget()
        central_widget.setLayout(container)
        self.setCentralWidget(central_widget)

    @pyqtSlot()
    def on_click1(self):
        pos, elapsed_time = predict(self.Ori_imagepath[self.image_index], self.model_weightpath)

        result_text = (f"花费的时间：{elapsed_time}\n"
                       f"诊断结果：{self.res_classify[np.argmax(pos)]}\n\n")
        for i, probability in enumerate(pos):
            result_text += f"{self.res_classify[i]:<20}" + f"Probability: {probability}   \n"
        self.text_edit.setText(result_text)
        self.camimage_label.setVisible(True)

    @pyqtSlot()
    def on_click2(self):
        image_filename = os.path.basename(self.CAM_imagepath[self.image_index])
        self.Campixmap = QPixmap(self.CAM_imagepath[self.image_index])
        self.camimage_label.setPixmap(self.Campixmap.scaled(700, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        CAM_text = f"CAM 生成完毕   诊断结果：{image_filename}"
        self.text_edit.setText(CAM_text)
        self.camimage_label.setVisible(True)

    @pyqtSlot()
    def on_click3(self):
        self.image_index += 1
        if self.image_index > 3:self.image_index = 0
        self.Oripixmap = QPixmap(self.Ori_imagepath[self.image_index])
        self.Oriimage_label.setPixmap(self.Oripixmap.scaled(700, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.camimage_label.setVisible(False)
    @pyqtSlot()
    def on_click4(self):
        self.image_index -= 1
        if self.image_index < 0: self.image_index = 3
        self.Oripixmap = QPixmap(self.Ori_imagepath[self.image_index])
        self.Oriimage_label.setPixmap(self.Oripixmap.scaled(700, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.camimage_label.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.resize(1500, 800)
    mainWin.setStyleSheet("font-size: 30px;")
    mainWin.show()
    sys.exit(app.exec_())
