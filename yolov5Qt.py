from math import radians
import json
from time import sleep
import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtGui, QtWidgets 
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
import threading
import datetime
from  form import Ui_Form
import sys
app = QtWidgets.QApplication(sys.argv)
ui=Ui_Form()


if __name__ == '__main__':
    w=QMainWindow()
    ui.setupUi(w)
    w.setWindowTitle("yolov5素材抓取")
    w.show()  # 显示窗体
    sys.exit(app.exec_())  # 程序关闭时退出进程