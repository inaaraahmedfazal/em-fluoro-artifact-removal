# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/vtkMainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(899, 949)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setKerning(True)
        self.centralwidget.setFont(font)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.imageWidget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageWidget.sizePolicy().hasHeightForWidth())
        self.imageWidget.setSizePolicy(sizePolicy)
        self.imageWidget.setObjectName("imageWidget")
        self.videoGridLayout = QtWidgets.QGridLayout(self.imageWidget)
        self.videoGridLayout.setObjectName("videoGridLayout")
        self.removedImageLabel = QtWidgets.QLabel(self.imageWidget)
        self.removedImageLabel.setText("")
        self.removedImageLabel.setObjectName("removedImageLabel")
        self.videoGridLayout.addWidget(self.removedImageLabel, 0, 2, 1, 1)
        self.preImageLabel = QtWidgets.QLabel(self.imageWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preImageLabel.sizePolicy().hasHeightForWidth())
        self.preImageLabel.setSizePolicy(sizePolicy)
        self.preImageLabel.setMinimumSize(QtCore.QSize(0, 434))
        self.preImageLabel.setText("")
        self.preImageLabel.setObjectName("preImageLabel")
        self.videoGridLayout.addWidget(self.preImageLabel, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.videoGridLayout.addItem(spacerItem, 0, 4, 1, 1)
        self.fiducialsImageLabel = QtWidgets.QLabel(self.imageWidget)
        self.fiducialsImageLabel.setText("")
        self.fiducialsImageLabel.setObjectName("fiducialsImageLabel")
        self.videoGridLayout.addWidget(self.fiducialsImageLabel, 0, 3, 1, 1)
        self.gridLayout_5.addWidget(self.imageWidget, 0, 0, 1, 2)
        self.vtkWidget = QtWidgets.QWidget(self.centralwidget)
        self.vtkWidget.setObjectName("vtkWidget")
        self.vtkGridLayout = QtWidgets.QGridLayout(self.vtkWidget)
        self.vtkGridLayout.setObjectName("vtkGridLayout")
        self.gridLayout_5.addWidget(self.vtkWidget, 1, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 899, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imgFrame = QtWidgets.QFrame(self.dockWidgetContents)
        self.imgFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imgFrame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.imgFrame.setObjectName("imgFrame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.imgFrame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_4 = QtWidgets.QFrame(self.imgFrame)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.carmCalDicom1Field = QtWidgets.QLineEdit(self.frame_4)
        self.carmCalDicom1Field.setObjectName("carmCalDicom1Field")
        self.gridLayout_6.addWidget(self.carmCalDicom1Field, 5, 0, 1, 1)
        self.browseCarmCalDicom1Button = QtWidgets.QToolButton(self.frame_4)
        self.browseCarmCalDicom1Button.setObjectName("browseCarmCalDicom1Button")
        self.gridLayout_6.addWidget(self.browseCarmCalDicom1Button, 5, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame_4)
        self.label_3.setObjectName("label_3")
        self.gridLayout_6.addWidget(self.label_3, 0, 0, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.frame_4)
        self.label_4.setObjectName("label_4")
        self.gridLayout_6.addWidget(self.label_4, 4, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        self.label_5.setObjectName("label_5")
        self.gridLayout_6.addWidget(self.label_5, 6, 0, 1, 1)
        self.carmCalDicom2Field = QtWidgets.QLineEdit(self.frame_4)
        self.carmCalDicom2Field.setObjectName("carmCalDicom2Field")
        self.gridLayout_6.addWidget(self.carmCalDicom2Field, 7, 0, 1, 1)
        self.browseCarmCalDicom2Button = QtWidgets.QToolButton(self.frame_4)
        self.browseCarmCalDicom2Button.setObjectName("browseCarmCalDicom2Button")
        self.gridLayout_6.addWidget(self.browseCarmCalDicom2Button, 7, 1, 1, 1)
        self.runCarmCalButton = QtWidgets.QPushButton(self.frame_4)
        self.runCarmCalButton.setObjectName("runCarmCalButton")
        self.gridLayout_6.addWidget(self.runCarmCalButton, 8, 0, 1, 2)
        self.gridLayout_3.addWidget(self.frame_4, 2, 0, 1, 1)
        self.frame = QtWidgets.QFrame(self.imgFrame)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.trackerCTRegFileField = QtWidgets.QLineEdit(self.frame)
        self.trackerCTRegFileField.setObjectName("trackerCTRegFileField")
        self.gridLayout.addWidget(self.trackerCTRegFileField, 8, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 2)
        self.intCalFileField = QtWidgets.QLineEdit(self.frame)
        self.intCalFileField.setObjectName("intCalFileField")
        self.gridLayout.addWidget(self.intCalFileField, 6, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.frame)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 16, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 15, 0, 1, 2)
        self.browseFiducialFileButton = QtWidgets.QToolButton(self.frame)
        self.browseFiducialFileButton.setObjectName("browseFiducialFileButton")
        self.gridLayout.addWidget(self.browseFiducialFileButton, 3, 1, 1, 1)
        self.trackerToggle = QtWidgets.QPushButton(self.frame)
        self.trackerToggle.setCheckable(True)
        self.trackerToggle.setObjectName("trackerToggle")
        self.gridLayout.addWidget(self.trackerToggle, 17, 0, 1, 2)
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 13, 0, 1, 1)
        self.browseIntCalButton = QtWidgets.QToolButton(self.frame)
        self.browseIntCalButton.setObjectName("browseIntCalButton")
        self.gridLayout.addWidget(self.browseIntCalButton, 6, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 7, 0, 1, 2)
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 12, 0, 1, 2)
        self.fiducialFileField = QtWidgets.QLineEdit(self.frame)
        self.fiducialFileField.setObjectName("fiducialFileField")
        self.gridLayout.addWidget(self.fiducialFileField, 3, 0, 1, 1)
        self.confirmParametersButton = QtWidgets.QPushButton(self.frame)
        self.confirmParametersButton.setObjectName("confirmParametersButton")
        self.gridLayout.addWidget(self.confirmParametersButton, 11, 0, 1, 2)
        self.browseTrackerCTRegButton = QtWidgets.QToolButton(self.frame)
        self.browseTrackerCTRegButton.setObjectName("browseTrackerCTRegButton")
        self.gridLayout.addWidget(self.browseTrackerCTRegButton, 8, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 5, 0, 1, 2)
        self.streamToggle = QtWidgets.QPushButton(self.frame)
        self.streamToggle.setCheckable(True)
        self.streamToggle.setObjectName("streamToggle")
        self.gridLayout.addWidget(self.streamToggle, 14, 0, 1, 2)
        self.refCentroidsField = QtWidgets.QLineEdit(self.frame)
        self.refCentroidsField.setObjectName("refCentroidsField")
        self.gridLayout.addWidget(self.refCentroidsField, 10, 0, 1, 1)
        self.browseRefCentroidsButton = QtWidgets.QToolButton(self.frame)
        self.browseRefCentroidsButton.setObjectName("browseRefCentroidsButton")
        self.gridLayout.addWidget(self.browseRefCentroidsButton, 10, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.frame)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 9, 0, 1, 2)
        self.gridLayout_3.addWidget(self.frame, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem1, 5, 0, 1, 2)
        self.frame_2 = QtWidgets.QFrame(self.imgFrame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_9 = QtWidgets.QLabel(self.frame_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 1, 0, 1, 2)
        self.simDicomField = QtWidgets.QLineEdit(self.frame_2)
        self.simDicomField.setObjectName("simDicomField")
        self.gridLayout_2.addWidget(self.simDicomField, 2, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 0, 0, 1, 2)
        self.browseSimDicomButton = QtWidgets.QToolButton(self.frame_2)
        self.browseSimDicomButton.setObjectName("browseSimDicomButton")
        self.gridLayout_2.addWidget(self.browseSimDicomButton, 2, 1, 1, 1)
        self.runSimButton = QtWidgets.QPushButton(self.frame_2)
        self.runSimButton.setObjectName("runSimButton")
        self.gridLayout_2.addWidget(self.runSimButton, 3, 0, 1, 2)
        self.gridLayout_3.addWidget(self.frame_2, 3, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.imgFrame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_12 = QtWidgets.QLabel(self.frame_3)
        self.label_12.setObjectName("label_12")
        self.gridLayout_4.addWidget(self.label_12, 2, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame_3)
        self.label_14.setObjectName("label_14")
        self.gridLayout_4.addWidget(self.label_14, 3, 0, 1, 1)
        self.CarmTyLCD = QtWidgets.QLCDNumber(self.frame_3)
        self.CarmTyLCD.setFrameShadow(QtWidgets.QFrame.Plain)
        self.CarmTyLCD.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.CarmTyLCD.setObjectName("CarmTyLCD")
        self.gridLayout_4.addWidget(self.CarmTyLCD, 2, 1, 1, 1)
        self.CarmTzLCD = QtWidgets.QLCDNumber(self.frame_3)
        self.CarmTzLCD.setFrameShadow(QtWidgets.QFrame.Plain)
        self.CarmTzLCD.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.CarmTzLCD.setObjectName("CarmTzLCD")
        self.gridLayout_4.addWidget(self.CarmTzLCD, 3, 1, 1, 1)
        self.CarmTxLCD = QtWidgets.QLCDNumber(self.frame_3)
        self.CarmTxLCD.setFrameShadow(QtWidgets.QFrame.Plain)
        self.CarmTxLCD.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.CarmTxLCD.setObjectName("CarmTxLCD")
        self.gridLayout_4.addWidget(self.CarmTxLCD, 1, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.frame_3)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.frame_3)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 0, 0, 1, 2)
        self.gridLayout_3.addWidget(self.frame_3, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.imgFrame)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.dockWidget.raise_()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.browseCarmCalDicom1Button.setText(_translate("MainWindow", "..."))
        self.label_3.setText(_translate("MainWindow", "C-arm Calibration"))
        self.label_4.setText(_translate("MainWindow", "DICOM - Axis 1"))
        self.label_5.setText(_translate("MainWindow", "DICOM - Axis 2"))
        self.browseCarmCalDicom2Button.setText(_translate("MainWindow", "..."))
        self.runCarmCalButton.setText(_translate("MainWindow", "Run C-arm Calibration"))
        self.label.setText(_translate("MainWindow", "Load Metal Fiducial Points (from CT)"))
        self.label_13.setText(_translate("MainWindow", "Magnetic Tracking"))
        self.browseFiducialFileButton.setText(_translate("MainWindow", "..."))
        self.trackerToggle.setText(_translate("MainWindow", "Start/Stop Tracking"))
        self.label_7.setText(_translate("MainWindow", "X-Ray/Fluoro Streaming"))
        self.browseIntCalButton.setText(_translate("MainWindow", "..."))
        self.label_8.setText(_translate("MainWindow", "Load CT/Tracking Registration"))
        self.confirmParametersButton.setText(_translate("MainWindow", "Confirm"))
        self.browseTrackerCTRegButton.setText(_translate("MainWindow", "..."))
        self.label_2.setText(_translate("MainWindow", "Load Fluoro Intrinsic Matrix"))
        self.streamToggle.setText(_translate("MainWindow", "Start/Stop Video Stream"))
        self.browseRefCentroidsButton.setText(_translate("MainWindow", "..."))
        self.label_15.setText(_translate("MainWindow", "Load Reference Centroids"))
        self.label_9.setText(_translate("MainWindow", "Load DICOM File"))
        self.label_6.setText(_translate("MainWindow", "Simulated Workflow"))
        self.browseSimDicomButton.setText(_translate("MainWindow", "..."))
        self.runSimButton.setText(_translate("MainWindow", "Run Simulation"))
        self.label_12.setText(_translate("MainWindow", "Ty"))
        self.label_14.setText(_translate("MainWindow", "Tz"))
        self.label_11.setText(_translate("MainWindow", "Tx"))
        self.label_10.setText(_translate("MainWindow", "C-arm Pose"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+E"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
