import sys
import vtk
import numpy as np
import cv2
import py_ct_reg as ct
import os
import HandEyeCalLogic as he
import pydicom as dicom
from eval_dereflection_cl import Evaluator
from PyQt5 import QtCore, QtGui, QtWidgets
from vtkMainWindow_ui import Ui_MainWindow
from skimage import measure
from pycpd import AffineRegistration
import matplotlib.pyplot as plt
import csv
import matlab.engine
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from sksurgerynditracker.nditracker import NDITracker

POSITIONER_PRIMARY_ANGLE_INCREMENT = 1
POSITIONER_SECONDARY_ANGLE_INCREMENT = 1
THRESHOLD_VALUE_1 = 0.3
THRESHOLD_VALUE_2 = 0.25
REORDER_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10]

FRAME_INTERVAL_MS = 1000

# Might need to change this based on how many COM ports show up on in device manager
NUM_PORTS = 2
SPHERE_RADIUS = 5

class DeepDecompositionViewer(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__()
        ckptsdir = "C:/Users/iahmedf/Documents/em-fluoro-artifact-removal/checkpoints"
        self.outdir = "C:/Users/iahmedf/Documents/em-fluoro-artifact-removal/image_files"
        # os.mkdir(f"{self.outdir}/preprocessed")
        # os.mkdir(f"{self.outdir}/fiducials")
        # os.mkdir(f"{self.outdir}/removed")
        self.eval = Evaluator(ckptsdir)
        self.setupUi(self)

        # Tracker widget setup
        self.ren = vtk.vtkRenderer()
        self.qvtkwin = QVTKRenderWindowInteractor(self.vtkWidget)
        self.qvtkwin.GetRenderWindow().AddRenderer(self.ren)
        self.ren.SetBackground(0.1, 0.2, 0.4)
        self.iren = self.qvtkwin.GetRenderWindow().GetInteractor()
        self.vtkGridLayout.addWidget(self.qvtkwin, 0, 0)
        self.updateTimer = QtCore.QTimer()

        # Tracker setup
        self.tracker = None
        self.isTrackerInitialized = False
        self.trackerSettings = {
            "tracker type": "aurora",
            "ports to probe": NUM_PORTS,
            "verbose": True
        }
        self.trackedTipSource = vtk.vtkSphereSource()
        self.trackedTipActor = vtk.vtkActor()
        self.trackedTipMapper = vtk.vtkPolyDataMapper()
        self.trackedTipTransform = vtk.vtkTransform()

        self.centers_original_ref = np.zeros((12, 2))
        self.fiducialWorldPts = []
        self.intCalMat = np.eye(3)
        self.trackerCTRegMat = np.eye(4)

        # Transformation from CT to tracker (loaded as CSV)
        self.trackerCTRegTransform = vtk.vtkTransform()

        # Transformation from CT to fluoro (calculated per frame)
        self.fluoroCTRegTransform = vtk.vtkTransform()
        
        self.fluoroTransform = vtk.vtkTransform()
        self.fluoroCubeSource = vtk.vtkCubeSource()
        self.fluoroCubeActor = vtk.vtkActor()
        self.fluoroCubeMapper = vtk.vtkPolyDataMapper()

        self.connectSignalsSlots()
        self.setupVtkObjects()

        self.fluoroStream = None
        self.isStreaming = False
        self.fluoroFrameCtr = 0
        self.eng = matlab.engine.start_matlab()

        self.updateTimer.timeout.connect(self.updateScene)
        self.updateTimer.start(FRAME_INTERVAL_MS)

    def setupVtkObjects(self):
        """Initializes and connects VTK objects"""
        self.fluoroCubeSource.SetCenter(0, 0, 0)
        self.fluoroCubeSource.SetXLength(100)
        self.fluoroCubeSource.SetYLength(100)
        self.fluoroCubeSource.SetZLength(100)

        self.trackedTipTransform.Identity()
        self.trackedTipSource.SetCenter(0, 0, 0)
        self.trackedTipSource.SetRadius(SPHERE_RADIUS)
        self.trackedTipMapper.SetInputConnection(self.trackedTipSource.GetOutputPort())
        self.trackedTipActor.SetMapper(self.trackedTipMapper)

        # Initialize all transforms to identity
        self.trackerCTRegTransform.Identity()
        self.fluoroCTRegTransform.Identity()
        self.fluoroTransform.Identity()

        # Mapper/actor for fluoro camera cube
        self.fluoroCubeMapper.SetInputConnection(self.fluoroCubeSource.GetOutputPort())
        self.fluoroCubeActor.SetMapper(self.fluoroCubeMapper)

        # Transformation chain for fluoro camera cube
        self.fluoroTransform.PostMultiply()
        self.fluoroTransform.Concatenate(self.fluoroCTRegTransform)
        self.fluoroTransform.Concatenate(self.trackerCTRegTransform)
        
    def connectSignalsSlots(self):
        self.browseFiducialFileButton.clicked.connect(self.browseFiducialFile)
        self.browseIntCalButton.clicked.connect(self.browseIntCal)
        self.browseCarmCalDicom1Button.clicked.connect(self.browseCarmCalDicom1)
        self.browseCarmCalDicom2Button.clicked.connect(self.browseCarmCalDicom2)
        self.browseTrackerCTRegButton.clicked.connect(self.browseTrackerCTReg)
        self.browseSimDicomButton.clicked.connect(self.browseSimDicom)
        self.browseRefCentroidsButton.clicked.connect(self.browseRefCentroids)
        
        self.trackerToggle.toggled.connect(self.startTracker)
        self.streamToggle.toggled.connect(self.handleStreamToggle)
        self.confirmParametersButton.clicked.connect(self.confirmParameters)
        self.runCarmCalButton.clicked.connect(self.runCarmCal)
        self.runSimButton.clicked.connect(self.runSimSingleImg)
    def browseRefCentroids(self):
        """Browsing window for CSV file containing reference centroids"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.refCentroidsField.setText(fname)
    def browseImage(self):
        """Browsing window for preprocessed image"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.imageField.setText(fname)
        self.preImageLabel.setPixmap(QtGui.QPixmap(fname))
    def browseFiducialFile(self):
        """Browsing window for fiducial world points (predetermined from CT)"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.fiducialFileField.setText(fname)
    def browseIntCal(self):
        """Browsing window for file containing intrinsic calibration matrix"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.intCalFileField.setText(fname)
    def browseCarmCalDicom1(self):
        """Browsing window for Axis 1 DICOM file for C-arm calibration"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.carmCalDicom1Field.setText(fname)
    def browseCarmCalDicom2(self):
        """Browsing window for Axis 2 DICOM file for C-arm calibration"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.carmCalDicom2Field.setText(fname)
    def browseTrackerCTReg(self):
        """Browsing window for CT-Tracking registration extrinsic matrix"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.trackerCTRegFileField.setText(fname)
    def browseSimDicom(self):
        """Browsing window for DICOM file for simulated workflow"""
        fname, d = QtWidgets.QFileDialog.getOpenFileName()
        self.simDicomField.setText(fname)
    def processImage(self):
        """Calls eval_dereflection algorithm"""
        outfile_fiducials, outfile_removed = self.eval.run_eval(self.imageField.text())
        self.postImageLabel.setPixmap(QtGui.QPixmap(outfile_removed))
        #self.getCarmPose()

    def startTracker(self):
        """ Starts NDI Aurora (magnetic) or Polaris (optical) tracker and sets up VTK tracked objects"""
        if self.trackerToggle.isChecked():
            if not self.isTrackerInitialized:
                try:
                    self.tracker = NDITracker(self.trackerSettings)
                    self.isTrackerInitialized = True
                    self.tracker.use_quaternions = False
                    
                except:
                    print("Unable to connect to NDI Tracked device")
                    self.isTrackerInitialized = False
                    self.trackerToggle.setChecked(False)
            if self.isTrackerInitialized:
                self.tracker.start_tracking()
                self.trackedTipActor.SetUserTransform(self.trackedTipTransform)
                self.ren.AddActor(self.trackedTipActor)
                self.qvtkwin.GetRenderWindow().Render()

        else:
            if self.isTrackerInitialized:
                self.isTrackerInitialized = False
                self.tracker.stop_tracking()
                self.ren.RemoveActor(self.trackedTipActor)
                self.qvtkwin.GetRenderWindow().Render()
                print("Tracking Stopped")
                    

    def runSimNoRemoval(self):
        # Takes in a DICOM file and calculates the C-arm pose for every frame, then creates visualization

        # Reads in DICOM and parameters
        dcmname = self.simDicomField.text()
        info = dicom.dcmread(dcmname)
        angleIncrement = info.PositionerSecondaryAngleIncrement
        uniqueAngles = np.unique(angleIncrement)
        numUniqueAngles = len(uniqueAngles)

        # Sets up visualization
        self.fluoroCubeActor.SetUserTransform(self.fluoroCTRegTransform)
        self.ren.AddActor(self.fluoroCubeActor)

        # Saves all C-arm positions
        translationVectorTotal = np.zeros((3, numUniqueAngles-20))
        
        # Calculates reference centroids
        indexRef = int(np.ceil(numUniqueAngles/2.0)) - 1
        framesIndex = np.where(angleIncrement == uniqueAngles[indexRef])
        frameIndex = int(np.ceil(np.median(framesIndex)))
        refimg = info.pixel_array[frameIndex]

        img_float = refimg/float(np.amax(refimg))
        img_float = img_float.astype(np.float32)

        # Threshold Segmentation
        bw = img_float < 0.3
        bwint = bw.astype(np.float32)
        L = measure.label(bw, connectivity=2)
        S = measure.regionprops(L)
        S_area = [S_i.area for S_i in S]
        S_area_arr = np.array(S_area)
        S_area_idx = np.where((S_area_arr >= 200) & (S_area_arr <= 2000))
        S_area_idx += np.ones(len(S_area_idx))
        bw2 = np.isin(L, S_area_idx)
        bw2flt = bw2.astype(np.float32)
        L2 = measure.label(bw2flt, connectivity=2)

        C = measure.regionprops(L2)
        self.centers_original_ref = np.array([[C_i.centroid[1], C_i.centroid[0]] for C_i in C])

        # Finds centroids and calculates pose for each frame
        for indexRef in range(numUniqueAngles - 20):
            framesIndex = np.where(angleIncrement == uniqueAngles[indexRef])
            frameIndex = int(np.ceil(np.median(framesIndex)))
            print(f"--frameIndex {frameIndex}--")
            img = info.pixel_array[frameIndex]

            inv = self.getCarmPose(img)
            translationVectorTotal[:, indexRef] = inv[0:3, 3]
            self.ren.ResetCameraClippingRange()
            self.qvtkwin.GetRenderWindow().Render()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(translationVectorTotal[0, :], translationVectorTotal[1, :], translationVectorTotal[2, :], s=10, c='b', marker='s')
        ax.set_xlim3d(-100, 100)
        with open('translationVectorTotal.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for t in np.transpose(translationVectorTotal):
                writer.writerow(t)

        plt.show()


    def runSimSingleImg(self):
        imgname = self.simDicomField.text()
        img = cv2.imread(imgname)

        inv = self.getCarmPose(img)
        print(inv)
    def confirmParameters(self):
        """Add fiducials to VTK widget scene"""
        self.readFiducialFile()
        self.readRefCentroidsFile()
        self.addFiducialsToScene()
        self.readIntCalFromFile()
        self.readTrackerCtRegFromFile()

    def readRefCentroidsFile(self):
        refcentroids_fname = self.refCentroidsField.text()
        with open(refcentroids_fname, 'r') as f:
            lines = f.readlines()
            for i in range(12):
                line_arr = lines[i].split(',')
                self.centers_original_ref[i, 0] = float(line_arr[0].strip())
                self.centers_original_ref[i, 1] = float(line_arr[1].strip())
                
    def readIntCalFromFile(self):
        intcal_fname = self.intCalFileField.text()
        
        with open(intcal_fname, 'r') as f:
            lines = f.readlines()
            for i in range(3):
                line_arr = lines[i].split(',')
                for j in range(3):
                    self.intCalMat[i, j] = float(line_arr[j].strip())

    def readTrackerCtRegFromFile(self):
        extcal_fname = self.trackerCTRegFileField.text()
        with open(extcal_fname, 'r') as f:
            lines = f.readlines()
            for i in range(4):
                line_arr = lines[i].split(',')
                for j in range(4):
                    self.trackerCTRegMat[i, j] = float(line_arr[j].strip())
        self.trackerCTRegTransform.SetMatrix(np.reshape(self.trackerCTRegMat, 16))
                
    def addFiducialsToScene(self):
        sphSrc = vtk.vtkSphereSource()
        sphSrc.SetRadius(5)
        sphSrc.SetCenter(0, 0, 0)
        for pt in self.fiducialWorldPts:
            tfrm = vtk.vtkTransform()
            ptMat = np.eye(4)
            ptMat[0:3, 3] = pt
            tfrm.SetMatrix(np.reshape(ptMat, 16))

            sphAct = vtk.vtkActor()
            sphMap = vtk.vtkPolyDataMapper()

            sphMap.SetInputConnection(sphSrc.GetOutputPort())
            sphAct.SetMapper(sphMap)
            sphAct.SetUserTransform(tfrm)
            self.ren.AddActor(sphAct)


    def getCarmPose(self, img):

        # Gets C-arm pose from fiducial image
        
        img_float = img/float(np.amax(img))
        img_float = img_float.astype(np.float32)

        # Threshold Segmentation
        bw = img_float < 0.35
        L = measure.label(bw, connectivity=2)
        S = measure.regionprops(L)
        S_area = [S_i.area for S_i in S]
        S_area_arr = np.array(S_area)
        S_area_idx = np.where((S_area_arr >= 200) & (S_area_arr <= 600))
        S_area_idx += np.ones(len(S_area_idx))
        bw2 = np.isin(L, S_area_idx)
        bw2flt = bw2.astype(np.float32)
        cv2.imshow("bw2flt", bw2flt)
        cv2.waitKey(0)
        L2 = measure.label(bw2, connectivity=2)

        C = measure.regionprops(L2)
        centers_original = np.array([[C_i.centroid[1], C_i.centroid[0]] for C_i in C])

        # Registration with reference centroids
        reg = AffineRegistration(X=self.centers_original_ref, Y=centers_original)
        TY, (B_reg, t_reg) = reg.register()
        delta = (self.centers_original_ref - TY)
        delta_rmse = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
        reorderIndex = []
        if delta_rmse.all() < 1:
            reorderIndex = np.arange(len(self.centers_original_ref))
        else:
            for i in range(len(self.centers_original_ref)):
                delta_temp = TY - self.centers_original_ref[i, :]
                delta_temp_rmse = delta_temp[:, 0] * delta_temp[:, 0] + delta_temp[:, 1] * delta_temp[:, 1]
                I = np.argmin(delta_temp_rmse)
                reorderIndex.append(I)

        centers = centers_original[reorderIndex, :]

        worldPt_T = np.transpose(np.array(self.fiducialWorldPts))
        imgPt = np.transpose(np.array(centers))
        radialDistortion = np.array([0.0, 0.0])
        tangentialDistortion = np.array([0.0, 0.0])
        worldPts2 = np.array(self.fiducialWorldPts[:, 0:2], dtype=float)
        cameraParams = self.eng.cameraParameters('IntrinsicMatrix', np.transpose(self.intCalMat), 'RadialDistortion', radialDistortion, 'TangentialDistortion', tangentialDistortion, 'WorldPoints', worldPts2);

        # MATLAB function calls
        [rotationMatrix, translationVector] = self.eng.extrinsics(np.array(centers, dtype=float), worldPts2, cameraParams, nargout=2)
        [orientation, location] = self.eng.extrinsicsToCameraPose(rotationMatrix, translationVector, nargout=2)

        # Assigns MATLAB results to VTK transforms
        xfrmMat = np.empty((4, 4))
        xfrmMat[0:3, 0:3] = orientation
        xfrmMat[0:3, 3] = location

        self.fluoroCTRegTransform.SetMatrix(np.reshape(xfrmMat, 16))
        self.fluoroTransform.Update()

        self.ToolTxLCD.display(xfrmMat[0, 3])
        self.ToolTyLCD.display(xfrmMat[1, 3])
        self.ToolTzLCD.display(xfrmMat[2, 3])
        return xfrmMat
    
    def updateScene_streamingtest(self):
        # Just tests streaming and artifact removal (no C-arm pose calculation)
        if self.isStreaming:
            _, img_pre = self.fluoroStream.read()
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
            self.fluoroFrameCtr += 1
            qimg_pre = self.cv2ToQPixmap(img_pre, QtGui.QImage.Format.Format_RGB888)
            self.preImageLabel.setPixmap(QtGui.QPixmap.fromImage(qimg_pre))
            cv2.imwrite(f"{self.outdir}/preprocessed/{self.fluoroFrameCtr}.png", img_pre)

            img_fiducials, img_removed = self.eval.run_eval(img_pre, self.outdir, self.fluoroFrameCtr)
            cv2.imshow("removed rgb", img_removed)
            cv2.waitKey(0)
            img_removed = cv2.cvtColor(img_removed, cv2.COLOR_RGB2BGR)
            img_fiducials = cv2.cvtColor(img_fiducials, cv2.COLOR_RGB2BGR)
            
            img_removed = cv2.cvtColor(img_removed, cv2.COLOR_BGR2RGB)
            img_fiducials = cv2.cvtColor(img_fiducials, cv2.COLOR_BGR2RGB)
            qimg_fiducials = self.cv2ToQPixmap(img_fiducials, QtGui.QImage.Format.Format_RGB888)
            qimg_removed = self.cv2ToQPixmap(img_removed, QtGui.QImage.Format.Format_RGB888)
            self.removedImageLabel.setPixmap(QtGui.QPixmap.fromImage(qimg_removed))
            self.fiducialsImageLabel.setPixmap(QtGui.QPixmap.fromImage(qimg_fiducials))
        self.ren.ResetCameraClippingRange()
        self.qvtkwin.GetRenderWindow().Render()
    
    def updateScene(self):
        if self.isTrackerInitialized:
            port_handles, time_stamps, frame_numbers, tracking, tracking_quality = self.tracker.get_frame()
            self.trackedTipTransform.SetMatrix(np.reshape(tracking[0], 16))
            self.trackedTipTransform.Update()
            self.ToolTxLCD.display(tracking[0][0, 3])
            self.ToolTyLCD.display(tracking[0][1, 3])
            self.ToolTzLCD.display(tracking[0][2, 3])
        # Full workflow (streaming, artifact removal, pose calculation)
        if self.isStreaming:
            # Read frame
            _, img_pre = self.fluoroStream.read()
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
            self.fluoroFrameCtr += 1

            # Show frame in Qt
            qimg_pre = self.cv2ToQPixmap(img_pre, QtGui.QImage.Format.Format_RGB888)
            self.preImageLabel.setPixmap(QtGui.QPixmap.fromImage(qimg_pre))

            # Save frame
            cv2.imwrite(f"{self.outdir}/preprocessed/{self.fluoroFrameCtr}.png", img_pre)

            # Artifact removal
            img_fiducials, img_removed = self.eval.run_eval(img_pre, self.outdir, self.fluoroFrameCtr)
            img_removed = cv2.cvtColor(img_removed, cv2.COLOR_RGB2BGR)
            img_fiducials = cv2.cvtColor(img_fiducials, cv2.COLOR_RGB2BGR)

            # Show separated images in Qt
            qimg_fiducials = self.cv2ToQPixmap(img_fiducials, QtGui.QImage.Format.Format_RGB888)
            qimg_removed = self.cv2ToQPixmap(img_removed, QtGui.QImage.Format.Format_RGB888)
            self.removedImageLabel.setPixmap(QtGui.QPixmap.fromImage(qimg_removed))
            self.fiducialsImageLabel.setPixmap(QtGui.QPixmap.fromImage(qimg_fiducials))

            # Calculate C-arm pose
            self.getCarmPose(img_fiducials)
            print(self.fluoroCubeActor.GetUserMatrix())
        self.ren.ResetCameraClippingRange()
        self.qvtkwin.GetRenderWindow().Render()
        
    def cv2ToQPixmap(self, cv2img, format):
        height, width, bytesPerComponent = cv2img.shape
        bytesPerLine = bytesPerComponent * width
        return QtGui.QImage(cv2img.data, width, height, bytesPerLine, format)
    def readFiducialFile(self):
        fname = self.fiducialFileField.text()
    
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_arr = line.split(',')
                self.fiducialWorldPts.append([float(line_arr[1].strip()), float(line_arr[2].strip()), float(line_arr[3].strip())])
        self.fiducialWorldPts = np.array(self.fiducialWorldPts, dtype=np.float32)
    
    def runCarmCal(self):

        fname1 = self.carmCalDicom1Field.text()
        fname2 = self.carmCalDicom2Field.text()
        fiducial_fname = ct.extractFiducials(fname1, fname2)
        worldpts_fname = self.fiducialFileField.text()
        print(self.intCalMat)
        ct.computeCarmPose(self.intCalMat, fiducial_fname, worldpts_fname)
        
    def handleStreamToggle(self):
        if self.streamToggle.isChecked():
            self.isStreaming = True
            self.fluoroStream = cv2.VideoCapture(0)
            self.fluoroCubeActor.SetUserTransform(self.fluoroCTRegTransform)
            self.ren.AddActor(self.fluoroCubeActor)
            #self.updateTimer.timeout.connect(self.updateScene)

        else:
            self.isStreaming = False
            self.fluoroStream = None

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DeepDecompositionViewer()
    window.show()
    window.iren.Initialize()
    sys.exit(app.exec())