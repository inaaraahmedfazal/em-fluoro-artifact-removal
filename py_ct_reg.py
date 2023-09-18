import pydicom as dicom
import numpy as np
import cv2
from skimage import measure
from pycpd import AffineRegistration
import matplotlib.pyplot as plt
import os
import HandEyeCalLogic as he


THRESHOLD_VALUE_1 = 0.3
THRESHOLD_VALUE_2 = 0.25
REORDER_INDEX = [7, 2, 10, 6, 0, 3, 8, 11, 5, 1, 9, 4]

def getROIImage(frameIndex: int, info: dicom.FileDataset, indexRef = -1):
    images = info.pixel_array[frameIndex]
    images_float = images/float(np.amax(images))
    [roiRow, roiCol] = np.where(images_float > 0.001)

    if indexRef >= 78:
        rowMax = 1024
        rowMin = 0
        colMax = 1024 - 300
        colMin = 0
    elif indexRef >= 64:
        rowMax = 1024
        rowMin = 0
        colMax = 1024 - 150
        colMin = 0
    else:
        rowMax = 1024
        rowMin = 0
        colMax = 1024
        colMin = 0

    return images_float[rowMin:rowMax, colMin:colMax], images_float.astype(np.float32)

def extractFiducials(fname1: str, fname2: str):
    # Step 1: read dicom data and extract fiducials for calibration
    info = dicom.dcmread(fname1)
    angleIncrement = info.PositionerSecondaryAngleIncrement
    numFrames = info.NumberOfFrames

    uniqueAngles = np.unique(angleIncrement)
    numUniqueAngles = len(uniqueAngles)

    indexRef = int(np.ceil(numUniqueAngles/2.0)) - 1
    framesIndex = np.where(angleIncrement == uniqueAngles[indexRef])
    frameIndex = int(np.ceil(np.median(framesIndex)))
    ROIImage, images_float = getROIImage(frameIndex, info)

    # cv2.imshow("ROI Image", ROIImage)
    # cv2.waitKey(0)

    # Threshold Segmentation
    bw = ROIImage < THRESHOLD_VALUE_1
    bwint = bw.astype(np.float32)
    # cv2.imshow("bw2", bwint)
    # cv2.waitKey(0)
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
    center = np.array([[C_i.centroid[1], C_i.centroid[0]] for C_i in C])
    numberBlobs = len(center)

    centers_original_ref = center

    textFontSize = 0.5
    labelFontSize = 1.5
    labelShiftY = -30
    textShiftX = -80
    textShiftY = 30

    images_colour = cv2.cvtColor(images_float, cv2.COLOR_GRAY2RGB)

    for k in range(numberBlobs):
        x = round(centers_original_ref[k, 0])
        y = round(centers_original_ref[k, 1])
        x0 = centers_original_ref[k, 0]
        y0 = centers_original_ref[k, 1]

        cv2.circle(images_colour, (x, y), 1, (0, 255, 255), -1)
        textPos = (x + textShiftX, y + textShiftY)
        cv2.putText(images_colour, f"[{x0:.2f}, {y0:.2f}]", textPos, cv2.FONT_HERSHEY_SIMPLEX, textFontSize, (0, 0, 0), 1)
        labelPos = (x, y + labelShiftY)
        cv2.putText(images_colour, str(k + 1), labelPos, cv2.FONT_HERSHEY_SIMPLEX, labelFontSize, (0, 255, 0), 2)
    # cv2.imshow("segmented", images_colour)
    # cv2.waitKey(0)

    # Collecting fiducial set # 1

    fiducialDataset_1 = np.zeros((12, 2, numUniqueAngles))

    for indexRef in range(numUniqueAngles-20):
        framesIndex = np.where(angleIncrement == uniqueAngles[indexRef])
        frameIndex = int(np.ceil(np.median(framesIndex)))

        ROIImage, images_float = getROIImage(frameIndex, info)
        bw = ROIImage < THRESHOLD_VALUE_2

        L = measure.label(bw, connectivity=2)
        S = measure.regionprops(L)
        S_area = [S_i.area for S_i in S]
        S_area_arr = np.array(S_area)
        S_area_idx = np.where((S_area_arr >= 200) & (S_area_arr <= 2000))
        S_area_idx += np.ones(len(S_area_idx))
        
        bw2 = np.isin(L, S_area_idx)
        L2 = measure.label(bw2, connectivity=2)
        C = measure.regionprops(L2)

        centers_original = np.array([[C_i.centroid[1], C_i.centroid[0]] for C_i in C])
        numberBlobs = len(center)

        reg = AffineRegistration(X=centers_original_ref, Y=centers_original)
        TY, (B_reg, t_reg) = reg.register()
        delta = (centers_original_ref - TY)
        delta_rmse = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
        reorderIndex = []
        if delta_rmse.all() < 1:
            reorderIndex = np.arange(len(centers_original_ref))
        else:
            for i in range(len(centers_original_ref)):
                delta_temp = TY - centers_original_ref[i, :]
                delta_temp_rmse = delta_temp[:, 0] * delta_temp[:, 0] + delta_temp[:, 1] * delta_temp[:, 1]
                I = np.argmin(delta_temp_rmse)
                reorderIndex.append(I)

        center_original_ordered = centers_original[reorderIndex, :]
        img_temp = cv2.cvtColor(images_float.astype(np.float32), cv2.COLOR_GRAY2RGB)
        for k in range(numberBlobs):
            x = round(center_original_ordered[k, 0])
            y = round(center_original_ordered[k, 1])
            x0 = center_original_ordered[k, 0]
            y0 = center_original_ordered[k, 1]

            cv2.circle(img_temp, (x, y), 1, (0, 255, 255), -1)
            textPos = (x + textShiftX, y + textShiftY)
            cv2.putText(img_temp, f"[{x0:.2f}, {y0:.2f}]", textPos, cv2.FONT_HERSHEY_SIMPLEX, textFontSize, (0, 0, 0), 1)
            labelPos = (x, y + labelShiftY)
            cv2.putText(img_temp, str(k + 1), labelPos, cv2.FONT_HERSHEY_SIMPLEX, labelFontSize, (0, 255, 0), 2)

        # cv2.imshow(f"frame {indexRef}", img_temp)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        fiducialDataset_1[:, :, indexRef] = center_original_ordered

    # Dataset 2
    info = dicom.dcmread(fname2)

    angleIncrement = info.PositionerPrimaryAngleIncrement
    numFrames = info.NumberOfFrames
    uniqueAngles = np.unique(angleIncrement)

    numUniqueAngles = len(uniqueAngles)
    fiducialDataset_2 = np.empty((12, 2, numUniqueAngles))

    indexRef = int(np.ceil(numUniqueAngles/2)) - 1
    framesIndex = np.where(angleIncrement == uniqueAngles[indexRef])
    frameIndex = int(np.ceil(np.median(framesIndex)))
    ROIImage, images_float = getROIImage(frameIndex, info)

    bw = ROIImage < THRESHOLD_VALUE_1
    L = measure.label(bw, connectivity=2)
    S = measure.regionprops(L)
    S_area = [S_i.area for S_i in S]
    S_area_arr = np.array(S_area)
    S_area_idx = np.where((S_area_arr >= 200) & (S_area_arr <= 2000))
    S_area_idx += np.ones(len(S_area_idx))

    bw2 = np.isin(L, S_area_idx)
    L2 = measure.label(bw2, connectivity=2)
    C = measure.regionprops(L2)

    center = np.array([[C_i.centroid[1], C_i.centroid[0]] for C_i in C])
    numberBlobs = len(center)

    centers_original_ref_2 = center

    textFontSize = 0.5
    labelFontSize = 1.5
    labelShiftY = -30
    textShiftX = -80
    textShiftY = 30

    reg = AffineRegistration(X=centers_original_ref, Y = centers_original_ref_2)
    TY, (B_reg, t_reg) = reg.register()
    B_reg = np.transpose(B_reg)
    fig_before = plt.figure()
    ax1 = fig_before.add_subplot()
    ax1.set_title("Before")
    ax1.scatter(centers_original_ref[:, 0], centers_original_ref[:, 1], s=10, c='b', marker='s', label='centers original ref')
    ax1.scatter(centers_original_ref_2[:, 0], centers_original_ref_2[:, 1], s=10, c='r', marker='o', label="center original ref 2")

    fig_after = plt.figure()
    ax2 = fig_after.add_subplot()
    ax2.set_title("After")
    ax2.scatter(centers_original_ref[:, 0], centers_original_ref[:, 1], s=10, c='b', marker='s', label="centers original ref")
    ax2.scatter(TY[:, 0], TY[:, 1], s=10, c='r', marker='o', label='TY')

    #plt.show()
    n = len(centers_original_ref_2)
    delta = centers_original_ref - TY[0:n, :]
    delta_rmse = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]

    reorderIndex = []
    center_original_ordered = []

    if delta_rmse.all() < 1:
        reorderIndex = np.arange(n)
        center_original_ordered = centers_original_ref_2

    else:
        for i in range(n):
            delta_temp = centers_original_ref[i, :] - TY
            delta_temp_rmse = delta_temp[:, 0] * delta_temp[:, 0] + delta_temp[:, 1] * delta_temp[:, 1]
            I = np.argmin(delta_temp_rmse)
            center_original_ordered.append(np.reshape(np.linalg.inv(B_reg) @ np.reshape(TY[I, :]-t_reg, (2, 1)), (2,)))
    centers_original_ref_2 = center_original_ordered

    images_colour = cv2.cvtColor(images_float, cv2.COLOR_GRAY2RGB)
    for k in range(numberBlobs):
        temp = centers_original_ref_2[k][0]
        x = round(centers_original_ref_2[k][0])
        y = round(centers_original_ref_2[k][1])
        x0 = centers_original_ref_2[k][0]
        y0 = centers_original_ref_2[k][1]

        cv2.circle(images_colour, (x, y), 1, (0, 255, 255), -1)
        textPos = (x + textShiftX, y + textShiftY)
        cv2.putText(images_colour, f"[{x0:.2f}, {y0:.2f}]", textPos, cv2.FONT_HERSHEY_SIMPLEX, textFontSize, (0, 0, 0), 1)
        labelPos = (x, y + labelShiftY)
        cv2.putText(images_colour, str(k + 1), labelPos, cv2.FONT_HERSHEY_SIMPLEX, labelFontSize, (0, 255, 0), 2)
    # cv2.imshow("segmented", images_colour)
    # cv2.waitKey(0)
    #cv2.destroyAllWindows()

    for indexRef in range(3, numUniqueAngles):
        framesIndex = np.where(angleIncrement == uniqueAngles[indexRef])
        frameIndex = int(np.ceil(np.median(framesIndex)))
        ROIImage, images_float = getROIImage(frameIndex, info, indexRef)

        bw = ROIImage < THRESHOLD_VALUE_2
        # cv2.imshow("bw", bw.astype(np.float32))
        # cv2.waitKey(0)
        L = measure.label(bw, connectivity=2)
        S = measure.regionprops(L)
        S_area = [S_i.area for S_i in S]
        S_area_arr = np.array(S_area)
        S_area_idx = np.where((S_area_arr >= 200) & (S_area_arr <= 1500))
        S_area_idx += np.ones(len(S_area_idx))

        bw2 = np.isin(L, S_area_idx)
        L2 = measure.label(bw2, connectivity=2)
        C = measure.regionprops(L2)

        center = np.array([[C_i.centroid[1], C_i.centroid[0]] for C_i in C])
        img_temp = cv2.cvtColor(images_float, cv2.COLOR_GRAY2RGB)
        for k in range(numberBlobs):
            x = round(center[k][0])
            y = round(center[k][1])
            cv2.circle(img_temp, (x, y), 1, (0, 255, 255), -1)

        # cv2.imshow(f"index {indexRef}", img_temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # blurred = cv2.medianBlur(bw, 25)
        # blurred = cv2.blur(blurred, (10, 10))
        # bw2flt = bw2.astype(np.uint8)
        # th, bw2int = cv2.threshold(bw2flt, 0, 255, cv2.THRESH_BINARY)
        # blurred = cv2.medianBlur(bw2int, 25)
        # blurred = cv2.blur(blurred, (10, 10))
        # cv2.imshow("blurred", blurred)
        # cv2.waitKey(0)

        # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 25, param1=50, param2=30, minRadius=10, maxRadius=15)

        # centers_original = np.array([[circle[0], circle[1]] for circle in circles[0, :]])

        centers_original = center
        numberBlobs = len(center)
        centers_original_ref_2 = np.array(centers_original_ref_2)
        reg = AffineRegistration(X=centers_original_ref_2, Y = centers_original)
        TY, (B_reg, t_reg) = reg.register()
        B_reg = np.transpose(B_reg)
        delta = centers_original_ref_2 - TY[:n, :]
        delta_rmse = delta[:, 0] * delta[:, 0] + delta [:, 1] * delta[:, 1]

        reorderIndex = []
        center_original_ordered = []

        if delta_rmse.all() < 1:
            reorderIndex = np.arange(n)
            center_original_ordered = centers_original
        else:
            for i in range(n):
                delta_temp = centers_original_ref_2[i, :] - TY
                delta_temp_rmse = delta_temp[:, 0] * delta_temp[:, 0] + delta_temp[:, 1] * delta_temp[:, 1]
                I = np.argmin(delta_temp_rmse)
                center_original_ordered.append(np.reshape(np.linalg.inv(B_reg) @ np.reshape(TY[I, :]-t_reg, (2, 1)), (2,)))
        
        img_temp = cv2.cvtColor(images_float, cv2.COLOR_GRAY2RGB)
        for k in range(numberBlobs):
            x = round(center_original_ordered[k][0])
            y = round(center_original_ordered[k][1])
            x0 = center_original_ordered[k][0]
            y0 = center_original_ordered[k][1]

            cv2.circle(img_temp, (x, y), 1, (0, 255, 255), -1)
            textPos = (x + textShiftX, y + textShiftY)
            cv2.putText(img_temp, f"[{x0:.2f}, {y0:.2f}]", textPos, cv2.FONT_HERSHEY_SIMPLEX, textFontSize, (0, 0, 0), 1)
            labelPos = (x, y + labelShiftY)
            cv2.putText(img_temp, str(k + 1), labelPos, cv2.FONT_HERSHEY_SIMPLEX, labelFontSize, (0, 255, 0), 2)
        # cv2.imshow(f"index {indexRef}", img_temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        fiducialDataset_2[:, :, indexRef] = center_original_ordered

    # Combine fiducials

    fiducialDataset_total = np.zeros((12, 2, 76 + 90 - 3))
    fiducialDataset_total[:, :, :76] = fiducialDataset_1
    fiducialDataset_total[:, :, 76:] = fiducialDataset_2[:, :, 3:]

    #print(fiducialDataset_total)

    # Save out fiducials
    fiducial_fname = f"{os.getcwd()}/fiducialDataset_total.csv"
    with open(fiducial_fname, 'w') as f:
        f.write("frame,x,y\n")
        for frameIdx in range(np.size(fiducialDataset_total, 2)):
            x_str = ' '.join([str(xi) for xi in fiducialDataset_total[:, 0, frameIdx]])
            y_str = ' '.join([str(yi) for yi in fiducialDataset_total[:, 1, frameIdx]])
            f.write(f"{frameIdx},{x_str},{y_str}\n")
    return fiducial_fname
# Step 2: C-arm calibration (using OpenCV method instead)
def calibrateCarm(pickedMarkers_fname, fiducial_fname):
    pickedMarkers = []
    
    with open(pickedMarkers_fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_arr = line.split(',')
            pickedMarkers.append([float(line_arr[1].strip()), float(line_arr[2].strip()), float(line_arr[3].strip())])
    pickedMarkers = np.array(pickedMarkers, dtype=np.float32)
    worldPoints = []
    fiducialDataset_total = []
    imagePoints = []
    numValidFrames = 0
    with open(fiducial_fname, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line_arr = line.split(',')
            x_ls = []
            for x_val in line_arr[1].split(' '):
                x_ls.append(float(x_val))
            y_ls = []
            for y_val in line_arr[2].split(' '):
                y_ls.append(float(y_val))

            pt_arr = np.empty((len(x_ls), 1, 2), dtype=np.float32)
            pt_arr[:, 0, 0] = np.array(x_ls, dtype=np.float32)
            pt_arr[:, 0, 1] = np.array(y_ls, dtype=np.float32)

            fiducialDataset_total.append(pt_arr)
            if not(pt_arr.all() == 0):
                imagePoints.append(pt_arr)
                worldPoints.append(pickedMarkers)
                numValidFrames += 1
    #fiducialDataset_total = np.array(fiducialDataset_total)

    numImages = np.size(fiducialDataset_total, 0)
    
    imageSize = (1024, 1024)
    ret, intMat, dist, rvecs, tvecs = cv2.calibrateCamera(worldPoints, imagePoints, imageSize, None, None, flags=cv2.CALIB_USE_LU)
    print(intMat)
    return intMat, dist, fiducialDataset_total, pickedMarkers, imagePoints
def rotm2axang(R):
    theta = np.real(np.arccos(np.complex128(0.5*(R[0, 0] + R[1, 1] + R[2, 2] - 1))))
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(theta))

    # degenerate cases where theta is divisible by pi or the axis consists of all zeros
    # singularLogical = theta % np.pi == 0 | v.all() == 0

    return np.append(v, theta)
def computeCarmPose(intMat, fiducial_fname, pickedMarkers_fname):
    pickedMarkers = []
    with open(pickedMarkers_fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_arr = line.split(',')
            pickedMarkers.append([float(line_arr[1].strip()), float(line_arr[2].strip()), float(line_arr[3].strip())])
            #pickedMarkers.append([float(line_arr[1].strip()), float(line_arr[2].strip()), 0.0])
    pickedMarkers = np.array(pickedMarkers, dtype=np.float32)

    # imagePts has the 0.0 frames filtered out
    worldPts = []
    fiducialDataset_total = []
    imagePts = []
    numValidFrames = 0
    with open(fiducial_fname, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line_arr = line.split(',')
            x_ls = []
            for x_val in line_arr[1].split(' '):
                x_ls.append(float(x_val))
            y_ls = []
            for y_val in line_arr[2].split(' '):
                y_ls.append(float(y_val))

            pt_arr = np.empty((len(x_ls), 1, 2), dtype=np.float32)
            pt_arr[:, 0, 0] = np.array(x_ls, dtype=np.float32)
            pt_arr[:, 0, 1] = np.array(y_ls, dtype=np.float32)

            fiducialDataset_total.append(pt_arr)
            if not(pt_arr.all() == 0):
                imagePts.append(pt_arr)
                worldPts.append(pickedMarkers)
                numValidFrames += 1

    numImages = len(imagePts)
    camLocationDataSet = np.zeros((3, numImages))
    camOrientationDataSet = np.zeros((3, 3, numImages))
    translationVectorTotal = np.zeros((3, numImages))
    rotationMatrixTotal = np.zeros((3, 3, numImages))
    for i in range(numImages):
        print(i)
        imgPt = np.reshape(imagePts[i], (12, 2))
        imgPt = np.transpose(imgPt)
        worldPt = worldPts[i]
        worldPt_T = np.transpose(worldPt)
        [rotationMatrix, translationVector] = he.hand_eye_p2l(worldPt_T, imgPt, intMat)
        #M_int_est, M_ext_est = he.cameraCombinedCalibration2(imgPt, worldPt_T)
        camOrientation = np.transpose(rotationMatrix)
        #camOrientation = rotationMatrix
        camLocation = -np.transpose(translationVector)@np.transpose(rotationMatrix)
        #camLocation = -np.transpose(translationVector) @ rotationMatrix
        translationVectorTotal[:, i] = translationVector[:, 0]
        rotationMatrixTotal[:, :, i] = rotationMatrix
        camLocationDataSet[:, i] = camLocation
        camOrientationDataSet[:, :, i] = camOrientation

    numDataset1 = 76
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(camLocationDataSet[0, :], camLocationDataSet[1, :], camLocationDataSet[2, :], s=10, c='b', marker='s')
    plt.show()

def evaluateCarmCalibration(numImages, imagePts, worldPts, intMat):
    # Step 5.1: Evaluate the pose estimation
    # Give range of pose estimation because intrinsics are not stable from calculation

    # Determine delta range for each intrinsic parameter
    deltaFx = 10
    deltaFy = 10
    deltaTx = 4
    deltaTy = 4

    radialDistortion = [0, 0]
    tangentialDistortiion = [0, 0]
    imageIndex = 0

    intMatTotal = np.zeros((3, 3, 9*9*5*5))
    numTotal = 0

    for i in np.arange(-deltaFx, deltaFx + 0.01, 2.5):
        for j in np.arange(-deltaFy, deltaFy + 0.01, 2.5):
            for k in np.arange(-deltaTx, deltaTx + 0.01, 2):
                for m in np.arange(-deltaTy, deltaTy + 0.01, 2):
                    intMat_temp = np.transpose(np.array([[3619.966478836598 + i, 0, 493.903840202292 + k],
                                                         [0, 3626.852406736492 + j, 527.8298899002965 + m],
                                                         [0, 0, 1]]))
                    intMatTotal[:, :, numTotal] = intMat_temp
                    numTotal += 1

    rotationAngleDeltaTotal = np.zeros((numImages, np.size(intMatTotal, 2)))
    translationDeltaTotal = np.zeros((numImages, np.size(intMatTotal, 2), 3))
    
    for imgIndex in range(numImages):
        print("imgIndex:", imgIndex)
        imgPt = np.reshape(imagePts[imgIndex], (12, 2))
        imgPt = np.transpose(imgPt)
        worldPts_T = np.transpose(worldPts)
        [rotationMatrixRef, translationVectorRef] = he.hand_eye_p2l(worldPts_T, imgPt, intMat)
        for i in range(np.size(intMatTotal, 2)):
            print("i:", i)
            [rotationMatrix, translationVector] = he.hand_eye_p2l(worldPts_T, imgPt, intMatTotal[:, :, i])
            relativeMatrix = rotationMatrixRef @ np.linalg.inv(rotationMatrix)
            axang = rotm2axang(relativeMatrix)
            absAxang = np.abs(axang[0:3])
            I = np.argmax(absAxang)
            rotationAngleDeltaTotal[imgIndex, i] = np.sign(axang[I]) * axang[3] * 180/np.pi
            translationDelta_temp = translationVector - translationVectorRef
            translationDeltaTotal[imgIndex, i, :] = np.transpose(translationDelta_temp)

    rotationAngleDeltaTotal1 = np.zeros((numImages, np.size(intMatTotal, 2)))
    translationDeltaTotal2 = np.zeros((numImages, np.size(intMatTotal, 2), 3))
    for imgIndex in range(numImages):
        imgPt = np.reshape(imagePts[imgIndex], (12, 2))
        imgPt = np.transpose(imgPt)
        worldPts_T = np.transpose(worldPts)
        [rotationMatrixRef, translationVectorRef] = he.hand_eye_p2l(worldPts_T, imgPt, intMat)
        for i in range(np.size(intMatTotal, 2)):
            [rotationMatrix, translationVector] = he.hand_eye_p2l(worldPts_T, imgPt, intMatTotal[:, :, i])
            relativeMatrix = np.transpose(rotationMatrixRef) @ np.linalg.inv(np.transpose(rotationMatrix))
            axang = rotm2axang(relativeMatrix)
            absAxang = np.abs(axang[0:3])
            I = np.argmax(absAxang)
            rotationAngleDeltaTotal1[imgIndex, i] = np.sign(axang[I]) * axang[3] * 180/np.pi
            translationDelta_temp = translationVector - translationVectorRef
            translationDeltaTotal2[imgIndex, i, :] = np.transpose(translationDelta_temp)

    # Translation deviation
    imgIndex = 36
    deltaX = np.mean(translationDeltaTotal2[imgIndex, :, 0])
    stdX = np.std(translationDeltaTotal2[:, :, 0], 1)

    deltaY = np.mean(translationDeltaTotal2[imgIndex, :, 1])
    stdY = np.std(translationDeltaTotal2[:, :, 1], 1)

    deltaZ = np.mean(translationDeltaTotal2[imgIndex, :, 2])
    stdZ = np.std(translationDeltaTotal2[:, :, 2], 1)

    # Angle deviation
    deltaAngle = np.mean(rotationAngleDeltaTotal1[imgIndex, :])
    stdAngle = np.std(rotationAngleDeltaTotal1, 1)

    plt.figure()
    x_stdX = np.arange(0, len(stdX))
    plt.bar(x_stdX, stdX)
    plt.xlabel("Angle readings from angio CT")
    plt.ylabel("STD at each angle")
    plt.title("Translation (X) deviation")

    plt.figure()
    x_stdY = np.arange(0, len(stdY))
    plt.bar(x_stdY, stdY)
    plt.xlabel("Angle readings from angio CT")
    plt.ylabel("STD at each angle")
    plt.title("Translation (Y) deviation")

    plt.figure()
    x_stdZ = np.arange(0, len(stdZ))
    plt.bar(x_stdZ, stdZ)
    plt.xlabel("Angle readings from angio CT")
    plt.ylabel("STD at each angle")
    plt.title("Translation (Z) deviation")
    plt.show()
    # Step 5.2: Compute reprojection error

    # Load in dicom files
    fname1 = "C:/Users/iahmedf/Documents/ShuweiRegistration/XA-3/XA000000.dcm"
    info1 = dicom.dcmread(fname1)
    angleIncrement1 = info1.PositionerSecondaryAngleIncrement
    numFrames1 = info1.NumberOfFrames
    uniqueAngles1 = np.unique(angleIncrement1)
    numUniqueAngles1 = len(uniqueAngles1)

    fname2 = "C:/Users/iahmedf/Documents/ShuweiRegistration/XA-9/XA000000.dcm"
    info2 = dicom.dcmread(fname2)
    angleIncrement2 = info2.PositionerPrimaryAngleIncrement
    numFrames2 = info2.NumberOfFrames
    uniqueAngles2 = np.unique(angleIncrement2)
def main():
    fiducial_fname = extractFiducials("C:/Users/iahmedf/Documents/ShuweiRegistration/XA-3/XA000000.dcm", "C:/Users/iahmedf/Documents/ShuweiRegistration/XA-9/XA000000.dcm")
    intMat, dist, fiducialDataset_total, worldPts, imagePts = calibrateCarm(fiducial_fname)
    # or if that calibration doesn't work:
    # intMat = np.transpose(np.array([[3619.966478836598, 0, 493.903840202292], [0, 3626.852406736492, 527.8298899002965], [0, 0, 1]]))
    computeCarmPose(intMat, dist, fiducialDataset_total, worldPts, imagePts)

if __name__ == "__main__":
    main()