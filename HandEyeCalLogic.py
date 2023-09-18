import numpy as np
import numpy.matlib
import cv2
import scipy.linalg as la
from copy import copy

def hand_eye_p2l(X, Q, A, tol=1e-3):
    """
    Arguments:  X (3xn):    3D coordinates, tracker space
                Q (2xn):    2D pixel locations, image space
                A (3x3):    camera matrix

    Returns:    R (3x3):    orthonormal rotation matrix
                t (3x1):    translation
    """
    n = Q.shape[1]
    e = np.ones(n)
    J = np.identity(n) - (np.divide((np.transpose(e) * e), n)) # why multiply two 1D arrays of 1s
    Q = np.linalg.inv(A) @ np.vstack((Q, e))
    Y = ([[], [], []])

    # Normalizing the 2D pixel coordinates?
    for i in range(n):
        x = Q[:, i]
        y = np.linalg.norm(x)
        z = x / y
        z = np.reshape(z, (3, 1))
        Y = np.hstack((Y, z))

    Q = Y
    err = np.inf
    E_old = 1000 * np.ones((3, n))

    while err > tol:
        a = Y @ J @ X.T.conj()
        U, S, V, = np.linalg.svd(a)

        # Get rotation
        R = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U @ V)]]) @ V 

        # Get translation
        T = Y - R @ X
        t = ([])
        for i in range(np.shape(Y)[0]): # could use n?
            t = np.append(t, np.mean(T[i]))
        t = np.reshape(t, (np.shape(Y)[0], 1))

        # Reprojection
        h = R @ X + t * e
        H = ([])
        for i in range(np.shape(Q)[1]):
            H = np.append(H, np.dot(h[:, i], Q[:, i]))
        Y = np.matlib.repmat(H, 3, 1) * Q

        # Get reprojection error
        E = Y - R @ X - t * e
        err = np.linalg.norm(E - E_old, 'fro')
        E_old = E
    
    return R, t
def cameraCombinedCalibration2(P_2D, P_3D):
    """
     Performed a combine camera calibration (both the intrinsic and extrinsic
     (i.e. hand-eye) based on corresponding 2D pixels and 3D points.

     Inputs:  P_2D - 2xN pixel coordinates
              P_3D - 3xN 3D coordinates, it is assumed that each 3D points is
              projected to the image plane, thus there is a known
              correspondence between P_2D and P_3D

     Outputs: M_int_est, estimated camera intrinsic parameters:
                       = [ fx 0 cx
                           0 fy cy
                           0  0  1 ]
              M_ext_est, estimate camera extrinsic parameters (i.e. hand-eye)
                       = [ R_3x3, t_3x1
                               0  1 ]
    """

    # space allocation for outposts
    M_int_est = np.identity(3)
    M_ext_est = np.identity(4)

    # size of the input
    N = P_2D.shape[1]
    if P_2D.shape[1] == P_3D.shape[1]:

      # construct the system of linear equations
      A = np.empty((0, 12))
      for i in range(N):
        a = np.array([[P_3D[0, i], P_3D[1, i], P_3D[2, i], 1, 0, 0, 0, 0, -P_2D[0, i] * P_3D[0, i],
                       -P_2D[0, i] * P_3D[1, i], -P_2D[0, i] * P_3D[2, i], -P_2D[0, i]]])
        b = np.array([[0, 0, 0, 0, P_3D[0, i], P_3D[1, i], P_3D[2, i], 1, -P_2D[1, i] * P_3D[0, i],
                       -P_2D[1, i] * P_3D[1, i], -P_2D[1, i] * P_3D[2, i], -P_2D[1, i]]])

        c = np.vstack((a, b))
        A = np.vstack((A, c))

      # The answer is the eigen vector corresponding to the single zero eivenvalue of the matrix (A' * A)
      D, V = la.eig(A.conj().T @ A)
      min_idx = np.where(D == min(D))[0][0]
      m = V[:, min_idx]

      # note that m is arranged as:
      # m = [ m11 m12 m13 m14 m21 m22 m23 m24 m31 m32 m33 m34]
      # rearrange to form:
      # m = [ m11 m12 m13 m14 ;
      #       m21 m22 m23 m24 ;
      #       m31 m32 m33 m34 ];

      m = np.reshape(m, (3, 4))

      # m is known as the projection matrix, basically it is the intrinsic matrix multiplied with the extrinsic (hand-eye calibration) matrix
      # The first step to resolve intrinsic/extrinsic matrices from m is to find the scaling factor. Note that the last row [m31 m32 m33] is the last row of the rotation matrix R, thus one can find the scale there
      gamma = np.absolute(np.linalg.norm(m[2, 0:3]))

      # determining the translation in Z and the sign of sigma
      gamma_sign = np.sign(m[2, 3])

      # due to the way we construct our viewing axis, we know that the objects must be IN FRONT of the camera, thus the translation must always be POSITIVE in the Z direction
      M = gamma_sign / gamma * m
      M_proj = M

      # translation in z
      Tz = M[2, 3]

      # third row of the rotation matrix
      M_ext_est[2, :] = M[2, :]

      # principal points
      ox = np.dot(M[0, 0:3], M[2, 0:3])
      oy = np.dot(M[1, 0:3], M[2, 0:3])

      # focal points
      fx = np.sqrt(np.dot(M[0, 0:3], M[0, 0:3]) - ox * ox)
      fy = np.sqrt(np.dot(M[1, 0:3], M[1, 0:3]) - oy * oy)

      # construct the output
      M_int_est[0, 2] = ox
      M_int_est[1, 2] = oy
      M_int_est[0, 0] = fx
      M_int_est[1, 1] = fy

      # 1st row of the rotation matrix
      M_ext_est[0, 0:3] = gamma_sign / fx * (ox * M[2, 0:3] - M[0, 0:3])

      # 2nd row of the rotation matrix
      M_ext_est[1, 0:3] = gamma_sign / fy * (oy * M[2, 0:3] - M[1, 0:3])

      # translation in x
      M_ext_est[0, 3] = gamma_sign / fx * (ox * Tz - M[0, 3])

      # translation in y
      M_ext_est[1, 3] = gamma_sign / fy * (oy * Tz - M[1, 3])

      M_ext_est_neg = copy(M_ext_est)
      M_ext_est_neg[0, :] = -1 * M_ext_est_neg[0, :]
      M_ext_est_neg[1, :] = -1 * M_ext_est_neg[1, :]

      if (np.linalg.norm(M_int_est @ M_ext_est[0:3, :] - M_proj, 'fro')) > (
      np.linalg.norm(M_int_est @ M_ext_est_neg[0:3, :] - M_proj, 'fro')):
        M_ext_est = copy(M_ext_est_neg)

      # given the 3x4 projection matrix M, calculate the projection error between the paired 2D/3D fiducials
      m, n = np.shape(P_3D)

      # calculate the projection of P3D given M
      temp = M @ np.vstack((P_3D, np.ones((1, n))))
      P = np.zeros((2, n))
      P[0, :] = temp[0, :] / temp[2, :]
      P[1, :] = temp[1, :] / temp[2, :]

      # mean projection error, i.e. euclidean distance between P3D after projection from P2D
      p = P - P_2D
      fre = []
      for i in range(n):
        x = np.linalg.norm(p[:, i])
        fre.append(x)

      fre = np.sum(fre) / n

    else:
      print("error: 2D and 3D matricies of different lengths")

    return M_int_est, M_ext_est

def cameraCombinedCalibration(P_2D, P_3D):
    # empty outputs
    M_int_est = np.eye(3)
    M_ext_est = np.eye(4)
    M_proj = np.eye(4)
    
    # input size
    N = np.size(P_2D, 1)

    # system of linear equations
    A = np.zeros((2*N, 12))
    for i in range(N):
        idx = i*2

        A[idx, 0:3] = P_3D[0:3, i]
        A[idx, 3:8] = [1, 0, 0, 0, 0]
        A[idx, 8:11] = -P_2D[0, i] * P_3D[0:3, i]
        A[idx, 11] = -P_2D[0, i]

        A[idx + 1, 0:4] = [0, 0, 0, 0]
        A[idx + 1, 4:7] = P_3D[0:3, i]
        A[idx + 1, 7] = 1
        A[idx + 1, 8:11] = -P_2D[1, i] * P_3D[0:3, i]
        A[idx + 1, 11] = -P_2D[1, i]
        
    print(np.transpose(A) @ A)
    # The answer is the eigenvector corresponding to the single zero eigenvalue of (A' * A)
    D, V = la.eig(A.conj().T @ A)

    # Answer is min column of V (cannot assume sorted)
    min_idx = np.where(D==min(D))[0][0]
    m = V[:, min_idx]
    m = np.reshape(m, (3, 4))

    # Get scaling factor from bottom row of rotation matrix R
    gamma = np.abs(np.linalg.norm(m[2, 0:3]))

    # Determine translation in Z and sign of gamma
    gamma_sign = np.sign(m[2, 3])

    # Objects must be IN FRONT of camera, so translation must by POSITIVE in z-direction
    M = gamma_sign/gamma * m
    M_proj = M

    # Translation in z
    Tz = M[2, 3]

    # 3rd row of rotation matrix
    M_ext_est[2, :] = M[2, :]

    # Principal points
    ox = np.dot(M[0, 0:3], M[2, 0:3])
    oy = np.dot(M[1, 0:3], M[2, 0:3])

    # Focal points
    fx = np.sqrt(np.dot(M[0, 0:3], M[0, 0:3]) - ox * ox)
    fy = np.sqrt(np.dot(M[1, 0:3], M[1, 0:3]) - oy * oy)

    # Construct the output
    M_int_est[0, 2] = ox
    M_int_est[1, 2] = oy
    M_int_est[0, 0] = fx
    M_int_est[1, 1] = fy

    # 1st row of rotation matrix
    M_ext_est[0, 0:3] = gamma_sign/fx*(ox*M[2, 0:3] - M[0, 0:3])

    # 2nd row of rotation matrix
    M_ext_est[1, 0:3] = gamma_sign/fy*(oy*M[2, 0:3] - M[1, 0:3])

    # Translation in x
    M_ext_est[0, 3] = gamma_sign/fx*(ox*Tz-M[0, 3])

    # Translation in y
    M_ext_est[1, 3] = gamma_sign/fy*(oy*Tz-M[1, 3])

    M_ext_est_neg = copy(M_ext_est)
    M_ext_est_neg[0, :] = -1 * M_ext_est_neg[0, :]
    M_ext_est_neg[1, :] = -1 * M_ext_est_neg[1, :]

    if np.linalg.norm(M_int_est @ M_ext_est[0:3, :] - M_proj, 'fro') > np.linalg.norm(M_int_est @ M_ext_est_neg[0:3, :] - M_proj, 'fro'):
        M_ext_est = copy(M_ext_est_neg)

    # Given the 3x4 projection matrix M, calculate projection error between paired 2D/3D fiducials
    m, n = np.shape(P_3D)
    temp = M @ np.vstack((P_3D, np.ones((1, n))))
    P = np.zeros((2, n))
    P[0, :] = temp[0, :] / temp[2, :]
    P[1, :] = temp[1, :] / temp[2, :]

    # Mean projection error: Euclidean distance from P_3D after projection from P_2D
    p = P - P_2D
    fre = []
    for i in range(n):
        x = np.linalg.norm(p[:, i])
        fre.append(x)

    fre = np.sum(fre) / n

    # U, dummy, V = np.linalg.svd(M_ext_est[0:3, 0:3])
    # D = np.eye(3)
    # D[2, 2] = np.linalg.det(U @ np.transpose(V))
    # M_ext_est[0:3, 0:3] = U @ D @ np.transpose(V)
    # M_ext_est[0:3, 0:3] = U @ np.transpose(V)
    return M_int_est, M_ext_est

def analyzeFrames(frames, transforms, intMtx, distCoeffs):
    # Set up lists to save final 3d and 2d data (to be merged into 3xN and 2xN matricies later)
    StylusTipCoordsX = ([])
    StylusTipCoordsY = ([])
    StylusTipCoordsZ = ([])

    CircleCentersX = ([])
    CircleCentersY = ([])

    StylusTipColour = "green"
    # Detect circle center in each frame (2D point)
    numFrames = len(frames)
    for count in range(numFrames):
        img = cv2.imread(frames[count])
        #img = img[0, ::-1, ::-1, :] # may be unnecessary with sksurg preprocessing (compared to slicer)

        # Undistort
        h, w = img.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(intMtx, distCoeffs, (w, h), 1, (w, h))
        img = cv2.undistort(img, intMtx, distCoeffs, None, newCameraMtx)
        # Save raw undistorted image if necessary?

        if StylusTipColour == "green":

            # Colour threshold for green
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (30, 50, 0), (80, 255, 255))
            target = cv2.bitwise_and(img, img, mask=mask)
            # Apply binary mask
            gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

            # Smooth
            blurred = cv2.medianBlur(binary, 25)

        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 25)

        blurred = cv2.blur(blurred, (10, 10))

        # Use Hough to find circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 0.1, 1000, param1=50, param2=30, minRadius=0, maxRadius=50)

        # Draw circle(s) onto image:



        # Save thresholded image (binary or blurred) if necessary?

        # Draw fiducials following spatial tracking for visual validation
        c = transforms[count]
        # x = c[0, 3]
        # y = c[1, 3]
        # z = c[2, 3]
        x = c[0]
        y = c[1]
        z = c[2]
        #do the actual drawing in VTK/GL render window

        # Draw calculated circle
        if circles is None:
            print(f"No circles detected in frame {count}. Try manual circle segmentation")
            def click_event(event, cx, cy, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    cv2.circle(img, (cx, cy), 1, (0, 255, 255), -1)
                    pts.append([cx, cy])

            pts = []

            cv2.namedWindow("Segment Image")

            cv2.setMouseCallback("Segment Image", click_event)

            while True:
                cv2.imshow("Segment Image", img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            cv2.destroyAllWindows()
            circle = cv2.minEnclosingCircle(np.array(pts))
            circle_x_int = np.uint16(np.around(circle[0][0]))
            circle_y_int = np.uint16(np.around(circle[0][1]))
            circle_r_int = np.uint16(np.around(circle[1]))

            cv2.circle(img, (circle_x_int, circle_y_int), 1, (0, 100, 100), 3)
            cv2.circle(img, (circle_x_int, circle_y_int), circle_r_int, (255, 0, 255), 3)
            cv2.imshow("circle overlay", img)
            cv2.waitKey(0)
            CircleCentersX = np.append(CircleCentersX, circle[0][0])
            CircleCentersY = np.append(CircleCentersY, circle[0][1])
            
            # Add corresponding transforms to list
            StylusTipCoordsX = np.append(StylusTipCoordsX, x)
            StylusTipCoordsY = np.append(StylusTipCoordsY, y)
            StylusTipCoordsZ = np.append(StylusTipCoordsZ, z)

        else:
            # Convert circle parameters a, b, r to ints
            circles_asint = np.uint16(np.around(circles))
            for i in circles_asint[0, :]:
                center_asint = (i[0], i[1])
                cv2.circle(img, center_asint, 1, (0, 100, 100), 3)
                radius = i[2]
                cv2.circle(img, center_asint, radius, (255, 0, 255), 3)
            cv2.imshow("circle overlay", img)
            cv2.waitKey(0)

            for i in circles[0, :]:
                center = (i[0], i[1])

            if len(StylusTipCoordsX) > 0 and StylusTipCoordsX[-1] == x:
                # Repeated 3D coordinate indicates that tracking is lost
                print(f"Spatial tracking lost in frame {count}")
            else:
                # Add circle centers to list
                CircleCentersX = np.append(CircleCentersX, center[0])
                CircleCentersY = np.append(CircleCentersY, center[1])

                # Add corresponding transforms to list
                StylusTipCoordsX = np.append(StylusTipCoordsX, x)
                StylusTipCoordsY = np.append(StylusTipCoordsY, y)
                StylusTipCoordsZ = np.append(StylusTipCoordsZ, z)

                # Save image with circles drawn if necessary?
    StylusTipCoords = np.vstack((StylusTipCoordsX, StylusTipCoordsY, StylusTipCoordsZ))
    CircleCenters = np.vstack((CircleCentersX, CircleCentersY))
    print(CircleCenters)
    CalibrationMethod = 1

    # Run chosen calibration procedure
    if CalibrationMethod == 1:
        R, t = hand_eye_p2l(StylusTipCoords, CircleCenters, newCameraMtx)
        calibration = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        print("Extrinsic Matrix:", calibration)
        px, pxErrs = PixelValidation(calibration, StylusTipCoords, CircleCenters, intMtx)
        distErrs = DistanceValidation(calibration, StylusTipCoords, CircleCenters, intMtx)
        angularErrs = AngularValidation(calibration, StylusTipCoords, CircleCenters, intMtx)

        return calibration, px, pxErrs, distErrs, angularErrs
def distortionCalibration(checkerboardFiles):
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*23

    # Arrays to store object and image points from images
    objPts = [] # 3D points (world space)
    imgPts = [] # 2D points (image plane)
    
    n = len(checkerboardFiles)
    for count in range(n):
        fpath = checkerboardFiles[count]
        img = cv2.imread(fpath)

        #img = img[0,::-1,::-1,:] # may be unnecessary with sksurg preprocessing (compared to slicer)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, refine and add image and object points
        if ret:
            objPts.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgPts.append(corners)

            # Draw corners
            cornersdrawn = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            
            cv2.imshow(fpath[-13:], cornersdrawn)
            cv2.waitKey(0)
            # Save drawn image if necessary?


    ret, intMtx, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objPts, imgPts, gray.shape[::-1], None, None)
    print("distortion coefficients:", distCoeffs)
    for count in range(n):
        img = cv2.imread(checkerboardFiles[count])
        #img = img[0,::-1,::-1,:] # may be unnecessary with sksurg preprocessing (compared to slicer)
        
        # Undistort
        h, w = img.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(intMtx, distCoeffs, (w, h), 1, (w, h))
        img = cv2.undistort(img, intMtx, distCoeffs, None, newCameraMtx)
        # Save raw undistorted image if necessary?

    # Add the obtained intrinsic matrix and distortion coefficients to the UI
    print("intMtx:", intMtx)
    print("distCoeffs:", distCoeffs)
    return intMtx, distCoeffs

def PixelValidation(extMtx, pts3D, pts2D, intMtx):
    """
    Validates hand-eye calibration by finding pixel error

    Arguments:  extMtx (4x4):   extrinsic calibration matrix
                pts3D (nx3):    3D tracker coordinates
                pts2D (nx2):    2D image pixel coordinates
                intMtx (3x3):   intrinsic camera matrix

    Returns:    pxs (nx2):      reprojected pixels (i.e. from 3D points)
                pxErrs (n,):    vector of pixel errors
    """
    proj_pxs = []
    pxErrs = []
    n = pts3D.shape[1]
    for k in range(n):
        # Make 3D pt into column vector
        pt = pts3D[:, k]
        pt = np.reshape(pt, (3, 1))

        # Make 2D pixel into column vector
        px = pts2D[:, k]
        px = np.reshape(px, (2, 1))

        pt = np.vstack((pt, 1))
        
        # Register 3D pt to line
        camPt = extMtx @ pt

        # Convert 3D pt to homogeneous coordinates
        camPt /= camPt[2]
        camPt = camPt[0:2, :]
        camPt = np.vstack((camPt, 1))

        # Project point onto image using intrinsic matrix
        proj_px = intMtx @ camPt
        proj_pxs.append(proj_px)

        xErr = abs(proj_px[0, 0] - px[0, 0])
        yErr = abs(proj_px[1, 0] - px[1, 0])
        print("proj_px[0, 0]:", proj_px[0, 0])
        print("px[0, 0]:", px[0, 0])
        print("xErr:", xErr)

        print("proj_px[1, 0]:", proj_px[1, 0])
        print("px[1, 0]:", px[1, 0])
        print("yErr:", yErr)

        pxErrs.append(np.sqrt(xErr * xErr + yErr * yErr))
    pxErrs = np.reshape(pxErrs, (n, 1))
    return proj_pxs, pxErrs

def DistanceValidation(extMtx, pts3D, pts2D, intMtx):
    """
    Validates hand-eye calibration by finding distance error

    Arguments:  extMtx (4x4):   extrinsic calibration matrix
                pts3D (3xn):    3D tracker coordinates
                pts2D (2xn):    2D image pixel coordinates
                intMtx (3x3):   intrinsic camera matrix

    Returns:    distErrs (n,):  vector of distance errors
    """
    n = pts3D.shape[1]
    e = np.ones((n,))
    pts2D = np.linalg.inv(intMtx) @ np.vstack((pts2D, e))
    Y = np.empty((3, n))
    for i in range(n):
        x = pts2D[:, i]
        y = np.linalg.norm(x)
        z = x / y
        Y[:, i] = z

    pts2D = Y

    # Transform optical point to camera space
    pts3D = extMtx @ np.vstack((pts3D, e))

    # Store point magnitudes
    mags = np.empty((n, 1))
    for i in range(n):
        mags[i, 0] = np.sqrt(pts3D[0, i] * pts3D[0, i] + pts3D[1, i] * pts3D[1, i] + pts3D[2, i] * pts3D[2, i])
    
    # Normalize vector
    Y = np.empty((4, n))
    for i in range(n):
        x = pts3D[:, i]
        y = np.linalg.norm(x)
        z = x / y
        Y[:, i] = z
    pts3D = Y

    distErrs = np.empty((n, 1))
    for i in range(n):
        x = pts3D[0:3, i]
        q = pts2D[:, i]

        rot_axis = np.cross(x, q) / np.linalg.norm(np.cross(x, q))
        rot_angle = np.arccos(np.dot(x, q) / (np.linalg.norm(x) * np.linalg.norm(q)))
        R = np.hstack((rot_axis, rot_angle))

        angle = rot_angle
        distErrs[i, 0] = mags[i, 0] * np.tan(angle)
    return distErrs

def AngularValidation(extMtx, pts3D, pts2D, intMtx):
    """
    Validates hand-eye calibration by finding distance error

    Arguments:  extMtx (4x4):   extrinsic calibration matrix
                pts3D (3xn):    3D tracker coordinates
                pts2D (2xn):    2D image pixel coordinates
                intMtx (3x3):   intrinsic camera matrix

    Returns:    angularErrs (n,):   vector of angular errors
    """
    n = pts3D.shape[1]
    e = np.ones((n,))
    pts2D = np.linalg.inv(intMtx) @ np.vstack((pts2D, e))
    Y = np.empty((3, n))
    for i in range(n):
        x = pts2D[:, i]
        y = np.linalg.norm(x)
        z = x / y
        Y[:, i] = z
    pts2D = Y

    # Transform optical point to camera space
    pts3D = extMtx @ np.vstack((pts3D, e))

    # Normalize vector
    Y = np.empty((4, n))
    for i in range(n):
        x = pts3D[:, i]
        y = np.linalg.norm(x)
        z = x / y
        Y[:, i] = z
    pts3D = Y

    angularErrs = np.empty((n, 1))
    for i in range(n):
        x = pts3D[0:3, i]
        q = pts2D[:, i]

        rot_axis = np.cross(x, q) / np.linalg.norm(np.cross(x, q))
        rot_angle = np.arccos(np.dot(x, q) / (np.linalg.norm(x) * np.linalg.norm(q)))
        R = np.hstack((rot_axis, rot_angle))
        angularErrs[i, 0] = np.degrees(rot_angle)

    return angularErrs