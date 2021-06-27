import numpy as np
import cv2 as cv
import glob

chessboardSize = (6, 9)
frameSize = (3024, 4032)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0],
                       0:chessboardSize[1]].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('./calibration/*.jpg')
for fname in images:
    print("o")
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    print("before")
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None)

print("Camera calibrated ", ret)
print(cameraMatrix)
print(dist)
print(rvecs)
print(tvecs)

imgL = cv.imread('img/towerL.png')
imgR = cv.imread("img/towerR.png")
h, w = imgL.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix, dist, (w, h), 1, (w, h))

dstL = cv.undistort(imgL, cameraMatrix, dist, None, newCameraMatrix)
dstR = cv.undistort(imgR, cameraMatrix, dist, None, newCameraMatrix)

x, y, w, h = roi
dstL = dstL[y:y+h, x:x+w]
dstR = dstR[y:y+h, x:x+w]

cv.imwrite('calibratedL.png', dstL)
cv.imwrite('calibratedR.png', dstR)

mapx, mapy = cv.initUndistortRectifyMap(
    cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)

dstL = cv.remap(imgL, mapx, mapy, cv.INTER_LINEAR)
dstR = cv.remap(imgR, mapx, mapy, cv.INTER_LINEAR)

x, y, w, h = roi
dstL = dstL[y:y+h, x:x+w]
dstR = dstR[y:y+h, x:x+w]
cv.imwrite('calibratedMapL.png', dstL)
cv.imwrite('calibratedMapR.png', dstR)
