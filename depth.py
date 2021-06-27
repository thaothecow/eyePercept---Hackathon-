import numpy as np
import cv2 as cv


# rotates images


def rotate(img):
    rotated = cv.transpose(img)
    rotated = cv.flip(rotated, flipCode=1)
    return rotated

# resizes images


def downscale(img):
    resized = cv.resize(img, (1000, 1000))
    return resized


if __name__ == "__main__":
    # Imports the image from the png files
    imgL = cv.imread('./calibratedL.png', cv.IMREAD_UNCHANGED)
    imgR = cv.imread('./calibratedR.png', cv.IMREAD_UNCHANGED)

    # Converts img to grayscale img
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Combining image to a single image that gives a depth
    stereo = cv.StereoBM_create(numDisparities=80, blockSize=5)
    disparity = stereo.compute(grayR, grayL)

    # Prefilter
    blur = cv.GaussianBlur(disparity, (5, 5), 0)
    filtered = cv.inRange(blur, 700, 1800)

    # ------ finding the white pixels (to identify an object) ------

    #
    img_rgb = cv.cvtColor(filtered, cv.COLOR_BGR2RGB)
    lower_grey = np.array([130, 130, 130])
    upper_grey = np.array([255, 255, 255])

    mask = cv.inRange(img_rgb, lower_grey, upper_grey)
    mask = downscale(mask)

    cv.imwrite("Output.png", mask)
    cv.imshow('mask', mask)
    cv.waitKey(0)
