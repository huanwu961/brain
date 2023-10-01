import cv2 as cv
from PIL import ImageGrab
import numpy as np


if __name__ == '__main__':
    frame = ImageGrab.grab()
    key = cv.waitKey(1)
    while key != ord('q'):
        frame = ImageGrab.grab()
        frame = cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR)
        cv.imshow("frame", frame)
        key = cv.waitKey(1)