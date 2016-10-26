import cv2
import numpy as np
import sys

if __name__ == '__main__':  
    for num in range(0,21):
        fname = "{:08d}.jpg".format(num)
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for idx, line in enumerate(gray):
            for idy, item in enumerate(line):
                if gray[idx][idy] >= 150:
                    gray[idx][idy] = 255
                else:
                    gray[idx][idy] = 0
                
        cv2.imwrite(fname, gray)
