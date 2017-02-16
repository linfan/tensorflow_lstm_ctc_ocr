import cv2
import numpy as np
import sys

if __name__ == '__main__':  
    if len(sys.argv) < 2:
        print "need a filename as argument"
        #return
    image = cv2.imread(sys.argv[1])
    size = (256, 64)  
    shrink = cv2.resize(image, size, interpolation=cv2.INTER_AREA) 
    gray = cv2.cvtColor(shrink, cv2.COLOR_BGR2GRAY)
    for idx, line in enumerate(gray):
        for idy, item in enumerate(line):
            gray[idx][idy] = 255 - gray[idx][idy]
            #if item > 127:
            #    gray[idx][idy] = 255
            #else:
            #    gray[idx][idy] = 0
    #cv2.imshow("Image", gray)  
    #cv2.waitKey(0)  
    #cv2.destroyAllWindows()  
    fname = sys.argv[1]
    cv2.imwrite(fname, gray)
