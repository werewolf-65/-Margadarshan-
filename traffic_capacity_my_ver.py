import os
import logging
import logging.handlers
import random
import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

#to avoid some strange errors
cv2.ocl.setUseOpenCL(False)
random.seed(13)

AREA_COLOR=(255,0,0)

#some conatant parameters
IMAGE_DIR="./out"
VIDEO_SRC="new.mp4"
SHAPE= (1080,1920) #h x w
AREA_PTS=np.array([
    [[929,452],[400,1080],[1750,1080],[1040,453]],
])

def init_logging(to_file=False):
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if to_file:
        handler_file = logging.handlers.RotatingFileHandler("debug.log", maxBytes=1024 * 1024 * 400  # 400MB
                                                            , backupCount=10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger

def main():
    log=logging.getLogger("main")

    #create exit mask from points
    base=np.zeros(SHAPE + (3,),dtype='uint8')
    area_mask=cv2.fillPoly(base,AREA_PTS,(255,255,255))[:,:,0]
    total_space_in_mask=np.count_nonzero(area_mask)

    #capture the video soource
    cap=cv2.VideoCapture(VIDEO_SRC)

    frame_number = -1
    st=time.time() #start time

    while True:
        ret,frame=cap.read()
        if frame is None:
            log.error("Frame caputure failed, stopping ...")
            break
        #real frame number
        frame_number +=1

        #capacity counter code

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # this used for noise reduction at night time
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #convert to grayscale
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        cl1=clahe.apply(gray_frame)

        #use canny edge detection now
        edges =cv2.Canny(gray_frame,50,70) #detects the objects (vehicles)
        edges =~edges #invert to find free space
        blur=cv2.bilateralFilter(cv2.blur(edges,(21,21),100),9,200,200)
        _,threshold=cv2.threshold(blur,230,255,cv2.THRESH_BINARY)

        t=cv2.bitwise_and(threshold,threshold,mask=area_mask)

        free=np.count_nonzero(t) #counts the free space
        capacity=1 - float(free)/total_space_in_mask #finds the capacity

        #save the image
        img=np.zeros(frame.shape,frame.dtype)
        img[:,:]=AREA_COLOR
        mask=cv2.bitwise_and(img,img,mask=area_mask)
        cv2.addWeighted(mask,1,frame,1,0,frame)

        fig=plt.figure()
        fig.suptitle("Traffic Capacity: {}%".format(capacity*100),fontsize=13)

        plt.subplot(2,1,1)
        plt.imshow(frame)
        plt.title('Original Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,1,2)
        plt.imshow(t)
        plt.title('Traffic capacity map')
        plt.xticks([])
        plt.yticks([])

        fig.savefig(IMAGE_DIR + ("/processed_%s.png" % frame_number),dp1=500)

        log.debug("Capacity: {}%".format(capacity*100))

        if Exception:
            log.exception("Error!")

if __name__ == "__main__":
    log=init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating imaging directory %s", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
