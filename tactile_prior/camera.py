#!/usr/bin/env python

import socket
import numpy as np
import cv2
import os
import time
import struct
import sys,pdb
sys.path.insert(0, '/usr/local/lib/')
import pyrealsense2 as rs
#pdb.set_trace()

class Camera(object):

    def __init__(self):


        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Data options (change me)
        self.im_height = 480
        self.im_width = 640
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, 30)

        self.align_to = rs.stream.color

        self.align = rs.align(self.align_to)

        self.pipeline.start(self.config)

        self.intrinsics=np.array([[616.734, 0, 322.861],
                                 [0, 616.851, 234.728],
                                 [0, 0, 1, ]])

        self.cam_pose = np.array([[-0.99886392,    -0.03496858,    -0.032374 , 0.04717288],
                                   [-0.03647167 ,   0.99822577 ,  0.04706524, -0.49527845],
                                   [0.03067075 ,  0.04819251 ,  -0.99836706, 0.71468168],
                                   [0,  0,  0,  1]])


        i = 0
        while i<10:
            self.get_data(False)
            i+=1

    def get_data(self,get_depth = True):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            # cv2.imshow('rgb',color_image)
            # cv2.waitKey()
            #color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
            dis_image = np.asanyarray(depth_frame.get_data()).astype(float)
            if get_depth:
                for y in xrange(480):
                    for x in xrange(640):
                        dis_image[y, x] = depth_frame.get_distance(x, y)
            break
        return color_image,dis_image

    def __del__(self):
        # Stop streaming
        self.pipeline.stop()



