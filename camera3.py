#!/usr/bin/env python
import time
import pyrealsense2 as rs
import numpy as np
import cv2

from PIL import Image


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07"]

def find_device_that_supports_advanced_mode() :
    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices();
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No device that supports advanced mode was found")

class Camera(object):
    def __init__(self):

        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        # Loop until we successfully enable advanced mode
        while not advnc_mode.is_enabled():
            print("Trying to enable advanced mode...")
            advnc_mode.toggle_advanced_mode(True)
            # At this point the device will disconnect and re-connect.
            print("Sleeping for 5 seconds...")
            time.sleep(5)
            # The 'dev' object will become invalid and we need to initialize it again
            dev = find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        with open("/home/schortenger/Desktop/IROS/tactile_prior/camera_config.json") as f:
            load_string = json.load(f)
        if type(next(iter(load_string))) != str:
            load_string = {k.encode('utf-8'): v.encode("utf-8") for k, v in load_string.items()}
        json_string = str(load_string).replace("'", '\"')
        advnc_mode.load_json(json_string)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline.start(self.config)
        for i in range(10):
            self.get_data(False)
        print('Camera is initialized')

    def get_data(self,get_depth=True):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            self.color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color_frame.get_data())
            dis_none = np.zeros([480, 640])
            if get_depth:
                depth_frame = aligned_frames.get_depth_frame()
                depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(float)
                return color_image, depth_image/1000
            else:
                return color_image, dis_none
    def get_intrinsics(self):
        return self.color_intrinsics




if __name__ == "__main__":
    camera = Camera()
    color_image, depth_image = camera.get_data(get_depth=True)
    print(color_image.shape, depth_image.shape)

    print(camera.color_intrinsics)
    count = 0.0
    # for x in range(depth_image.shape[0]):
    #     for y in range(depth_image.shape[1]):
    #         if depth_image[x,y]==0:
    #             count = count+1
    #
    # plt.subplot(211)
    # plt.imshow(color_image)
    # plt.subplot(212)
    # plt.imshow(depth_image)
    # plt.show()
    # camera.pipeline.stop()