#!/usr/bin/env python
import cv2
import sys,os
from timer import Timer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pdb

class Logger():
    def __init__(self):
        self.raw_image_path = None
        self.num=None
        self.path=self.create_result_path()


    def save_image(self,rgb,depth,num):
        rgb_name='result_%0.2d_rgb_%0.2d.jpg'%(self.num,num)
        depth_name='result_%0.2d_depth_%0.2d.tiff'%(self.num,num)
        rgb_name=os.path.join(self.raw_image_path,rgb_name)
        depth_name=os.path.join(self.raw_image_path,depth_name)
        cv2.imwrite(rgb_name,rgb)
        depth=depth*1000.0
        depth=depth.astype(np.uint32)
        depth=Image.fromarray(depth)
        depth.save(depth_name)




    def record_result(self,rgb, num_pick, result, all_results, point_1, all_points):
        classes, bboxes, scores = all_results
        path=self.path

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(rgb, aspect='equal')
        for i in range(len(classes)):
            class_name, bbox, score = classes[i], bboxes[i], scores[i]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='blue', linewidth=2.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
        class_name, bbox = result
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=4.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='red', alpha=0.8),
                fontsize=14, color='white')
        for point in all_points:
            ax.scatter(point[0], point[1], c='k')
        ax.scatter(point_1[0], point_1[1], c='r', marker='8')
        name = os.path.join(path, ('pick_%0.2d' % num_pick) + '.png')
        #plt.show()
        plt.savefig(name)
        plt.close()

    def create_result_path(self):
        file_path = './output'
        files = os.listdir(file_path)
        nums = []
        if len(files) == 0:
            path = os.path.join(file_path, 'result_%2d' % 1)
            os.makedirs(path)
            return path
        for file in files:
            path = os.path.join(file_path, file)
            if os.path.isdir(path):
                nums.append(int(path.split('_')[1]))
        nums.sort()
        num = nums[-1] + 1
        path = os.path.join(file_path, 'result_%0.2d' % num)
        self.num=num
        os.makedirs(path)
        self.raw_image_path=os.path.join(path,'raw_image')
        os.makedirs(self.raw_image_path)
        return path

    # def brighter(self,img):
    #     w = img.shape[1]
    #     h = img.shape[0]
    #
    #     for xi in range(0, w):
    #         for xj in range(0, h):
    #             img[xj, xi, 0] = int(img[xj, xi, 0] * 0.2)
    #             img[xj, xi, 1] = int(img[xj, xi, 1] * 0.2)
    #             img[xj, xi, 2] = int(img[xj, xi, 2] * 0.2)