#! /usr/bin/env python

import numpy as np
import math
from PIL import Image

# import rospy


import cv2


class GraspPlann(object):
    def __init__(self):
        pass

    # raw_moment() and moments_cov() calculate main axis
    def raw_moment(self, data, i_order, j_order):
        nrows, ncols = data.shape
        y_indices, x_indicies = np.mgrid[:nrows, :ncols]
        return (data * x_indicies ** i_order * y_indices ** j_order).sum()

    def moments_cov(self, data):
        # data_sum = np.sum(data)
        # print type(data)
        m10 = self.raw_moment(data, 1, 0)
        m01 = self.raw_moment(data, 0, 1)
        x_centroid = m10 / data_sum
        y_centroid = m01 / data_sum
        u11 = (self.raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
        u20 = (self.raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
        u02 = (self.raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
        cov = np.array([[u20, u11], [u11, u02]])
        return cov, x_centroid, y_centroid

    def rgb2gray(self, rgb):

        im = np.array(Image.fromarray(rgb).convert('L'))  # convert RGB to gray picture
        return im

    def binaralization(self, image):
        image_shape = image.shape
        image_rows = image_shape[0]
        image_cols = image_shape[1]
        t = image
        count = 0
        for i in range(image_rows):
            for j in range(image_cols):
                if t[i, j] == 0:
                    t[i, j] = 0
                    count += 1
                else:
                    t[i, j] = 1
        return t

    def main_point_angle(self, mask_roi):
        # try:
        #     cv_mask=self.bridge.imgmsg_to_cv2(mask_roi, 'mono8')
        # except CvBridgeError as e:
        #     print e
        cv_mask = np.asanyarray(mask_roi)
        print ('--$$', type(cv_mask), cv_mask.shape)
        cv_mask = cv_mask.reshape((480, 640))
        print ('....', type(cv_mask))
        cov, xmean, ymean = self.moments_cov(cv_mask)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]

        # scale = 20
        # plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
        #          [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')
        # plt.plot([x_v2 * -scale, x_v2 * scale],
        #          [y_v2 * -scale, y_v2 * scale], color='blue')
        theta = math.atan(y_v1 / x_v1)
        if theta > 0:
            theta = theta - math.pi / 2
        else:
            theta = theta + math.pi / 2
        return xmean, ymean, theta


if __name__ == "__main__":
    grasp_plan = GraspPlann()
    """
    """
