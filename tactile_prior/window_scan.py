import socket
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import struct
import sys,pdb
import math
#from moment import GraspPlann

class Winscanning(object):
    def __init__(self):
        pass

    def winbuild(self, img, x_mean, y_mean):#gpy represents the x value of the plot
        # Origin_coordinates = (140,110)
        print ("gpy, gpx", x_mean, y_mean)
        window_size = 40
        xmin = x_mean - window_size/2
        xmax = x_mean + window_size/2
        ymin = y_mean - window_size/2
        ymax = y_mean + window_size/2
        #considering the sides of the whole image
        '''if xmin <= 0:
            xmin= xmin+ window_size/2
        if xmax >=255:
            xmin=xmin -window_size/2

        if ymin <=0:
            ymin = ymin + window_size/2
        if ymax >=255:
            ymin = ymin -window_size/2'''
        # draw window, the paraments are Image,Upper left coordinate, lower right coordinate, frame color, frame line thickness
        #cv2.rectangle(img, (ymax, xmin), (ymin, xmax), (0, 255, 0), 2)

        #cv2.imshow('local_pixel', img)
        #cv2.waitKey(0)

        imgarray = np.array(img)
        #print ('--imgarray', type(imgarray), imgarray.shape)
        local_pixel = imgarray[ymin:ymin+window_size,xmin:xmin+window_size]

        #print ('--local_pixel', type(local_pixel), local_pixel.shape,ymin)
        return  local_pixel

    # acquire local pixel and calculate the moment theta
    def theta_calculate(self, mask_roi):

        cv_mask = np.asanyarray(mask_roi)
        #print ('--$$',type(cv_mask),cv_mask.shape)
        #cv_mask=cv_mask.reshape((40, 40))
        #print ('....',type(cv_mask))
        cov, xmean, ymean = self.moments_cov(cv_mask)

        #print ('--cov', type(cov), cov.shape)
        evals, evecs = np.linalg.eig(cov)

        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]

        theta = math.atan(y_v1 / x_v1)

        if theta > 0:
            theta = theta - math.pi / 2
        else:
            theta = theta + math.pi / 2
        return ymean, xmean, theta


    def moments_cov(self,data):
        data_sum = np.sum(data)
        #print "data",type(data),data_sum
        #print data

        m10 = self.raw_moment(data, 1, 0)
        m01 = self.raw_moment(data, 0, 1)
        x_centroid = m10 / data_sum
        y_centroid = m01 / data_sum
        u11 = (self.raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
        u20 = (self.raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
        u02 = (self.raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
        cov = np.array([[u20, u11], [u11, u02]])
        return cov,x_centroid,y_centroid

 # raw_moment() and moments_cov() calculate main axis
    def raw_moment(self, data, i_order, j_order):
        nrows, ncols = data.shape
        y_indices, x_indicies = np.mgrid[:nrows, :ncols]
        return (data * x_indicies ** i_order * y_indices ** j_order).sum()

if __name__ == "__main__":
    winscan = Winscanning()
    """
    """