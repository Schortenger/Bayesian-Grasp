#! /usr/bin/env python

import numpy as np
import math
from PIL import Image



#import rospy


import cv2


class GraspPlann(object):
    def __init__(self):
        pass

    # raw_moment() and moments_cov() calculate main axis
    def raw_moment(self, data, i_order, j_order):
        nrows, ncols = data.shape
        y_indices, x_indicies = np.mgrid[:nrows, :ncols]
        return (data * x_indicies ** i_order * y_indices ** j_order).sum()

    def moments_cov(self,data):
        data_sum = np.sum(data)
        #print type(data)
        m10 = self.raw_moment(data, 1, 0)
        m01 = self.raw_moment(data, 0, 1)
        x_centroid = m10 / data_sum
        y_centroid = m01 / data_sum
        u11 = (self.raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
        u20 = (self.raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
        u02 = (self.raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
        cov = np.array([[u20, u11], [u11, u02]])
        return cov,x_centroid,y_centroid

    def rgb2gray(self,rgb):

        im = np.array(Image.fromarray(rgb).convert('L'))       #convert RGB to gray picture
        return im

    def binaralization(self,image):
        image_shape = image.shape
        image_rows = image_shape [0]
        image_cols = image_shape [1]
        t = image
        #print ("t", t)
        #print t[:2]
        #print t[:100]
        count =0
        rand_x = []
        rand_y = []
        #mat = np.ones(480,640)
        rowedge1 = 90
        rowedge2 = 360
        columnedge1 = 70
        columnedge2 = 420
        t[0:rowedge1][::]=t[rowedge2:480]=0
        t[:, 0:columnedge1]=t[:, columnedge2:640]=0
        for i in range(image_rows):
            for j in range(image_cols):
                if t[i,j] >= 240  or t[i,j]<=10:#night
                #if t[i,j] >= 200 or t[i,j]<=50: #day
                    t[i,j]= 0
                    #mat[i,j]=0
                    count+=1
                else:
                    t[i,j] =255
                    rand_x.append(i)
                    rand_y.append(j)
        print ("0_pixel:", count)
        return rand_y,rand_x, t


    def theta_calculate(self,mask_roi):
        # try:
        #     cv_mask=self.bridge.imgmsg_to_cv2(mask_roi, 'mono8')
        # except CvBridgeError as e:
        #     print e
        cv_mask=np.asanyarray(mask_roi)
        #print ('--$$',type(cv_mask),cv_mask.shape)
        #cv_mask=cv_mask.reshape((480,640))
        #print ('....',type(cv_mask))
        cov, xmean, ymean = self.moments_cov(cv_mask)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[ :, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]

        theta = math.atan(y_v1 / x_v1)

        if theta>0:
            theta=theta-math.pi/2
        else:
            theta=theta+math.pi/2
        return ymean,xmean,theta

    def width_cal(self, y, x, theta, img): #pixel. pixel. radian
        count=1
        print "theta", theta
        print "costheta", math.cos(theta)   #cos(1.57)=0, which means it is radian measurement
        # theta=0.785
        # print"costheta",math.cos(theta) #0.707
        # print "tantheta", math.tan(theta) #0.999
        k = 200/148
        # k = k/math.cos(theta) #200/148 mm/pixel,vertical]
        #print "k", k
        yinit = y
        xinit = x
        plotx=[]
        ploty=[]
        print"calculate the width"
        while img[int(y),int(x)] >0 or img[int(y+1),int(x+math.tan(theta+math.pi/2))] >0  :
            y=y+1
            x=x+math.tan(theta+math.pi/2)
            print x,y
            plotx.append(x)
            ploty.append(y)
            if img[int(y),int(x)]>0:
                count+=1
        while img[int(yinit),int(xinit)] or img[int(yinit-1),int(xinit-math.tan(theta+math.pi/2))]>0:
            yinit=yinit-1
            xinit=xinit-math.tan(theta+math.pi/2)
            print xinit,yinit
            plotx.append(xinit)
            ploty.append(yinit)
            if img[int(yinit), int(xinit)] > 0:
                count+=1

        print count
        # width = count*k
        widthx=abs(x-xinit)*k
        widthy=abs(y-yinit)*k
        width = math.sqrt(math.pow(widthx,2)+math.pow(widthy,2))

        print width
        return width,plotx, ploty


    

    



if __name__=="__main__":
    grasp_plan=GraspPlann()
    """
    """
