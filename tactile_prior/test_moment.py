#coding:utf-8
#!/usr/bin/env python
import cv2
import sys,os,pdb
import matplotlib.patches as patches
#pdb.set_trace()
#from soft_grasp.softnet import SoftGraspNet, listMean, plotGrasp
from TakkTile_usb.TakkTile import TakkTile
from camera3 import Camera
from ur5 import UR5
from timer import Timer
from moment import GraspPlann
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
from logger import Logger
from window_scan import Winscanning
from PIL import Image, ImageDraw
#from ur5_manipulator_node import RobotiqCGripper, UR5_Gripper_Manipulator
print ('import done')

'''
20190.03
base -> camera_frame (x,y,z,quaternion) : 0.008207 -0.463471 0.764447 0.999636 -0.012670 -0.012789 -0.020090
Hgrid2worldAvg
array([[ 0.9993518 , -0.02584511, -0.02506014,  0.00820739],
       [-0.02481738, -0.99887173,  0.04048897, -0.46347139],
       [-0.02607831, -0.0398408 , -0.99886567,  0.76444656],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
mtx
array([[652.09308916,   0.        , 309.395794  ],
       [  0.        , 649.56905455, 243.97166679],
       [  0.        ,   0.        ,   1.        ]])
'''


def main():


    num_pick=0  #calculate the grasping number
    no_object=0
    timer=Timer()
    tact = TakkTile()
    print tact.UIDs
    print tact.alive
    # try:
    #     count = int(sys.argv[1])
    # except:
    #     count = 1
    # import time
    # tact.startSampling()

    camera=Camera()
    #class of image moment

    #num , pressure, temprature = tact.getrawdata()
    #print pressure

    #pdb.set_trace()
    Grasp_Plan=GraspPlann()
    winscan = Winscanning()
    #show the value of TakkTile sensors
    # for i in range(count):
    #     num , pressure, temprature = tact.getDataRaw()
    # print num, pressure, temprature

    #pdb.set_trace()
    #gripper = RobotiqCGripper()
    #gripper.close()
    #pdb.set_trace()


    # configuration of UR5
    ur5 = UR5(tcp_host_ip='192.168.1.2', tcp_port=30003)
    cam_joint=np.array([-2.38,-111,111,-90,-90,161])*3.1415/180.
    # [60,-120,112,-80,-90,200]
    ur5.move_joints(cam_joint)

    # grasp intermediate points
    grasp_home = [0 ,-0.350 ,0.400 ,3.1415 ,0 ,0]

    grasp_times = 500
    grasp_time = 1

    #pdb.set_trace()
    grasp_success = False

    ur5.move_joints(cam_joint)

    rgb_back, depth_back = camera.get_data()

    gray_back = Grasp_Plan.rgb2gray(rgb_back)
    # ur5.move_joints(cam_joint)
    ROI_back = gray_back  # [140:460, 110:590]
    print 'depth_back',depth_back
    #cv2.imshow('gray_back', ROI_back)
    cv2.imshow('depth_back', depth_back)
    cv2.waitKey(0)




    #ur5.move_joints(cam_joint)
    # if grasp_time == 1 or grasp_success:
    #     rgb_back, depth_back = camera.get_data()
    #     gray_back = Grasp_Plan.rgb2gray(rgb_back)
    # #ur5.move_joints(cam_joint)
    #     ROI_back = gray_back#[140:460, 110:590]
    #     cv2.imshow('gray_back', ROI_back)
    #     cv2.waitKey(0)


    #recording time
    timer.tic()
    # get color image with grasped object
    rgb,depth=camera.get_data()
    print 'depth',depth
    gray_img = Grasp_Plan.rgb2gray(rgb)
    ROI_img = gray_img#[140:460, 110:590]
    #cv2.imshow('gray_img', ROI_img)
    cv2.imshow('depth',depth)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    subdepth= np.array(depth_back)-np.array(depth)
    subtracted = ROI_img - ROI_back  # gray_img of grasping object
    print "subtracted", type(subtracted), subtracted.shape
    #（meany,meanx）represents the possible grasp points
    # binasub represents the image with only 0 and 255 contrary to substracted
    meany, meanx, binasub = Grasp_Plan.binaralization(subdepth)
    # print ('meany', type(meany), meany[0], len(meany))
    # print ('meanx', type(meanx), meanx[0], len(meanx))
    # print ('binasub', type(binasub), binasub.shape)
    print 'binasub',binasub
    plt.imshow(binasub)
    plt.show()
    pdb.set_trace()

    # wholex, wholey , wholetheta = Grasp_Plan.theta_calculate(binasub)
    # print "wholey, wholex,wholetheta", wholey, wholex, wholetheta

    wholedeg = math.degrees(wholetheta)        #deg=[0,180]
    # if deg >= 90:
    #     deg =deg-90
    # else:
    #     deg = deg+90
    # draw the point and the degree
    # p1x = gpxmean + gpx - winsize / 2
    # p1y = gpymean + gpy - winsize / 2
    wholep1 = (int(wholey), int(wholex))

    wholep2 = (int(int(wholey) + math.tan(wholetheta) * (int(wholex) - 150)), 150)
    #print ("gpy gpx", gpy, gpx)
    #print("p1 p2", p1, p2)
    # cv.circle(img, point, point_size, point_color, thickness)
    # cv2.circle(binasub, p1, point_size, (0,0,255), thickness)
    # cv2.circle(binasub, p2, point_size, (0, 0, 255),thickness)
    cv2.line(binasub, wholep1, wholep2, (155, 155, 155), 5)
    plt.imshow(binasub, cmap='jet')
    plt.show()
    cv2.destroyAllWindows()

    # width,plotx, ploty = Grasp_Plan.width_cal(203,303, wholetheta, binasub)
    # print "width",width
    # plt.scatter(plotx, ploty, s=20, c='r', label='width')
    # plt.imshow(binasub, cmap='jet')
    # plt.show()
    # cv2.destroyAllWindows()
    # gripk = -215/135 #215:0mm;0:135mm
    # gripwidth = 215+gripk*width
    # print "gripwidth", gripwidth



    #ur5.soft_grasp([303,203, 0.132, wholedeg])
    #gripper.graclose(gripwidth)
    #pdb.set_trace()



    # choose a grasp point from so many possible points
    #generate random number
    num = random.randint(0, len(meany))
    #grasp point
    gpy = meany[num]
    gpx = meanx[num]
    #print ('meany', type(meany), gpy, len(meany))
    #print ('meanx', type(meanx), gpx, len(meanx))

    #plt.figure(8)
    #plt.plot(gpy, gpx, marker='o', c='r')
    #plt.show()

    #build a window on the image
    winsize = 40
    #if on the edge of the picture, then choose another point
    img_edge = 50
    while True:
        if abs(gpy-320)>320 -img_edge or abs(gpx-240)>240 - img_edge:
            print( gpy, gpx, "The point is on the edge, and another point will be chosen")
            num = random.randint(0, len(meany))
            gpy = meany[num]
            gpx = meanx[num]
        else:
            break
    #show the window
    print("gpy, gpx", gpy, gpx)
    cv2.rectangle(binasub, (gpy+winsize/2,gpx-winsize/2), (gpy - winsize / 2, gpx + winsize / 2), (255, 255, 0), 3)
    plt.imshow(binasub, cmap='jet')
    plt.show()


    plt.figure(figsize=(6.4, 4.8))

    #define the size of image
    x_lim=640
    y_lim=480
    plt.xlim(1, x_lim)
    plt.ylim(1, y_lim)

    #copy the pixel in the window
    local_win = winscan.winbuild(binasub, int(gpy), int(gpx))
    #plt.figure()
    #plt.imshow(local_win, cmap='jet')
    #plt.show()
    #print("local_win", local_win.shape, np.sum(local_win))
    #calculate local theta
    gpymean, gpxmean, theta = winscan.theta_calculate(local_win)
    #transform to global pixel coordination
    #gpymean = gpymean+gpy-winsize/2
    #gpxmean = gpxmean+gpx-winsize/2
    deg = math.degrees(theta)        #deg=[0,180]
    if deg >= 90:
        deg =deg-90
    else:
        deg = deg+90

    print "complete grasping structure"
    grasp_depth = depth[int(gpxmean),int(gpymean)]
    print "(gpymean, gpxmean,grasp_depth,theta)",gpymean, gpxmean, grasp_depth,theta


    # draw the point and the degree
    p1x=gpxmean+gpx-winsize/2
    p1y=gpymean+gpy-winsize/2
    p1 = ( int(p1y), int(p1x))

    p2 = (int(int(p1y)+math.tan(theta)*(int(p1x)-150)), 150)
    print ("gpy gpx", gpy,gpx)
    print("p1 p2", p1,p2)
    #cv.circle(img, point, point_size, point_color, thickness)
    #cv2.circle(binasub, p1, point_size, (0,0,255), thickness)
    #cv2.circle(binasub, p2, point_size, (0, 0, 255),thickness)
    cv2.line(binasub, p1, p2, (155,155,155), 5)
    plt.imshow(binasub, cmap='jet')
    plt.show()
    cv2.destroyAllWindows()
    #pdb.set_trace()
        #     gripper.close()
        #
        #     #tactile information
        #     num, pressure, temprature = tact.getrawdata()
        #     pre_p = pressure
        #     print "Before grasping, the pressure is ", pre_p
        #
        #     gripper.activate()
        #     #grasp the object
        #     ur5.soft_grasp([p1y, p1x ,grasp_depth,deg])
        #     print "start to grasp"
        #
        #     #close the gripper
        #     gripper.close()
        #
        #     grasp_time = grasp_time + 1
        #
        #
        #
        #     #pdb.set_trace()
        #     #ur5.move_pose(grasp_home)
        #
        #
        #     #tactile information`
        #     ur5.move_pose([-0.18312, -0.30185, 0.40340,3.1415 ,0 ,0 ])
        #     time.sleep(2)
        #
        #     num, pressure, temprature = tact.getrawdata()
        #     print "After grasping, the pressure is ", pressure
        #     later_p = pressure
        #     pressure_suc = abs(later_p[0] - pre_p[0]) + abs(later_p[1] - pre_p[1]) + abs(
        #         later_p[2] - pre_p[2]) + abs(later_p[3] - pre_p[3]) + abs(later_p[4] - pre_p[4])
        #     if pressure_suc >350:
        #         grasp_success = True
        #         print "grasp successfully"
        #     else:
        #         grasp_success = False
        #         print "grasp is failure, regrasping"
        #
        # #pdb.set_trace()
        # #shake the robot arm for 8s
        # shake_point1 = [0 ,-0.350 ,0.400 ,3.1415 ,0 ,0]
        # shake_point2 = [-0.052, -0.567, 0.23, 3.1415, 0, 0]
        # i=1
        # while i< 3:
        #     ur5.shake(shake_point1)
        #     ur5.shake(shake_point2)
        #     i=i+1
        # #ur5.shake(shake_point)
        # #ur5.shake([0 ,-0.350 ,0.8 ,3.1415 ,0 ,0])
        #
        # num, pressure, temprature = tact.getrawdata()
        # print "After shaking, the pressure is ", pressure
        # third_p = pressure
        # pressure_dif = abs(later_p[0]-third_p[0])+abs(later_p[1]-third_p[1])+abs(later_p[2]-third_p[2])+abs(later_p[3]-third_p[3])+abs(later_p[4]-third_p[4])
        # if pressure_dif <=800 & pressure_dif >=50:
        #     print "The object drops off"
        # elif pressure_dif > 800:
        #     print "The object is slippery"
        # else:
        #     print "The object is stable"
        #
        # ur5.move_joints(cam_joint)
        # rgb_back, depth_back = camera.get_data()
        # gray_back = Grasp_Plan.rgb2gray(rgb_back)
        # # ur5.move_joints(cam_joint)
        # ROI_back = gray_back  # [140:460, 110:590]
        # cv2.imshow('gray_back', ROI_back)
        # cv2.waitKey(1000)
        #
        #
        # ur5.soft_place([450, 270 ,0.18 , 90])
        # gripper.activate()
        #
        # if grasp_time == grasp_times:
        #     break
        #
        # print "start to next grasp"
        # grasp_success = False
        #








if __name__=='__main__':
    main()

    #create_result_path()







