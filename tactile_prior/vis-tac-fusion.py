#coding:utf-8
#!/usr/bin/env python
import cv2
import sys,os,pdb
import matplotlib.patches as patches
#pdb.set_trace()
from soft_grasp.softnet import SoftGraspNet, listMean, plotGrasp
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
from ur5_manipulator_node import RobotiqCGripper, UR5_Gripper_Manipulator
from data.writedata import Data
from data.process import tac_metric
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


    camera=Camera()
    #class of image moment

    num , pressure, temprature = tact.getrawdata()
    print pressure

    #pdb.set_trace()
    Grasp_Plan=GraspPlann()
    winscan = Winscanning()
    writedata = Data()
    metric = tac_metric()


    #pdb.set_trace()
    gripper = RobotiqCGripper()
    #gripper.close()

    # configuration of UR5
    ur5 = UR5(tcp_host_ip='192.168.1.2', tcp_port=30003)
    cam_joint=np.array([-2.38,-111,111,-90,-90,11])*3.1415/180.
    # [60,-120,112,-80,-90,200]
    ur5.move_joints(cam_joint)

    # grasp intermediate points
    grasp_home = [0 ,-0.350 ,0.400 ,3.1415 ,0 ,0]
    #pdb.set_trace()

    grasp_times = 200
    grasp_time = 1

    #pdb.set_trace()
    grasp_success = False

    ur5.move_joints(cam_joint)
    #pdb.set_trace()
    if grasp_time == 1 :
        rgb_back, depth_back = camera.get_data()
        gray_back = Grasp_Plan.rgb2gray(rgb_back)
        # ur5.move_joints(cam_joint)
        ROI_back = gray_back  # [140:460, 110:590]
        cv2.imshow('gray_back', ROI_back)
        cv2.waitKey(0)

    while grasp_time <= grasp_times:

        # move ur5 to camera position to capture background images
        ur5.move_joints(cam_joint)

        #recording time
        timer.tic()
        # get color image with grasped object
        rgb,depth=camera.get_data()
        rgboriginal = rgb
        gray_img = Grasp_Plan.rgb2gray(rgb)
        ROI_img = gray_img#[140:460, 110:590]
        cv2.imshow('gray_img', ROI_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        subtracted = ROI_img - ROI_back  # gray_img of grasping object
        print "subtracted", type(subtracted), subtracted.shape
        #（meany,meanx）represents the possible grasp points
        # binasub represents the image with only 0 and 255 contrary to substracted
        meany, meanx, binasub = Grasp_Plan.binaralization(subtracted)


        # choose a grasp point from so many possible points
        #generate random number
        num = random.randint(0, len(meany))
        #grasp point
        gpy = meany[num]
        gpx = meanx[num]

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
        #print("gpy, gpx", gpy, gpx)
        #plt.ion()
        cv2.rectangle(binasub, (gpy+winsize/2,gpx-winsize/2), (gpy - winsize / 2, gpx + winsize / 2), (255, 255, 0), 3)
        plt.imshow(binasub, cmap='jet')
        #plt.show()

        #plt.ioff()
        plt.close()
        #



        plt.figure(figsize=(6.4, 4.8))

        #define the size of image
        # x_lim=640
        # y_lim=480
        # plt.xlim(1, x_lim)
        # plt.ylim(1, y_lim)

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


        # draw the point and the degree
        p1x=gpxmean+gpx-winsize/2
        p1y=gpymean+gpy-winsize/2
        grasp_depth = depth[int(p1y), int(p1x)]
        print "(gpymean, gpxmean,grasp_depth,theta)", gpymean, gpxmean, grasp_depth, theta

        p1 = ( int(p1y), int(p1x))

        p2 = (int(int(p1y)+math.tan(-theta)*(150-int(p1x))), 150)
        print ("gpy gpx", gpy,gpx)
        print("p1 p2", p1,p2)
        #cv.circle(img, point, point_size, point_color, thickness)
        #cv2.circle(binasub, p1, point_size, (0,0,255), thickness)
        #cv2.circle(binasub, p2, point_size, (0, 0, 255),thickness)
        #lt.ion()
        cv2.line(binasub, p1, p2, (155,155,155), 5)
        plt.imshow(binasub, cmap='jet')
        #plt.show()

        #pdb.set_trace()
        #
        #plt.ioff()
        #time.sleep(1)
        plt.close()


        Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//original/yogurt/x y theta:" + str(int(p1y))+str(int(p1x))+str(theta) +str(time.strftime('  %Y-%m-%d %H:%M:%S',time.localtime(time.time())))+ ".png"
        cv2.imwrite(Img_Name, rgboriginal)
        print "image saved"
        #cv2.imwrite(Img_Name, rgb)
        # Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//original/0406/img"  + str(
        #     time.strftime('  %Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ".png"


        cv2.destroyAllWindows()
        #pdb.set_trace()

        # wholex, wholey, wholetheta = Grasp_Plan.theta_calculate(binasub)
        # print "wholey, wholex,wholetheta", wholey, wholex, wholetheta
        #
        # wholedeg = math.degrees(wholetheta)  # deg=[0,180]
        #
        # wholep1 = (int(wholey), int(wholex))
        # wholep2 = (int(int(wholey) + math.tan(wholetheta) * (int(wholex) - 150)), 150)
        # cv2.line(binasub, wholep1, wholep2, (155, 155, 155), 5)
        # plt.imshow(binasub, cmap='jet')
        # plt.show()
        # cv2.destroyAllWindows()

        #pdb.set_trace()
        #gripper.close()

        #tactile information
        num, pressure, temprature = tact.getrawdata()
        pre_p = pressure
        print "Before grasping, the pressure is ", pre_p
        writedata.apdata(pre_p)

        #gripper.activate()
        #grasp the object
        print "start to grasp"
        #ur5.io_open()
        gripper.activate()

        # #close the gripper
        # width, plotx, ploty = Grasp_Plan.width_cal(p1x,p1y, wholetheta, binasub)
        # print "width", width
        # plt.scatter(plotx, ploty, s=20, c='r', label='width')
        # plt.imshow(binasub, cmap='jet')
        # plt.show()
        # cv2.destroyAllWindows()
        # gripk = -215 / 135  # 215:0mm;0:135mm
        # gripwidth = 215 + gripk * width
        # print "gripwidth", gripwidth
        ur5.soft_grasp([p1y, p1x,  grasp_depth ,deg])
        #ur5.io_close()
        gripper.close()
        # gripper.graclose(gripwidth)


        #pdb.set_trace()
        grasp_time = grasp_time + 1




        #tactile information`
        ur5.move_pose([-0.18312, -0.30185, 0.40340,3.1415 ,0 ,0 ])
        time.sleep(1)

        num, pressure, temprature = tact.getrawdata()
        print "After grasping, the pressure is ", pressure
        later_p = pressure

        writedata.apdata(later_p)
        # pressure_suc = abs(later_p[0] - pre_p[0]) + abs(later_p[1] - pre_p[1]) + abs(
        #     later_p[2] - pre_p[2]) + abs(later_p[3] - pre_p[3]) + abs(later_p[4] - pre_p[4])
        #max_dif = max(abs(later_p[0] - pre_p[0]), abs(later_p[1] - pre_p[1]),abs(later_p[2] - pre_p[2]), abs(later_p[3] - pre_p[3]), abs(later_p[4] - pre_p[4]))

        # if max_dif >90:
        #     grasp_success = True
        #     print "grasp successfully"
        # else:
        #     grasp_success = False
        #     print "grasp is failure, regrasping"
        strin = raw_input("please input, 1-success, 2-fail:")
        if strin == '1':
            grasp_success = True
            print "grasp successfully"
            writedata.suc_result()
        if strin == '2':
            grasp_success = False
            print "grasp is failure"
            symbol = "failure"
            writedata.fai_result()
            writedata.endonepro()

        if grasp_success:
            #pdb.set_trace()
            # shake the robot arm for 8s
            shake_point1 = [0, -0.350, 0.375, 3.1415, 0, 0]
            shake_point2 = [-0.310, -0.448, 0.450, 3.1415, 0, 0]
            shake_point3 = [-0.12866,-0.32689, 0.60290, 3.1415,0,0]
            shake_point4 = [-0.12866, -0.32689, 0.32651, 3.1415, 0, 0]

            shake_startpoint = np.array([45, -110, 110, -90, -90, 11]) * 3.1415 / 180.
            shake_joint1 = np.array([45, -110, 110, -90, -50, 11]) * 3.1415 / 180.
            shake_joint2 = np.array([45, -110, 110, -90, -136, 11]) * 3.1415 / 180.
            shakematrix = [[0]*5 for _ in range(5)]
            shakematrix = np.array(shakematrix)
            shakematrix[0] = later_p

            i=0
            while i< 2:
                ur5.shake_pose(shake_point3)
                num, pressure, temprature = tact.getrawdata()
                print "After shaking_pose",i , "the pressure is ", pressure
                pre_p=np.array(pre_p)
                pressure=np.array(pressure)

                writedata.apdata(pressure)
                shakematrix[i+1]=pressure
                ur5.shake_pose(shake_point4)
                i = i + 1

            j = 0
            ur5.move_joints(shake_startpoint)
            #pdb.set_trace()
            while j < 2:
                ur5.shake_joints(shake_joint1)
                num, pressure, temprature = tact.getrawdata()
                pre_p = np.array(pre_p)
                pressure = np.array(pressure)
                print "After shaking_joints", i, "the pressure is ", pressure
                writedata.apdata(pressure)
                shakematrix[j + 3] = pressure
                ur5.shake_joints(shake_joint2)
                j = j + 1
            #ur5.shake(shake_point)
            #ur5.shake([0 ,-0.350 ,0.8 ,3.1415 ,0 ,0])

            num, pressure, temprature = tact.getrawdata()
            print "After shaking, the pressure is ", pressure
            pressure = np.array(pressure)
            writedata.apdata(pressure)

            #third_p = pressure
            # pressure_dif = abs(later_p[0]-third_p[0])+abs(later_p[1]-third_p[1])+abs(later_p[2]-third_p[2])+abs(later_p[3]-third_p[3])+abs(later_p[4]-third_p[4])
            # shake_dif = abs(pre_p[0] - third_p[0]) + abs(pre_p[1] - third_p[1]) + abs(
            #     pre_p[2] - third_p[2]) + abs(pre_p[3] - third_p[3]) + abs(pre_p[4] - third_p[4])
            strin = raw_input("if stable, please input 3. if drop,please input 4:")

            if strin == '4':
                print"The object drops off"
                writedata.drop_result()
                symbol = "drop"

            elif strin == '3':
                print"The object is stable or slippery"
                writedata.stable_result()
                symbol = "slippery"
            # if pressure_dif <=500 & pressure_dif >=50 or shake_dif <250:
            #     print "The object drops off"
            # elif pressure_dif > 800:
            #     print "The object is slippery"
            # else:
            #     print "The object is stable"

            ur5.move_joints(cam_joint)
            #grasp_success = False
            #pdb.set_trace()

        #ur5.move_pose([-0.15267, -0.43011, 0.336, 3.1415, 0, 0])
        #ur5.io_open()
        writedata.endonepro()

        if symbol == "failure":
            score = 0
        else:
            writedata.shakedata(shakematrix)
            score = metric.score(shakematrix, symbol)
        writedata.writescore(score)
        print "The grasp score is:", score
        cv2.circle(rgb, (int(p1y), int(p1x)), 5, (0, 255, 0), -1)
        linelength = 30
        dp = (int(int(p1y) + math.sin(-theta) * linelength), int(int(p1x) + math.cos(-theta) * linelength))
        cv2.line(rgb, p1, dp, (0, 0, 255), 3)
        plt.imshow(rgb)
        plt.show()
        #time.sleep(2)
        #plt.close()
        Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//background/yogurt/score:" + str(score) + str(
            time.strftime('  %Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ".png"
        cv2.imwrite(Img_Name, rgb_back)



        writedata.xytheta_wr(p1y,p1x,theta)

        Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//processed/yogurt/score:" + str(score)+str(time.strftime('  %Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ".png"
        cv2.imwrite(Img_Name, rgb)
        writedata.trainscore(score)
        print 'data saved successfully'
        print "start to next grasp"

        if symbol == "slippery":
            ur5.move_joints(cam_joint)
            #ur5.move_pose([-0.29997, 0.18618, 0.30,3.1415 ,0 ,0 ])
            rgb_back, depth_back = camera.get_data()
            gray_back = Grasp_Plan.rgb2gray(rgb_back)
            # ur5.move_joints(cam_joint)
            ROI_back = gray_back  # [140:460, 110:590]
            cv2.imshow('gray_back', ROI_back)
            cv2.waitKey(1000)


            ur5.soft_place([185, 402 ,0.18 , 90])
            #ur5.io_open()
            gripper.activate()

        if grasp_time == grasp_times or symbol == "drop":
            gripper.activate()
            break

        print "start to next grasp"
        #grasp_success = False









if __name__=='__main__':
    main()

    #create_result_path()







