#coding:utf-8
#!/usr/bin/env python
import cv2
import sys,os,pdb
from TakkTile_usb.TakkTile import TakkTile
from camera3 import Camera
from ur5 import UR5
from timer import Timer
from moment import GraspPlann
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from Bayesiangrasp import BayesianGraspNet, img_rot
from logger import Logger
from window_scan import Winscanning
from PIL import Image, ImageDraw
from ur5_manipulator_node import RobotiqCGripper, UR5_Gripper_Manipulator
from data.writedata import Data
from data.process import tac_metric
print ('import done')
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn.functional as F
import math
import torch
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

    # num , pressure, temprature = tact.getrawdata()
    # print pressure

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
    threshod=0.85
    successtimes=0
    times=0
    fail1times=0
    fail2times=0
    BayesianNet = BayesianGraspNet(model1_input='rgb', model2_input='rgb')

    while grasp_time <= grasp_times:


        torch.cuda.empty_cache()
        # move ur5 to camera position to capture background images
        ur5.move_joints(cam_joint)

        #recording time
        timer.tic()
        # get color image with grasped object
        rgb,depth=camera.get_data()
        # network prediction of state 1

        points_stage1 = BayesianNet.predict_stage1([150, 180, 450, 380], rgb)

        # pdb.set_trace()
        if points_stage1[0] == 'fail':
            fail1times+=1
            print 'stage1 failed, no objects detected'
            # continue
            if fail1times==20:
                print "no objects found"
                break

        else:

            # pdb.set_trace()
            tt1 = time.time()
            print 'stage2 random time is :', time.time() - tt1
            #pdb.set_trace()

            points_stage2, angles_stage2, actpatch = BayesianNet.predict_stage2(points_stage1, rgb, fail2times)
            #cv2.circle(actpatch, (int(points_stage2[0]), int(points_stage2[1])), 2, (255, 255, 0), 5)
            cv2.imshow('grasp_point', actpatch)
            q = cv2.waitKey(1000)
            print q
            cv2.destroyAllWindows()

            #pdb.set_trace()

            if points_stage2 == 'fail' :
                print 'stage2 failed, grasp again'
                # continue
                fail2times+=1
            else:
                print points_stage2, angles_stage2
                print 'stage 2 time is :', time.time() - tt1
                #print 'detection time is: %f' % (time.time() - t0)
                points_result = []  #
                points_angles = {}  # {index:[angle1,angle2,...]},  points_result[index] is (x,y)
                tt2 = time.time()

                print 'plot time is :', time.time() - tt2
                # pdb.set_trace()

                Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//datapath/test/visiononly/tennis/stage:" + str(
                    2) + str(time.strftime('  %Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ".png"
                cv2.imwrite(Img_Name, actpatch)

                #tactile information
                num, pressure, temprature = tact.getrawdata()
                pre_p = pressure
                print "Before grasping, the pressure is ", pre_p
                writedata.apdata(pre_p)

                # gripper.activate()
                # grasp the object
                print "start to grasp"
                # ur5.io_open()
                #gripper.activate()
                grasp_depth = depth[int(points_stage2[0]), int(points_stage2[1])]

                # start to grasp
                ur5.soft_grasp([int(points_stage2[0]), int(points_stage2[1]), grasp_depth, int(angles_stage2-90)])
                gripper.close()

                grasp_time = grasp_time + 1

                # tactile information`
                ur5.move_pose([-0.18312, -0.30185, 0.40340, 3.1415, 0, 0])
                time.sleep(1)

                num, pressure, temprature = tact.getrawdata()
                print "After grasping, the pressure is ", pressure
                later_p = pressure

                writedata.apdata(later_p)

                grasp_success = True
                print "grasp successfully"

                # shake the robot arm for 8s
                shake_point1 = [0, -0.350, 0.375, 3.1415, 0, 0]
                shake_point2 = [-0.310, -0.448, 0.450, 3.1415, 0, 0]
                shake_point3 = [-0.12866, -0.32689, 0.60290, 3.1415, 0, 0]
                shake_point4 = [-0.12866, -0.32689, 0.32651, 3.1415, 0, 0]

                shake_startpoint = np.array([45, -110, 110, -90, -90, 11]) * 3.1415 / 180.
                shake_joint1 = np.array([45, -110, 110, -90, -50, 11]) * 3.1415 / 180.
                shake_joint2 = np.array([45, -110, 110, -90, -136, 11]) * 3.1415 / 180.
                shakematrix = [[0] * 5 for _ in range(5)]
                shakematrix = np.array(shakematrix)
                shakematrix[0] = later_p

                i = 0
                while i < 2:
                    ur5.shake_pose(shake_point3)
                    num, pressure, temprature = tact.getrawdata()
                    print "After shaking_pose", i, "the pressure is ", pressure
                    pre_p = np.array(pre_p)
                    pressure = np.array(pressure)

                    # writedata.apdata(pressure)
                    shakematrix[i + 1] = pressure
                    ur5.shake_pose(shake_point4)
                    i = i + 1

                j = 0
                ur5.move_joints(shake_startpoint)
                # pdb.set_trace()
                while j < 2:
                    ur5.shake_joints(shake_joint1)
                    num, pressure, temprature = tact.getrawdata()
                    pre_p = np.array(pre_p)
                    pressure = np.array(pressure)
                    print "After shaking_joints", i, "the pressure is ", pressure
                    # writedata.apdata(pressure)
                    shakematrix[j + 3] = pressure
                    ur5.shake_joints(shake_joint2)
                    j = j + 1
                # ur5.shake(shake_point)
                # ur5.shake([0 ,-0.350 ,0.8 ,3.1415 ,0 ,0])

                # num, pressure, temprature = tact.getrawdata()
                # print "After shaking, the pressure is ", pressure
                # pressure = np.array(pressure)
                # writedata.apdata(pressure)



                print"The object is stable or slippery"
                # writedata.stable_result()

                ur5.move_joints(cam_joint)

                writedata.endonepro()


                writedata.shakedata(shakematrix)
                score = metric.score(shakematrix, 'slippery')

                writedata.writescore(score)
                print "The grasp score is:", score

                if score>threshod:
                    successtimes+=1
                times+=1

                writedata.testscore(score)
                print 'grasp successfully'
                print "start to next grasp"

                ur5.soft_place([185, 402, 0.18, 90])
                # ur5.Bayese_place([-0.469, -0.032, 0.18, 90]) #robotx,roboty ## m
                # ur5.io_open()
                gripper.activate()
                fail2times=0
                # p =subprocess.Popen("ps aux|grep xxuser|grep python|grep anaconda|awk '{print $2}'|xargs kill", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # p.wait()
                # if p.returncode != 0:
                #     print "Error."
                #     return -1
        # if times==20:
        #     pdb.set_trace()





if __name__=='__main__':
    main()

    #create_result_path()







