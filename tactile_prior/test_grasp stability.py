import cv2
import sys,os,pdb
import matplotlib.patches as patches
#pdb.set_trace()
#from soft_grasp_ur5.soft_grasp.softnet import SoftGraspNet, listMean, plotGrasp
from TakkTile_usb.TakkTile import TakkTile
#from camera3 import Camera
from ur5 import UR5
from timer import Timer
from moment import GraspPlann
from data.process import tac_metric
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
from data.writedata import Data

from logger import Logger
from window_scan import Winscanning
from PIL import Image, ImageDraw
#from ur5_manipulator_node import RobotiqCGripper, UR5_Gripper_Manipulator
print ('import done')

def main():


    num_pick=0  #calculate the grasping number
    no_object=0
    timer=Timer()
    tact = TakkTile()
    print tact.UIDs
    print tact.alive


    #camera=Camera()
    #class of image moment
    # mat= [[0]*5 for _ in range(5)]
    # num , pressure, temprature = tact.getrawdata()
    # print type(pressure),pressure
    # pre1=np.array(pressure)
    # print type(pre1), pre1
    #
    # num , pressure, temprature = tact.getrawdata()
    # pre2=np.array(pressure)
    # mat=np.array(mat)
    # mat[0]=pre2-pre1
    # print mat
    # pdb.set_trace()


    #pdb.set_trace()
    Grasp_Plan=GraspPlann()
    winscan = Winscanning()
    writedata = Data()
    metric = tac_metric()


    #pdb.set_trace()
    #gripper = RobotiqCGripper()
    #gripper.activate()
    #gripper.close()
    #pdb.set_trace()
    #gripper.close()

    # configuration of UR5
    ur5 = UR5(tcp_host_ip='192.168.1.2', tcp_port=30003)
    cam_joint=np.array([-2.38,-111,111,-90,-90,161])*3.1415/180.
    # [60,-120,112,-80,-90,200]
    ur5.move_joints(cam_joint)

    grasp_success = False

    while True:

        while not grasp_success:
            time.sleep(0.5)
            #gripper.close()
            #pdb.set_trace()
            # tactile information
            # num, pressure, temprature = tact.getrawdata()
            # pre_p = pressure
            # print "Before grasping, the pressure is ", pre_p
            #gripper.activate()
            # grasp the object
            ur5.io_open()
            # tactile information

            num, pressure, temprature = tact.getrawdata()
            pre_p = pressure
            print "Before grasping, the pressure is ", pre_p
            writedata.apdata(pre_p)

            #ur5.move_pose([-0.03373, -0.43011, 0.136, 3.1415, 0, 0])
            ur5.move_pose([-0.03373, -0.43011, 0.15288, 3.1415, 0, 0])

            #ur5.soft_grasp([p1y, p1x, 0.132, wholedeg])
            print "start to grasp"
            ur5.io_close()



            # tactile information`
            ur5.move_pose([-0.18312, -0.30185, 0.40340, 3.1415, 0, 0])
            time.sleep(2)

            num, pressure, temprature = tact.getrawdata()
            print "After grasping, the pressure is ", pressure
            later_p = pressure
            writedata.apdata(later_p)
            # pressure_suc = abs(later_p[0] - pre_p[0]) + abs(later_p[1] - pre_p[1]) + abs(
            #     later_p[2] - pre_p[2]) + abs(later_p[3] - pre_p[3]) + abs(later_p[4] - pre_p[4])
            max_dif = max(abs(later_p[0] - pre_p[0]), abs(later_p[1] - pre_p[1]), abs(later_p[2] - pre_p[2]),
                          abs(later_p[3] - pre_p[3]), abs(later_p[4] - pre_p[4]))
            str = raw_input("please input, 1-success, 2-fail:")
            if str == '1':
                grasp_success = True
                print "grasp successfully"
                writedata.suc_result()
            if str == '2':
                grasp_success = False
                print "grasp is failure"
                symbol = "failure"
                writedata.fai_result()
                writedata.endonepro()

        #pdb.set_trace()
        # shake the robot arm for 8s
        shake_point1 = [0, -0.350, 0.375, 3.1415, 0, 0]
        shake_point2 = [-0.310, -0.448, 0.450, 3.1415, 0, 0]
        shake_point3 = [-0.12866,-0.32689, 0.60290, 3.1415,0,0]
        shake_point4 = [-0.12866, -0.32689, 0.32651, 3.1415, 0, 0]

        shake_startpoint = np.array([45, -110, 110, -90, -90, 116]) * 3.1415 / 180.
        shake_joint1 = np.array([45, -110, 110, -90, -50, 116]) * 3.1415 / 180.
        shake_joint2 = np.array([45, -110, 110, -90, -136, 116]) * 3.1415 / 180.
        shakematrix = [[0]*5 for _ in range(5)]
        shakematrix = np.array(shakematrix)
        i = 0
        while i < 2:
            ur5.shake_pose(shake_point3)
            num, pressure, temprature = tact.getrawdata()
            print "After shaking_pose",i , "the pressure is ", pressure
            pre_p=np.array(pre_p)
            pressure=np.array(pressure)

            writedata.apdata(pressure)
            shakematrix[i]=pressure
            ur5.shake_pose(shake_point4)
            i = i + 1

        j = 0
        ur5.move_joints(shake_startpoint)
        #pdb.set_trace()
        while j < 2:
            ur5.shake_joints(shake_joint1)
            num, pressure, temprature = tact.getrawdata()
            pre_p=np.array(pre_p)
            pressure=np.array(pressure)
            print "After shaking_joints", i, "the pressure is ", pressure
            writedata.apdata(pressure)
            shakematrix[j+2]=pressure
            ur5.shake_joints(shake_joint2)
            j = j+1
        # ur5.shake(shake_point)
        # ur5.shake([0 ,-0.350 ,0.8 ,3.1415 ,0 ,0])

        num, pressure, temprature = tact.getrawdata()
        print "After shaking, the pressure is ", pressure
        pressure = np.array(pressure)
        writedata.apdata(pressure)
        shakematrix[4]=pressure
        third_p = pressure
        # pressure_dif = abs(later_p[0] - third_p[0]) + abs(later_p[1] - third_p[1]) + abs(
        #     later_p[2] - third_p[2]) + abs(later_p[3] - third_p[3]) + abs(later_p[4] - third_p[4])
        # shake_dif = abs(pre_p[0] - third_p[0]) + abs(pre_p[1] - third_p[1]) + abs(
        #     pre_p[2] - third_p[2]) + abs(pre_p[3] - third_p[3]) + abs(pre_p[4] - third_p[4])

        str = raw_input("if stable, please input 3. if drop,please input 4:")
        timer = Timer()
        if str =='4':
            print"The object drops off"
            writedata.drop_result()
            symbol = "drop"
        elif str == '3':
            print"The object is stable or slippery"
            writedata.stable_result()
            symbol = "slippery"
        # if pressure_dif <= 500 & pressure_dif >= 50 or shake_dif <250:
        #     print "The object drops off"
        # elif pressure_dif >= 500:
        #     print "The object is slippery"
        # else:
        #     print "The object is stable"

        ur5.move_joints(cam_joint)
        grasp_success = False
        #pdb.set_trace()

        ur5.move_pose([-0.15267, -0.43011, 0.336, 3.1415, 0, 0])
        ur5.io_open()
        writedata.endonepro()
        writedata.shakedata(shakematrix)
        if symbol == "failure":
            score = 0
        else:
            score = metric.score(shakematrix,symbol)
        writedata.writescore(score)
        print "The grasp score is:", score

        print "start to next grasp"
        grasp_success = False




if __name__=='__main__':
    main()

    #create_result_path()
