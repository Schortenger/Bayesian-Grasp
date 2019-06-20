#!/usr/bin/env python
import sys
import rospy
import random
import math
#import tf
import copy
import sys,os,pdb
from moveit_commander import RobotCommander,MoveGroupCommander, PlanningSceneInterface, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import PoseStamped
import moveit_msgs.msg
import geometry_msgs.msg
from tf.transformations import *
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
from TakkTile_usb.TakkTile import TakkTile

#from grasp_msgs_srvs.srv import *
#from grasp_msgs_srvs.msg import *


import numpy as np

import roslib
#roslib.load_manifest('grasp_motion_planning')

import rospy
from robotiq_c_model_control.msg import CModel_robot_output as outputMsg
from robotiq_c_model_control.msg import CModel_robot_input as inputMsg

print "All import are done"

class RobotiqCGripper(object):
    def __init__(self):
        rospy.init_node('RobotiqCGripper', anonymous=True)
        self.cur_status = None
        self.status_sub = rospy.Subscriber('CModelRobotInput', inputMsg,
                                           self._status_cb)
        self.cmd_pub = rospy.Publisher('CModelRobotOutput', outputMsg,queue_size=3)
        self.wait_for_connection()
        self.reset()
        self.activate()
        #self.init_gripper()
        rospy.loginfo('Gripper is on')
        print "Gripper initialized"

    def _status_cb(self, msg):
        self.cur_status = msg

    def wait_for_connection(self, timeout=-1):
        rospy.sleep(0.1)
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (timeout >= 0. and rospy.get_time() - start_time > timeout):
                return False
            if self.cur_status is not None:
                print "***********Gripper is connected.***********"
                return True
            r.sleep()
        return False

    def is_ready(self):
        return self.cur_status.gSTA == 3 and self.cur_status.gACT == 1

    def is_reset(self):
        return self.cur_status.gSTA == 0 or self.cur_status.gACT == 0

    def is_moving(self):
        return self.cur_status.gGTO == 1 and self.cur_status.gOBJ == 0

    def is_stopped(self):
        return self.cur_status.gOBJ != 0

    def object_detected(self):
        return self.cur_status.gOBJ == 1 or self.cur_status.gOBJ == 2

    def get_fault_status(self):
        return self.cur_status.gFLT

    def get_pos(self):
        po = self.cur_status.gPO
        return np.clip(0.087/(13.-230.)*(po-230.), 0, 0.087)

    def get_req_pos(self):
        pr = self.cur_status.gPR
        return np.clip(0.087/(13.-230.)*(pr-230.), 0, 0.087)

    def is_closed(self):
        return self.cur_status.gPO >= 230

    def is_opened(self):
        return self.cur_status.gPO <= 13

    # in mA
    def get_current(self):
        return self.cur_status.gCU * 0.1

    # if timeout is negative, wait forever
    def wait_until_stopped(self, timeout=-1):
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
                return False
            if self.is_stopped():
                return True
            r.sleep()
        return False

    def wait_until_moving(self, timeout=-1):
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
                return False
            if not self.is_stopped():
                return True
            r.sleep()
        return False

    def reset(self):
        cmd = outputMsg()
        cmd.rACT = 0
        self.cmd_pub.publish(cmd)

    def activate(self, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = 0
        cmd.rSP = 255
        cmd.rFR = 150
        self.cmd_pub.publish(cmd)
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if timeout >= 0. and rospy.get_time() - start_time > timeout:
                return False
            if self.is_ready():
                return True
            print 'is_read',self.is_ready()
            r.sleep()
        return False

    def auto_release(self):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rATR = 1
        self.cmd_pub.publish(cmd)

    ##
    # Goto position with desired force and velocity
    # @param pos Gripper width in meters. [0, 0.087]
    # @param vel Gripper speed in m/s. [0.013, 0.100]
    # @param force Gripper force in N. [30, 100] (not precise)
    def goto(self, pos, vel, force, block=False, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = int(np.clip((13.-255.)/0.087 * pos + 255., 0, 255))
        cmd.rSP = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd.rFR = int(np.clip(255./(100.-30.) * (force-30.), 0, 255))
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)
        if block:
            if not self.wait_until_moving(timeout):
                return False
            return self.wait_until_stopped(timeout)
        return True

    def gotoclose(self, pos, vel, force, block=False, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        #cmd.rPR = int(np.clip((13.-230.)/0.087 * pos + 230., 0, 255))
        #cmd.rPR = 108       #goblet      #0-255,control the degree of the gripper closed
        cmd.rPR = 245
        cmd.rSP = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd.rFR = force
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)

        if block:
            if not self.wait_until_moving(timeout):
                return False
            return self.wait_until_stopped(timeout)
        return True

    #increase rPR gradually
    def gradu_close(self,pos,rPR, vel, force=1, block=False, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1

        #cmd.rPR = int(np.clip((13.-230.)/0.087 * pos + 230., 0, 255))
        cmd.rPR = rPR       #goblet      #0-255,control the degree of the gripper closed
        cmd.rSP = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd.rFR = 1
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)
        if block:
            if not self.wait_until_moving(timeout):
                return False
            return self.wait_until_stopped(timeout)
        return True


    def stop(self, block=False, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 0
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)
        if block:
            return self.wait_until_stopped(timeout)
        return True

    def open(self, vel=0.1, force=50, block=False, timeout=-1):
        if self.is_opened():
            return True
        return self.goto(1.0, vel, force, block=block, timeout=timeout)

    def close(self, vel=0.5, force=1, block=False, timeout=-1): #force =25
        if self.is_closed():
            print 'is_closed'
            return True
        print 'go to closed'
        return self.gotoclose(-1.0, vel, force, block=block, timeout=timeout)
        #return self.goto(-1.0, vel, force, block=block, timeout=timeout)

    def graclose(self, rPR, vel=0.1, force=1, block=False, timeout=-1):
        if self.is_closed():
            print 'is_closed'
            return True
        #print 'go to closed'
        #return self.gotoclose(-1.0, vel, force, block=block, timeout=timeout)
        return self.gradu_close(-1.0,rPR, vel, force, block=block, timeout=timeout)

    def init_gripper(self):
        if self.is_reset():
            print "*****Gripper is intialized.*********",self.is_reset()
            self.reset()
            self.activate()
            self.open(block=True)
            self.close(block=True)
        else:
            print 'self.is_reset()',self.is_reset()
            self.reset()
            print 'self.is_reset()', self.is_reset()
            self.activate()
            print 'self.is_reset()', self.is_reset()
            self.open(block=True)
            self.close(block=True)
            print 'closed'


    def gripper_callback(self,req):
        cmd=req.cmd
        pos=cmd[0]
        vel=cmd[1]
        force=cmd[2]
        block=req.block
        res=GripperCommandResponse()
        goal=self.goto(pos,vel,force,block)
        if goal:
            res.success.data=True
            return res
        else:
            res.success.data=False
            return res






class UR5_Gripper_Manipulator(object):
    def __init__(self, scene=None, manip_name="manipulator", eef_name="endeffector"):
        self.robot = RobotCommander()
        self.group = MoveGroupCommander(manip_name)
        self.eef = MoveGroupCommander(eef_name)
        self.scene = scene

        # self.gripper = RobotiqCGripper()
        # self.gripper.wait_for_connection()
        # self.init_gripper()
        # print "Gripper is OK."



    def check_joints(self,joints):
        goal_achieved=False
        time=rospy.get_rostime().secs
        while(not goal_achieved):
            current_joints=self.group.get_current_joint_values()
            avg_joint_diff=0
            for i in range(6):
                avg_joint_diff+=abs(current_joints[i]-joints[i])
            avg_joint_diff=avg_joint_diff/6.0
            if avg_joint_diff<0.02:
                goal_achieved=True
            duration=rospy.get_rostime().secs-time
            if duration>8:
                rospy.loginfo("Time limit exceeded to reach desired position")
                if avg_joint_diff<0.05:
                    goal_achieved=True
                else:
                    goal_achieved=False
                break
        if goal_achieved:
            return True
        else:
            return False

    def move_joint(self,joints,vel_factor=0.1):
        self.group.clear_pose_targets()
        if type(joints) is not list:
            joints_list=[]
            for i in range(6):
                joints_list.append(joints[i])
        else:
            joints_list=joints

        self.group.set_joint_value_target(joints_list)
        self.group.set_max_velocity_scaling_factor(vel_factor)
        self.group.set_max_acceleration_scaling_factor(0.1)
        plan = self.group.plan()
        self.group.execute(plan)
        goal_achieved=self.check_joints(joints_list)
        if goal_achieved:
            print "Reached Desired Joint"
            return True
        else:
            return False

    # def matrix_quaternion(self,matrix):
    #     xaxis, yaxis, zaxis = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    #     if type(matrix) is list:
    #         assert len(matrix)==3
    #         # 'sxyz'=rot(z)*rot(y)*rot(x)
    #         qx=quaternion_about_axis(matrix[0], xaxis)
    #         qy=quaternion_about_axis(matrix[1], yaxis)
    #         qz=quaternion_about_axis(matrix[z], zaxis)
    #         q = quaternion_multiply(qz, qy)
    #         q = quaternion_multiply(q, qx)
    #         return q
    #     elif type(matrix) is numpy.ndarray:
    #         al, be, ga = euler_from_matrix(matrix, 'sxyz')
    #         qx = quaternion_about_axis(al, xaxis)
    #         qy = quaternion_about_axis(be, yaxis)
    #         qz = quaternion_about_axis(ga, zaxis)
    #         q = quaternion_multiply(qz, qy)
    #         q = quaternion_multiply(q, qx)
    #         return q
    #     else:
    #         return matrix




    def move_pose(self,pose_list,vel_factor=0.1):
        pose_target = geometry_msgs.msg.Pose()
        if type(pose_list) is list:
            pose_target.position.x=pose_list[0]
            pose_target.position.y =pose_list[1]
            pose_target.position.z =pose_list[2]
            q=quaternion_from_euler(pose_list[3],pose_list[4],pose_list[5])
            pose_target.orientation = Quaternion(*q)

        self.group.set_pose_target(pose_target)
        print 'pose_target.',pose_target
        self.group.set_max_velocity_scaling_factor(vel_factor)
        self.group.set_max_acceleration_scaling_factor(0.1)
        plan = self.group.plan()
        self.group.execute(plan)
        #print 'plane',plan
        print 'point[-1]:',plan.joint_trajectory.points
        target_joints=plan.joint_trajectory.points[-1].positions
        goal_achieved = self.check_joints(target_joints)
        if goal_achieved:
            print "Reached Desired Joint"
            return True
        else:
            return False

    # poses_seq,[pose1,pose2,...]
    # pose=[x,y,z,r,p,y,vel],'sxyz'
    def move_poses_sequences(self,poses_seq):
        assert type(poses_seq) is list
        num=len(poses_seq)
        print "**** There are %d poses to move.****" % num
        for i in range(num):
            if len(poses_seq[i])==6:
                one_goal=self.move_pose(poses_seq[i])
            elif len(poses_seq[i])==7:
                one_goal=self.move_pose(poses_seq[i][:6], poses_seq[i][6])
            if one_goal:
                continue
            else:
                return False
        return True

    # poses_seq,[joint1,joint2,...]
    # pose=[j1,j2,j3,j4,j5,j6,vel]
    def move_joints_sequences(self,joints_seq):
        assert type(joints_seq) is list
        num=len(joints_seq)
        print "**** There are %d joints to move.****" % num
        for i in range(num):
            if len(joints_seq[i])==6:
                one_goal=self.move_joint(joints_seq[i])
            elif len(joints_seq[i])==7:
                one_goal=self.move_joint(joints_seq[i][:6], joints_seq[i][6])
            if one_goal:
                continue
            else:
                return False
        return True

    def move_joint_callback(self,req):
        joint=req.goal.data
        assert len(joint)==6
        vel=req.vel
        goal=self.move_joint(joint,vel)
        res=UR5JointGoalResponse()
        if goal:
            res.success.data=True
            return res
        else:
            res.success.data = False
            return res

    def move_joints_seq_callback(self,req):
        num=len(req.goal)
        joints_seq=[]
        vel=req.vel
        for i in range(num):
            joints_seq.append(req.goal[i].data+(vel[i],))

        goal=self.move_joints_sequences(joints_seq)
        res = UR5JointSeqGoalResponse()
        if goal:
            res.success.data=True
            return res
        else:
            res.success.data = False
            return res

    def move_pose_callback(self,req):
        pose=req.goal.data
        assert len(pose)==6
        vel = req.vel
        goal = self.move_pose(joint, vel)
        res = UR5PositionGoalResponse()
        if goal:
            res.success.data = True
            return res
        else:
            res.success.data = False
            return res

    def move_poses_seq_callback(self,req):
        num=len(req.goal)
        poses_seq=[]
        vel=req.vel
        for i in range(num):
            poses_seq.append(req.goal[i].data+[vel[i]])

        goal=self.move_poses_sequences(poses_seq)
        res = UR5PositionSeqGoalResponse()
        if goal:
            res.success.data=True
            return res
        else:
            res.success.data = False
            return res

    def plane_grasp_callback(self,req):
        res=PlaneGraspPoseResponse()
        #self.group.get_current_pose(self.get)
        position=req.position
        grasp_prepoint=req.grasp_prepoint
        vel=req.vel
        angle,x,y,z=position[0],position[1],position[2],position[3]
        p1,p2=grasp_prepoint[0],grasp_prepoint[1]

        # R1_angle=np.zeros([4,4])
        # R2_noangle = np.zeros([4, 4])
        #
        # R2_noangle[3][3]=1
        # R2_noangle[0][0], R2_noangle[1][0], R2_noangle[2][0] = 0, 0,-1  # x-axis
        # R2_noangle[0][1], R2_noangle[1][1], R2_noangle[2][1] = 1, 0, 0  # y-axis
        # R2_noangle[0][2], R2_noangle[1][2], R2_noangle[2][2] = 0, -1, 0 # z-axis
        # euler_noangle=euler_from_matrix(R2_noangle,'sxyz')
        #
        # R1_angle[3][3] = 1
        # R1_angle[0][0], R1_angle[1][0], R1_angle[2][0] = 0, 0, -1  # x-axis
        # R1_angle[0][1], R1_angle[1][1], R1_angle[2][1] = math.cos(angle), -math.sin(angle), 0   # y-axis
        # R1_angle[0][2], R1_angle[1][2], R1_angle[2][2] = -math.sin(angle), -math.cos(angle), 0   # z-axis
        # euler_angle = euler_from_matrix(R1_angle, 'sxyz')

        # euler_noangle = [-2.499, 1.569, -2.182]
        euler_noangle = [2.429, 1.568, 0.859]   # need to change

        xaxis=(1,0,0)
        Rx = rotation_matrix(angle, xaxis)
        Re = euler_matrix(euler_noangle[0], euler_noangle[1],euler_noangle[2], 'sxyz')
        R1_angle=np.dot(Re,Rx)
        euler_angle = euler_from_matrix(R1_angle, 'sxyz')

        pose0=[x,y,z,euler_angle[0],euler_angle[1],euler_angle[2]]
        pose1=[x,y,z+p1,euler_angle[0],euler_angle[1],euler_angle[2]]
        pose2=[x,y,z+p2,euler_noangle[0],euler_noangle[1],euler_noangle[2]]

        phase1=[pose2+[vel],pose1+[vel*2.0]]
        print 'phase1:',phase1
        phase2=[pose0+[vel*0.8]]
        phase3=[pose1+[vel],pose2+[vel*2.0]]
        #Phase 1: from Home to P1
        if not self.move_poses_sequences(phase1):
            print 'move failure'
            res.success.data=False
            return res
        # Open the gripper
        if not self.call_gripper_service([1,0.1,100,True]):
            res.success.data = False
            return res
        #Phase2: from p1 to p0---grasp position
        if not self.move_poses_sequences(phase2):
            res.success.data = False
            return res
        # Close the gripper
        if not self.call_gripper_service([-1, 0.1, 100, True]):
            res.success.data = False
            return res
        #Phase 3: from p0 to p2
        if not self.move_poses_sequences(phase3):
            res.success.data = False
            return res
        res.success.data = True
        return res


    def call_gripper_service(self,cmd):
        rospy.wait_for_service('/grasp/motion_planner/robotiq_gripper')
        gripper_request = rospy.ServiceProxy('/grasp/motion_planner/robotiq_gripper', GripperCommand)
        gripper_req = GripperCommandRequest()
        gripper_req.cmd = [cmd[0], cmd[1], cmd[2]]
        gripper_req.block.data = cmd[3]
        gripper_res=gripper_request(gripper_req)
        return gripper_res.success.data









if __name__=='__main__':
    # initialize the ROS node
    rospy.init_node('Grasp_motion_planner_Node')
    ur5=UR5_Gripper_Manipulator()
    gripper=RobotiqCGripper()
    rospy.loginfo('**********Calculation of Grasp Planning Server Initialized**********')

    # move to a desired target joint postion
    #service1 = rospy.Service('/grasp/motion_planner/ur5_joint_goal', UR5JointGoal, ur5.move_joint_callback)

    # move to a desired target pose posetion
    #service2 = rospy.Service('/grasp/motion_planner/ur5_pose_goal', UR5PositionGoal, ur5.move_pose_callback)

    # move to a joint sequence
    #service3 = rospy.Service('/grasp/motion_planner/ur5_joints_sequences', UR5JointSeqGoal, ur5.move_joints_seq_callback)

    # move to a pose sequence
    #service4 = rospy.Service('/grasp/motion_planner/ur5_poses_sequences', UR5PositionSeqGoal, ur5.move_poses_seq_callback)

    # gripper control
    service5 = rospy.Service('/grasp/motion_planner/robotiq_gripper', GripperCommand,
                             gripper.gripper_callback)

    # plane grasp, from home->p2->p1->open gripper->p0->close gripper->p1->p2
    #service6 = rospy.Service('/grasp/motion_planner/plane_grasp_execute', PlaneGraspPose,
                             #ur5.plane_grasp_callback)


    rospy.spin()















