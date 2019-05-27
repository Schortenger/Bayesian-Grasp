import socket
import select
import struct
import time
import os,sys
import numpy as np
import math
# sys.path.insert(0,'/opt/ros/indigo/lib/python2.7/dist-packages/')
# from tf.transformations import euler_matrix,rotation_from_matrix
import pdb
# pdb.set_trace()
min_x,min_y,max_x,max_y=96,48,544,400
depth_bin=0.6655



def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
_NEXT_AXIS = [1, 2, 0, 1]
def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M




class UR5(object):
    def __init__(self,tcp_host_ip="192.168.1.2",tcp_port=30003):
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port

        # Default joint speed configuration
        self.joint_acc = 1  # Safe: 1.4
        self.joint_vel = 1  # Safe: 1.05

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 0.3 #1.6  # Safe: 0.5,1.2
        self.tool_vel = 0.3  # Safe: 1.8
        self.quick_acc = 3 #4.2
        self.quick_vel = 1.8


        # self.orientation=[0.0,3.1415,0.0]
        self.orientation = None

        self.tool_pose_tolerance = [0.009, 0.009, 0.009, 0.01, 0.01, 0.01]

        self.info=['current_joints','current_poses']
        self.box=['left_box','right_box']
        self.boxes=[[],[]]
        self.z_limit=[0.141,0.473]
        self.x_limit=[]
        self.x,self.y,self.z=None,None,None
        #self.set_bin_position()
    def set_bin_position(self,bin,bin_depth):
        right_px, right_py=bin[0]-50,bin[1]
        left_px, left_py=bin[2]+50,bin[3]
        left_x,left_y,z=self.frametrans(left_px,left_py,bin_depth)
        right_x,right_y,z=self.frametrans(right_px,right_py,bin_depth)
        print('....z claculate:', z + 0.21)
        z=0.5
        left_box_x,left_box_y,left_box_z=left_x+0.15,left_y,z-0.05
        right_box_x,right_box_y,right_box_z=right_x-0.12,right_y,z-0.05
        rx,ry,rz=self.orientation
        print('....left_x,left_box_x', left_x, left_box_x)
        self.boxes[0].append([left_x,left_y,z,rx,ry,rz])
        self.boxes[0].append([left_box_x,left_box_y,left_box_z, rx, ry, rz])
        self.boxes[1].append([right_x,right_y,z, rx, ry, rz])
        self.boxes[1].append([right_box_x,right_box_y,right_box_z, rx, ry, rz])

    def get_robot_state(self,info):
        assert (info in self.info), ' Can only get info: '+str(self.info)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #s.settimeout(10)
            s.connect((self.tcp_host_ip , self.tcp_port))
            time.sleep(0.2)
            if info=='current_joints':
                packet = s.recv(252)
                joint=[]
                for i in range(6):
                    packet_8 = s.recv(8)
                    packet_8 = packet_8.encode("hex")  # convert the data from \x hex notation to plain hex
                    x = str(packet_8)
                    x = struct.unpack('!d', packet_8.decode('hex'))[0]
                    joint.append(x)
                return joint
            elif info=='current_poses':
                packet = s.recv(444)
                pose = []
                for i in range(6):
                    packet_8 = s.recv(8)
                    packet_8 = packet_8.encode("hex")  # convert the data from \x hex notation to plain hex
                    x = str(packet_8)
                    x = struct.unpack('!d', packet_8.decode('hex'))[0]
                    pose.append(x)
        except socket.error as socketerror:
            print("Error: ", socketerror)
        return pose

    def move_joints(self, joint_configuration):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (1, 1)
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        while True:
            robot_state=self.get_robot_state('current_joints')
            if all([np.abs(robot_state[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
                break
        return True

    def move_pose(self, pose):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
            pose[0],pose[1], pose[2], pose[3], pose[4], pose[5], self.tool_acc, self.tool_vel)

        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (
        #     x, y, 0.45, orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)

        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(0.5)
        self.tcp_socket.close()
        target_pose = [pose[0],pose[1],pose[2]]
        t = time.time()
        while True:
            if time.time() - t > 5:
                print('... fail to move')
            robot_state = self.get_robot_state('current_poses')
            # print robot_state
            if all([np.abs(robot_state[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        return True

    def shake_pose(self, pose):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
            pose[0],pose[1], pose[2], pose[3], pose[4], pose[5], self.quick_acc, self.quick_vel)
        #Rx= 3.14, Ry =0, Rz = 0 means the effctor is vertical
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (
        #     x, y, 0.45, orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)

        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(0.5)
        self.tcp_socket.close()
        target_pose = [pose[0],pose[1],pose[2]]
        t = time.time()
        while True:
            if time.time() - t > 5:
                print('... robot is shaking')
            robot_state = self.get_robot_state('current_poses')
            # print robot_state
            if all([np.abs(robot_state[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        return True

    def shake_joints(self, joint_configuration):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (5, 3)
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        while True:
            robot_state=self.get_robot_state('current_joints')
            if all([np.abs(robot_state[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
                break
        return True


    def normaltrans(self,x_pixel,y_pixel):

        print "x_pixel,y_pixel", x_pixel,y_pixel

        lfx_pixel = 146
        lfy_pixel = 329
        rgx_pixel = 606
        rgy_pixel = 4
        lfx_robot = 196.9
        lfy_robot = -340.43
        rgx_robot = -443.16
        rgy_robot = -649.01

        x_robot = (x_pixel - lfx_pixel)*(rgx_robot-lfx_robot)/(rgx_pixel-lfx_pixel)+lfx_robot
        y_robot = (y_pixel - lfy_pixel)*(rgy_robot-lfy_robot)/(rgy_pixel-lfy_pixel)+lfy_robot

        return x_robot, y_robot


    def frametrans(self,x_pixel,y_pixel,cam_depth):
        '''
        camera pixel to robot x y z
        :param x_pixel:
        :param y_pixel:
        :param cam_depth:
        :return:
        '''

        # inner_matrix = np.array([[613.43304529, 0., 326.33055684],
        #                  [0., 614.89480019, 240.43265312],
        #                  [0., 0., 1.]])
        # outer_matrix_R = np.array([[9.89799200e-01, -1.06066399e-02, 1.42074074e-01],
        #                    [1.35889939e-01, -2.29265172e-01, -9.63831627e-01],
        #                    [4.27956520e-02, 9.73306211e-01, -2.25485148e-01]])
        # outer_matrix_T = np.array([4.47532121e+02, 9.61022874e+01, 5.39654265e+02])

        inner_matrix = np.array([[616.734,   0.        , 322.861],
                                 [  0.        , 616.851 , 234.728],
                                 [  0.        ,   0.        ,   1.        ]])
        outer_matrix_R = np.array([[ -0.99453061, 0.10236743, -0.02073115],
                                   [0.10018636, 0.99109774 ,  0.08768108],
                                   [0.02952228, 0.085124548, -0.99593285]])
        outer_matrix_T = np.array([0.01904794, -0.61460924, 0.69458614])

        # R_z=rotation_matrix(math.pi,(0, 0, 1))
        # outer_matrix_R=np.dot(R_z,outer_matrix_R)
        # outer_matrix_T=np.dot(R_z,outer_matrix_T.T).T
        inv_inner_matrix = np.linalg.inv(inner_matrix)
        pixels=np.array([x_pixel,y_pixel,1])*cam_depth
        trans=np.dot(inv_inner_matrix,pixels)
        trans=np.dot(outer_matrix_R,trans)+outer_matrix_T
        #trans=trans+np.array([0,0,0.20])   # CHANGE the value to compensate the tool position
        x,y,z=trans[0],trans[1],trans[2]
        return [x,y,z]


    def ur5_euler_rotation(self,rpy):
        matrix = euler_matrix(rpy[0],rpy[1],rpy[2],axes='sxyz')
        angle, direction, point = rotation_from_matrix(matrix)
        return angle*direction

    def soft_grasp(self,position): # [px,py,z,theta], m, deg
        x,y,z=self.frametrans(position[0], position[1], position[2])
        print 'x_pixel, y_pixel:', position[0],position[1]
        print 'x, y:', x, y
        #x, y = self.normaltrans(position[0], position[1])
        #tool_offset = [ 0.010, -0.039, 0.325]
        tool_offset = [0, 0, 0.325]

        print 'theta', position[3]
        orientataion = [3.1415, 0, (-position[3])/ 180. * 3.1415]
        T_target = euler_matrix(orientataion[0], orientataion[1], orientataion[2], axes='sxyz')
        T_target[:3,3] = np.array([x,y,z])
        T_tool = np.identity(4)
        T_tool[:3,3] = np.array(tool_offset)
        T_tool = np.linalg.inv(T_tool)
        T = np.dot(T_target, T_tool)
        angle, direction, point = rotation_from_matrix(T)
        rotAxis,trans = angle * direction,T[:3,3]
        x,y,z = trans
        z = 0.14160

        # print rpy angle, not rotaxis
        print('Executing: grasp at (%f, %f, %f,%f, %f, %f)' % (x, y, z,
                                                              orientataion[0],orientataion[1],orientataion[2]))


        # self.x,self.y,seqlf.z=x,y,z
        # print('Executing: grasp at (%f, %f, %f)' % (x, y, z ))
        #pdb.set_trace()
        # Compute tool orientation from heightmap rotation angle
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

        orientataion = self.ur5_euler_rotation(orientataion)
        tcp_command = "def process():\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
            x, y, z+0.2, rotAxis[0], rotAxis[1], rotAxis[2],self.tool_acc, self.tool_vel)  # r = 0.09
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.02, orientataion[0], orientataion[1], orientataion[2], self.tool_acc,self.tool_vel)
        tcp_command += " set_digital_out(4,True)\n"
        tcp_command += " set_digital_out(5,True)\n"
        tcp_command += " sleep(0.8)\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            x, y, z, rotAxis[0], rotAxis[1], rotAxis[2], self.tool_acc, self.tool_vel)  # r = 0.09
        tcp_command += " set_digital_out(5,False)\n"
        tcp_command += " sleep(0.8)\n"
        # tcp_command += " set_digital_out(4,False)\n"
        # tcp_command += " set_digital_out(5,False)\n"
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.01, orientataion[0], orientataion[1], orientataion[2], self.tool_acc*0.5, self.tool_vel*0.5)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(0.5)
        self.tcp_socket.close()
        #pdb.set_trace()
        target_pose=[x, y, z]
        t=time.time()
        while True:
            if time.time()-t>5:
                print('... ka zai grasp zhe li')
            robot_state=self.get_robot_state('current_poses')
            if all([np.abs(robot_state[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        return True


    def io_close(self): # [px,py,z,theta,d], m, deg

        # Compute tool orientation from heightmap rotation angle
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

        tcp_command = "def process():\n"

        tcp_command += " set_digital_out(1,True)\n"

        tcp_command += " sleep(0.8)\n"

        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(0.5)
        self.tcp_socket.close()

        return True

    def io_open(self): # [px,py,z,theta,d], m, deg

        # Compute tool orientation from heightmap rotation angle
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

        tcp_command = "def process():\n"

        tcp_command += " set_digital_out(1,False)\n"

        tcp_command += " sleep(0.8)\n"


        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(0.5)
        self.tcp_socket.close()

        return True



    def soft_place(self,position): # [px,py,z,theta,d], m, deg
        x,y,z=self.frametrans(position[0], position[1], position[2])
        #x, y = self.normaltrans(position[0], position[1])
        z = 0.20
        self.x,self.y,self.z=x,y,z
        print('Executing: place at (%f, %f, %f)' % (x,y,z+0.1))
        # Compute tool orientation from heightmap rotation angle
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        orientataion=[3.1415, 0, -position[3]/180.*3.1415]
        orientataion = self.ur5_euler_rotation(orientataion)
        tcp_command = "def process():\n"
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.1, orientataion[0], orientataion[1], orientataion[2],self.tool_acc, self.tool_vel)  # r = 0.09
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            x, y, z + 0.03, orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.02, orientataion[0], orientataion[1], orientataion[2], self.tool_acc,self.tool_vel)
        # tcp_command += " set_digital_out(4,True)\n"
        # tcp_command += " set_digital_out(5,True)\n"
        tcp_command += " set_digital_out(4,True)\n"
        tcp_command += " set_digital_out(5,True)\n"
        tcp_command += " sleep(0.5)\n"
        tcp_command += " set_digital_out(4,False)\n"
        tcp_command += " set_digital_out(5,False)\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            x, y, z + 0.05, orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)
        # tcp_command += " sleep(0.5)\n"
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.01, orientataion[0], orientataion[1], orientataion[2], self.tool_acc*0.5, self.tool_vel*0.5)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(1)
        self.tcp_socket.close()
        target_pose=[x, y, z+0.05]
        t=time.time()
        while True:
            if time.time()-t>5:
                print('... unsuccessful place')
            robot_state=self.get_robot_state('current_poses')
            if all([np.abs(robot_state[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        return True

    def Bayese_place(self, position):  # [robotx,roboty,z,theta,d], m, deg
        #x, y, z = self.frametrans(position[0], position[1], position[2])
        # x, y = self.normaltrans(position[0], position[1])
        z = 0.20
        self.x, self.y, self.z = position[0], position[1], z
        print('Executing: place at (%f, %f, %f)' % (position[0], position[1], z + 0.1))
        # Compute tool orientation from heightmap rotation angle
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        orientataion = [3.1415, 0, -position[3] / 180. * 3.1415]
        orientataion = self.ur5_euler_rotation(orientataion)
        tcp_command = "def process():\n"
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.1, orientataion[0], orientataion[1], orientataion[2],self.tool_acc, self.tool_vel)  # r = 0.09
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            position[0], position[1], z + 0.03, orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.02, orientataion[0], orientataion[1], orientataion[2], self.tool_acc,self.tool_vel)
        # tcp_command += " set_digital_out(4,True)\n"
        # tcp_command += " set_digital_out(5,True)\n"
        tcp_command += " set_digital_out(4,True)\n"
        tcp_command += " set_digital_out(5,True)\n"
        tcp_command += " sleep(0.5)\n"
        tcp_command += " set_digital_out(4,False)\n"
        tcp_command += " set_digital_out(5,False)\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            position[0], position[1], z + 0.05, orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)
        # tcp_command += " sleep(0.5)\n"
        # tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        #     x, y, z+0.01, orientataion[0], orientataion[1], orientataion[2], self.tool_acc*0.5, self.tool_vel*0.5)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(1)
        self.tcp_socket.close()
        target_pose = [position[0], position[1], z + 0.05]
        t = time.time()
        while True:
            if time.time() - t > 5:
                print('... unsuccessful place')
            robot_state = self.get_robot_state('current_poses')
            if all([np.abs(robot_state[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        return True

    def pick(self,position): # [px,py,z]
        x,y,z=self.frametrans(position[0], position[1], position[2])
        #x, y = self.normaltrans(position[0], position[1])
        self.x,self.y,self.z=x,y,z
        print('Executing: pick at (%f, %f, %f)' % (x,y,z+0.01))
        # Compute tool orientation from heightmap rotation angle
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        orientataion=self.orientation

        tcp_command = "def process():\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (
            x, y, 0.45, orientataion[0], orientataion[1], orientataion[2],self.tool_acc, self.tool_vel)
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            x, y, z+0.02, orientataion[0], orientataion[1], orientataion[2], self.tool_acc,self.tool_vel)
        tcp_command += " set_digital_out(7,True)\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            x, y, z+0.01, orientataion[0], orientataion[1], orientataion[2], self.tool_acc*0.5, self.tool_vel*0.5)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(0.5)
        self.tcp_socket.close()
        target_pose=[x, y, z+0.01]
        t=time.time()
        while True:
            if time.time()-t>5:
                print('... ka zai pick zhe li')
            robot_state=self.get_robot_state('current_poses')
            if all([np.abs(robot_state[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        return True

    def place(self,box): #
        assert (box in self.box), 'Can only place to box: '+str(self.box)
        print('Executing: place at %s' % (box))

        # Compute tool orientation from heightmap rotation angle
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        orientataion = self.orientation
        if box=='left_box':
            poses=self.boxes[0]
        elif box=='right_box':
            poses = self.boxes[1]
        else:
            print('Not defining ', box)
        print('Executing: place at (%f, %f, %f)' % (poses[1][0], poses[1][1], poses[1][2]))
        tcp_command = "def process():\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.006)\n" % (
            self.x, self.y, self.z+0.14, orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (
            poses[0][0], poses[0][1], poses[0][2], orientataion[0], orientataion[1], orientataion[2], self.tool_acc, self.tool_vel)
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
            poses[1][0], poses[1][1], poses[1][2], orientataion[0], orientataion[1], orientataion[2], self.tool_acc,self.tool_vel)
        tcp_command += " set_digital_out(7,False)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        time.sleep(0.5)
        self.tcp_socket.close()

        target_pose = [poses[1][0], poses[1][1], poses[1][2]]
        t=time.time()
        while True:
            if time.time()-t>5:
                print('... ka zai place zhe li')
            robot_state = self.get_robot_state('current_poses')
            if all([np.abs(robot_state[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        return True




def rotation_matrix(angle, direction, point=None):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = (direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction = unit_vector(direction[:3])
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    return R

def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


if __name__=="__main__":
    ur5=UR5(tcp_host_ip='192.168.1.2',tcp_port=30003)
    num = input("Please input the number of robot position for calibration: ")
    poses=[]
    for i in range(num):
        print('Move the robot to the %dth postion\n' % (i + 1))
        while True:
            cmd = input('Press 1 to record: ')
            if int(cmd) == 1:
                pose = ur5.get_robot_state('current_poses')
                poses.append(pose)
                break
            else:
                print('no other cmd\n')
                continue
    print(str(poses))

    # inner_matrix = np.array([[600.0, 0, 330.9],
    #                          [0, 600.0, 213.6],
    #                          [0, 0, 1, ]])
    # outer_matrix_R = np.array([[-1.0000, 0.0014, 0.0084],
    #                            [0.0014, 1.0000, -0.0079],
    #                            [-0.0085, -0.0079, -0.9999]])
    # outer_matrix_T = np.array([-0.0205775, 0.4909913, 0.6777570])
    # R_z = rotation_matrix(math.pi, (0.0, 0.0, 1.0))
    # outer_matrix_R = np.dot(R_z, outer_matrix_R)
    # outer_matrix_T = np.dot(R_z, outer_matrix_T.T).T
    # print outer_matrix_T
    # print outer_matrix_R

