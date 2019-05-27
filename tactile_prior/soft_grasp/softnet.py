#!/usr/bin/env python

import os
import datetime

import h5py
import numpy as np

from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.layers import Conv2D, MaxPool2D,Flatten,Dropout,Dense
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model, load_model
from keras.utils import np_utils
from skimage import io,transform
from keras.utils import plot_model

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

from keras import backend as K
from skimage.filters import gaussian
import glob

import matplotlib.pyplot as plt
import math
import cv2
import time
import os
from PIL import Image
from math import *
from func_list import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pdb
import random
currt_path = os.path.dirname(os.path.realpath(__file__))
# pdb.set_trace()
MODEL1_FOLDER= os.path.join(currt_path,'stage1.hdf5')
MODEL2_FOLDER= os.path.join(currt_path,'stage2.hdf5')
# MODEL2_FOLDER= './output/two_stages/181220_1617_vgg16_rgbddd_stage2_360/epoch_50_model.hdf5'
MODEL_FOLDER= './output/two_stages/181224_1120_vgg16_rgbddd_stage2_360/epoch_50_model.hdf5'
WIN = 64
STEP = 9
ROTATION_NUM = 18
IMAGE_PATH='./data/demo_img'
rgb_path=os.path.join(IMAGE_PATH,'rgb_2.png')
depth_path=os.path.join(IMAGE_PATH,'depth_2.png')

WIDTH_NORM = 0.05
SIDE=64
font = cv2.FONT_HERSHEY_SIMPLEX
def predict(event,x,y,flags,param):
    global model,img,rgb,depth
    if event == cv2.EVENT_LBUTTONUP:
        rgb_patch=rgb[y-SIDE/2:y+SIDE/2,x-SIDE/2:x+SIDE/2,:]
        depth_patch = depth[y - SIDE / 2:y + SIDE / 2, x - SIDE / 2:x + SIDE / 2]
        depth_patch=np.expand_dims(depth_patch,-1)
        depth_patch = process_depth(depth_patch)
        depth_patch=np.concatenate((depth_patch,depth_patch,depth_patch),2)
        rgb_patch=np.expand_dims(rgb_patch,0)
        depth_patch = np.expand_dims(depth_patch, 0)

        #pdb.set_trace()
        patch=[rgb_patch,depth_patch]
        pre=model.predict(patch)
        #pdb.set_trace()
        # print pre[0], pre[1], (pre[1][0][0] *180) , pre[1][0][1]*WIDTH_NORM
        print (pre)
        #print np.arctan2(pre[1][0][1], pre[1][0][0]) * 180 / math.pi / 2
        # print pre[0],pre[1],(pre[1][0][0]*math.pi-math.pi/2)/math.pi*180,pre[1][0][1]
        #print pre[0],pre[1],(pre[2]*math.pi-math.pi/2)/math.pi*180,pre[3]
        #pre=pre[0]
        #text='%f_%0.3f_%0.3f_%0.2f'%(pre[0],pre[1],pre[2],pre[3])
        cv2.rectangle(img,(x-SIDE/2,y-SIDE/2),(x+SIDE/2,y+SIDE/2),(0,255,0),1)
        #cv2.putText(img,text,(x-SIDE/2,y-SIDE/2-3),font,0.3, (255, 255, 255), 1)

def process_depth(data_depth,show=False):
    depth = data_depth[:]
    max_depth = np.max(depth)
    depth[np.where(depth != 0)] = max_depth - depth[np.where(depth != 0)]
    depth = gaussian(depth, 2, preserve_range=True)
    depth = depth*1000.
    if show:
        plt.imshow(depth[:, :, 0])
        plt.show()
    return depth

def confirm_show_dataset_label(data,label,model,show=True,ROTATION_NUM=36):
    assert isinstance(data,list) and len(data) ==2, "data is [rgb_train, depth_train]"
    rgb,depth = data[0],data[1]
    num = rgb.shape[0]
    for i in range(num):
        rgb_img,depth_img,pre,cmd = rgb[i],depth[i],label[0][i],label[1][i]
        if show :
            f = plt.figure()
            ax = f.add_subplot(1, 2, 1)
            # pdb.set_trace()
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            ax.imshow(bgr_img)
            ax = f.add_subplot(1, 2, 2)
            # max_depth = np.max(depth_img)
            # depth_img[np.where(depth_img!=0)]=max_depth-depth_img[np.where(depth_img!=0)]
            # gaussian(depth_img[:, :, 0], 5.0, preserve_range=True)
            ax.imshow(depth_img[:, :, 2])
            #label_show = label[:]
            cmd[0] = round(cmd[0] * 180,0)
            cmd[1] = round(cmd[1] * WIDTH_NORM,3)
            print '......'
            print 'label:',pre, cmd
            rgb_patch = np.expand_dims(rgb_img, 0)
            depth_patch = np.expand_dims(depth_img, 0)
            patch = [rgb_patch, depth_patch]
            pre = model.predict(patch)
            print 'predic:',pre[0], pre[1], (pre[1][0][0] * 180), pre[1][0][1] * WIDTH_NORM
            plt.show()
def display_test():
    INPUT_DATASET = './data/181107/dataset_181112_1957.hdf5'
    f = h5py.File(INPUT_DATASET, 'r')
    print 'Loading Data ...'
    t0 = time.time()
    rgb_test = np.array(f['test/rgb'])
    depth_test = np.expand_dims(np.array(f['test/depth']), -1)
    process_depth(depth_test)
    depth_test = np.concatenate((depth_test, depth_test, depth_test), axis=3)
    label_test = np.array(f['test/label'])
    test_label_pre = np_utils.to_categorical(label_test[:, 0], 2)
    # label_test=np.hstack((label_pre,label_test[:,1:]))
    label_test = [test_label_pre, np.concatenate((label_test[:, 1:2], label_test[:, 2:] / WIDTH_NORM), -1)]
    f.close()
    print 'Data Done. time is %f ' % (time.time() - t0)
    x_test = [rgb_test,depth_test]
    y_test = label_test
    model = load_model(MODEL1_FOLDER)
    confirm_show_dataset_label(data=x_test,label=y_test, model=model)

class SoftGraspNet(object):
    def __init__(self,model1_input = 'rgbddd',model2_input = 'rgbddd'):
        self.model1 = None
        self.model2 = None
        self.model = None
        self.model1_input = model1_input # 'rgb','ddd',
        self.model2_input = model2_input

    def process_depth(self,data_depth, show=False, unit ='mm'):
        depth = np.zeros(data_depth.shape)
        max_depth = np.max(data_depth)
        depth[np.where(data_depth != 0)] = max_depth - data_depth[np.where(data_depth != 0)]
        # pdb.set_trace()
        depth = gaussian(depth, 2, preserve_range=True)
        # depth = depth*1000 if unit is 'mm' else depth
        depth=depth*1000
        if show:
            plt.imshow(depth)
            plt.show()
        return depth

    def predict_one_stage(self,bbox,depth,rgb):
        if self.model == None:
            self.model = load_model(MODEL_FOLDER)
        rgb_img = rgb.copy()
        depth_img = depth.copy()
        # pdb.set_trace()
        patches, points = self.clip_box_rgbd(bbox, rgb_img, depth_img)
        # pdb.set_trace()
        if len(points) == 0:
            return False
        return self.predict_stage2(points,depth,rgb,model=MODEL_FOLDER)

    def predict_stage2(self,points_stage1,depth,rgb,model = MODEL2_FOLDER,threshod = 0.75):
        ttt = time.time()
        if self.model2 == None:
            self.model2=load_model(model)
            print 'stage2 loading'
        # pdb.set_trace()
        patches_rgb = []
        patches_depth = []
        points = []
        angles = []
        print 'stage2 mode loading time is :',time.time()-ttt
        tt = time.time()
        for i, point in enumerate(points_stage1):
            for k in range(ROTATION_NUM):
                theta = 180 / ROTATION_NUM * k
                rgb_patch = img_rot(rgb, point, theta, SIDE)
                dis_patch = img_rot(depth, point, theta, SIDE)
                dis_patch = np.expand_dims(dis_patch, -1)
                dis_patch = self.process_depth(dis_patch)
                dis_patch = np.concatenate((dis_patch, dis_patch, dis_patch), axis=2)
                patches_rgb.append(rgb_patch)
                patches_depth.append(dis_patch)
                points.append(point)
                angles.append(theta)
        num = len(patches_rgb)
        pp = np.zeros((num, SIDE, SIDE, 3))
        dd = np.zeros((num, SIDE, SIDE, 3))
        for i in range(num):
            pp[i] = patches_rgb[i]
            dd[i] = patches_depth[i]
        patches = [pp,dd]
        print 'stage2 rotate patches time is :',time.time()-tt
        t0 =time.time()
        pre = self.model2.predict(patches)
        print 'Grasp Detection Stage2 took {:.3f}s for {:d} patches'.format(time.time()-t0,patches[0].shape[0])
        tt1 = time.time()
        grasp_angle = pre[0]
        grasp_depth = pre[1]
        index = np.where(grasp_angle[:, 1] > threshod)[0]
        points_stage2 = []
        angles_stage2 = []
        depth_stage2 = []
        for i in index:
            points_stage2.append(points[i])
            angles_stage2.append(angles[i])
            depth_stage2.append(grasp_depth[i])
        # pdb.set_trace()
        print 'stage2 threshod time is :',time.time()-tt1
        return points_stage2,angles_stage2,depth_stage2

    def predict_stage1(self,bbox,rgb,depth,threshod = 0.9):
        # print('Suction Point Detection')
        # pdb.set_trace()
        ttt = time.time()
        if self.model1 == None:
            self.model1=load_model(MODEL1_FOLDER)
            print 'stage1 loading'
            # pdb.set_trace()
        print 'stage1 mode loading time is :',time.time()-ttt
        print '.........',bbox
        tt = time.time()
        patches, points=self.clip_box_rgbd(bbox,rgb,depth)
        print 'stage 1 clip box time is :',time.time()-tt
        if len(points)==0:
            return False
            # return False,[None,None,None]
        timer = time.time()
        print '..........',len(patches),patches[0].shape
        if self.model1_input is 'rgbddd':
            pres = self.model1.predict(patches)
        elif self.model1_input is 'rgb':
            pres = self.model1.predict(patches[0])
        elif self.model1_input is 'ddd':
            pres = self.model1.predict(patches[1])
        elif self.model1_input is 'dddddd':
            pres = self.model1.predict([patches[1],patches[1]])
        elif self.model1_input is 'rgbrgb':
            pres = self.model1.predict([patches[0],patches[0]])
        print 'Grasp Detection Stage1 took {:.3f}s for {:d} patches'.format(time.time()-timer,patches[0].shape[0])
        # index = np.where(pres.argmax(1) == 1)[0]
        tt2 = time.time()
        index = np.where(pres[:,1] > threshod)[0]
        # pdb.set_trace()
        points_stage1 = []
        for i in index:
            points_stage1.append(points[i])
        print 'stage 1 output %d points'%len(points_stage1)
        print 'stage1 threshod time is :',time.time()-tt2
        return points_stage1
        # min_index = index[0]
        # min = abs(points[min_index][0] - (bbox[2] + bbox[0]) / 2) + abs(points[min_index][1] - (bbox[3] + bbox[1]) / 2)
        # for i in index:
        #     point = points[i]
        #     if abs(point[0] - (bbox[2] + bbox[0]) / 2) + abs(point[1] - (bbox[3] + bbox[1]) / 2) < min:
        #         min = abs(point[0] - (bbox[2] + bbox[0]) / 2) + abs(point[1] - (bbox[3] + bbox[1]) / 2)
        #         min_index = i
        # px,py=float(points[min_index][0]),float(points[min_index][1])
        # z=patches[1][min_index][WIN/2,WIN/2][0]
        # return True,[px,py,z]

    def clip_box_rgbd(self,bbox,rgb,depth,show = False):
        # rgb = cv2.imread(rgb_path)
        # depth = np.asarray(Image.open(depth_path))
        image = rgb.copy()
        depth = depth.copy()
        image = np.array(image)
        bbox = [int(i) for i in bbox]
        patches_rgb = []
        patches_depth = []
        points = []
        win = WIN / 2
        for x in range(bbox[0], bbox[2], STEP):
            for y in range(bbox[1], bbox[3], STEP):
                if x - win < 0 or y - win < 0 or x + win >= image.shape[1] or y + win >= image.shape[0]:
                    continue
                patch = image[(y - win):(y + win), (x - win):(x + win)]
                dis_patch = depth[(y - win):(y + win), (x - win):(x + win)]
                dis_patch = np.expand_dims(dis_patch, -1)
                # pdb.set_trace()
                dis_patch = self.process_depth(dis_patch)
                if show:
                    cv2.circle(rgb, (x, y), 2, (255, 0, 0), -1)
                # fig = plt.figure()
                # ax = fig.add_subplot(121)
                # ax.imshow(dis_patch[:, :, 0])
                # ax = fig.add_subplot(122)
                # ax.imshow(patch)
                # plt.figure()
                # plt.imshow(depth)
                # plt.show()
                dis_patch = np.concatenate((dis_patch, dis_patch, dis_patch), axis=2)
                patches_rgb.append(patch)
                patches_depth.append(dis_patch)
                points.append((x, y))
                # pdb.set_trace()
                # print patch.shape
        num = len(patches_rgb)
        pp = np.zeros((num, win * 2, win * 2, 3))
        dd = np.zeros((num, win * 2, win * 2, 3))
        for i in range(num):
            pp[i] = patches_rgb[i]
            dd[i] = patches_depth[i]
        return [pp, dd], points
def listMean(list_):
    sum = 0
    for i in list_:
        sum += i
    return float(sum)/len(list_)

def getEndPt(point,angle,r):
    x0, y0 = point
    x1, y1 = x0 + r * cos(radians(angle)), y0 + r * sin(radians(angle))
    return (int(x1),int(y1))

def getEndP(point,angle,r):
    x0, y0 = point
    x1, y1 = x0 + r * sin(radians(angle)), y0 - r * cos(radians(angle))
    return (int(x1),int(y1))

def plotGrasp(rgb,point,angle,r1=25,r2=7):
    p1 = getEndPt(point,angle,r1)
    p2 = getEndPt(point,angle,-r1)
    p11 = getEndP(p1,angle,r2)
    p12 = getEndP(p1,angle,-r2)
    p21 = getEndP(p2,angle,r2)
    p22 = getEndP(p2,angle,-r2)

    cv2.line(rgb, p1,p2, (0, 0, 255), 1)
    cv2.line(rgb, p11, p12, (0, 0, 255), 1)
    cv2.line(rgb, p21, p22, (0, 0, 255), 1)
    cv2.circle(rgb, point, 2, (255, 0, 0), -1)

if __name__=='__main__':
    #display_test()
    # rgb=cv2.imread(rgb_path)
    # depth=np.array(Image.open(depth_path))/1000.
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    # ax = fig.add_subplot(122)
    # ax.imshow(depth)
    # plt.show()

    from camera import Camera

    camera = Camera()
    softgraspnet = SoftGraspNet(model1_input='rgb')
    q = cv2.waitKey(1)
    while q!=27:
        rgb, depth = camera.get_data()
        print 'cam done'
        # cv2.imshow('rgb',rgb)
        # cv2.waitKey()
        # pdb.set_trace()
        t0 = time.time()
        # 1000.
        points_stage1 = softgraspnet.predict_stage1([40, 65, 626, 410], rgb, depth)
        print 'stage 1 time is :',time.time()-t0
        # pdb.set_trace()
        tt1 = time.time()
        if len(points_stage1) > 5:
            points_stage1 = random.sample(points_stage1, 5)
        print 'stage2 random time is :',time.time()-tt1
        points_stage2, angles_stage2, depth_stage2 = softgraspnet.predict_stage2(points_stage1, depth, rgb)
        print 'stage 2 time is :',time.time() - tt1
        # pdb.set_trace()
        # not 1000.
        # points_stage2, angles_stage2 = suctionnet.predict_one_stage([40,65,626,410],depth,rgb)
        print 'detection time is: %f' % (time.time() - t0)
        points_result = []  #
        points_angles = {}  # {index:[angle1,angle2,...]},  points_result[index] is (x,y)
        tt2 = time.time()
        for i, point in enumerate(points_stage2):
            if not point in points_result:
                points_result.append(point)
                points_angles.update({len(points_result) - 1: []})
            index = points_result.index(point)
            points_angles[index].append(angles_stage2[i])
        # pdb.set_trace()
        for index in points_angles.keys():
            angles = points_angles[index]
            for inx, angle in enumerate(angles):
                if 90 < angle <= 270:
                    angles[inx] = angle - 180
                elif angle > 270:
                    angles[inx] = angle - 360
            ang = listMean(angles)
            print ang,points_result[index]
            # cv2.circle(rgb,point,2,-1)
            # angle = angles_stage2[i]
            plotGrasp(rgb, points_result[index], ang)
            # r = 25
            # p = points_result[index]
            # x = p[0] + r*cos(radians(ang))
            # y = p[1] + r*sin(radians(ang))
            # x0 = p[0] - r * cos(radians(ang))
            # y0 = p[1] - r * sin(radians(ang))
            # cv2.line(rgb, (int(x0),int(y0)),(int(x),int(y)),(0,0,255),1)
            # cv2.circle(rgb, (x, y), 1, (255, 0, 0), -1)
        print 'plot time is :',time.time()-tt2
        cv2.imshow('pre_stage1', rgb)
        q = cv2.waitKey()
        print q
        cv2.destroyAllWindows()
    pdb.set_trace()

    # pdb.set_trace()
    # h, w, c = rgb.shape
    #pdb.set_trace()
    # model = load_model(MODEL1_FOLDER)
    # img=rgb.copy()
    # cv2.namedWindow('predict')
    # cv2.setMouseCallback('predict', predict)
    #
    #
    # while True:
    #     #cv2.imshow('predict',img)
    #     cv2.imshow('predict', img)
    #     k = cv2.waitKey(1) & 0xFF
    #     # if k==ord('q'):
    #     #     break
    #
    #
    #
    # pdb.set_trace()

