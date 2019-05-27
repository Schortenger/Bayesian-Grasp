# coding: utf-8
from matplotlib import pyplot as plt
from PIL import Image
import torch, torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from camera3 import Camera
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

import torch.optim as optim
from tac_grasp.resnet import ResNet, Bottleneck, BasicBlock, resnet50, resnet101, resnet152
from tac_grasp.VGGnet import vgg16
import time
import cv2
import numpy
import pdb
from math import *
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

MODEL1_FOLDER = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/resstage1.pkl'
MODEL1_PARAM = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/resstage1_params.pkl'
# MODEL2_FOLDER = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/unknown.pkl'
# MODEL2_PARAM = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/unknown_params.pkl'
MODEL2_FOLDER = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/visionstage2.pkl'
MODEL2_PARAM = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/visionstage2_params.pkl'


STEP1 = 80
STEP2 = 60
WIN = 100 #100
ROTATION_NUM = 6
#SIDE = 100


class MyDataset(Dataset):
    def __init__(self, txt_path, ignore_zero=True, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        fh = fh.readlines()
        # fh.sort()
        # pdb.set_trace()
        imgs = []
        count0 = count1 = count2 = count3 = 0
        trainsum = 400
        for line in fh:
            line = line.rstrip()
            words = line.split("#", 1)
            # pdb.set_trace()
            if ignore_zero and words[1] == '0' and np.random.random() > 0.3:  # np.random.random()>0.05:
                # print('.')
                # pdb.set_trace()
                continue
            # if float(words[1])==0:
            #     count0+=1
            #     words[1]='0'
            #     if count0 < trainsum:
            #         imgs.append((words[0], words[1]))
            # # else:
            # #     words[1] = '1'
            # elif 0<float(words[1])<=0.5:
            #     count1 += 1
            #     words[1]='1'
            #     if count1 < trainsum:
            #         imgs.append((words[0], words[1]))
            # elif 0.5<float(words[1])<=0.85:
            #     count2 += 1
            #     words[1] = '2'
            #     if count2 < trainsum:
            #         imgs.append((words[0], words[1]))
            if float(words[1]) <= 0.85:
                words[1] = '0'
                # if count1 < trainsum:
                #     imgs.append((words[0], words[1]))
            elif float(words[1]) > 0.85:
                count3 += 1
                words[1] = '1'
                # if count3 < trainsum:
                #     imgs.append((words[0], words[1]))

            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class BayesianGraspNet(object):
    def __init__(self, model1_input='rgb', model2_input='rgb'):
        # self.banet = torch.load('/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/unknownnet.pkl')
        # self.banet.load_state_dict(
        #     torch.load('/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/unknownnet_params.pkl'))
        self.model1 = None
        self.model2 = None
        # self.model = None
        self.model1_input = model1_input  # 'rgb',
        self.model2_input = model2_input

    def preprocess(self, bbox, rgb, show=False):
        # rgb= self.model1_input
        img = rgb.copy()
        img = np.array(img)
        bbox = [int(i) for i in bbox]
        patches_rgb = []
        points = []
        win = WIN / 2
        for x in range(bbox[0], bbox[2], STEP1):
            for y in range(bbox[1], bbox[3], STEP2):
                if x - win < 0 or y - win < 0 or x + win >= img.shape[1] or y + win >= img.shape[0]:
                    continue
                patch = img[(y - win):(y + win), (x - win):(x + win)]
                # dis_patch = depth[(y - win):(y + win), (x - win):(x + win)]
                # dis_patch = np.expand_dims(dis_patch, -1)
                # pdb.set_trace()
                # dis_patch = self.process_depth(dis_patch)
                if show:
                    cv2.circle(rgb, (x, y), 2, (255, 0, 0), 2)
                    cv2.rectangle(rgb, (x + win, y - win),
                                  (x - win, y + win), (255, 255, 0), 3)
                    cv2.imshow('point', rgb)
                    cv2.waitKey(0)
                # fig = plt.figure()
                # ax = fig.add_subplot(121)
                # ax.imshow(dis_patch[:, :, 0])
                # ax = fig.add_subplot(122)
                # ax.imshow(patch)
                # plt.figure()
                # plt.imshow(depth)
                # plt.show()

                patches_rgb.append(patch)

                points.append((x, y))

        return patches_rgb, points
        # pdb.set_trace()
        # print patch.shape

    def jiugongge(self, point, rgb, show=True):
        # rgb= self.model1_input
        img = rgb.copy()
        img = np.array(img)

        patches_rgb = []
        points = []
        win = WIN / 2
        for x in range(point[0] - 40, point[0] + 45, 40):
            for y in range(point[1] - 40, point[1] + 45, 40):
                if x - win < 0 or y - win < 0 or x + win >= img.shape[1] or y + win >= img.shape[0]:
                    continue
                patch = img[(y - win):(y + win), (x - win):(x + win)]
                # dis_patch = depth[(y - win):(y + win), (x - win):(x + win)]
                # dis_patch = np.expand_dims(dis_patch, -1)
                # pdb.set_trace()
                # dis_patch = self.process_depth(dis_patch)
                if show:
                    cv2.circle(rgb, (x, y), 2, (255, 0, 0), 2)
                    cv2.rectangle(rgb, (x + win, y - win),
                                  (x - win, y + win), (255, 255, 0), 3)
                    cv2.imshow('point', rgb)
                    cv2.waitKey(0)
                # fig = plt.figure()
                # ax = fig.add_subplot(121)
                # ax.imshow(dis_patch[:, :, 0])
                # ax = fig.add_subplot(122)
                # ax.imshow(patch)
                # plt.figure()
                # plt.imshow(depth)
                # plt.show()

                patches_rgb.append(patch)

                points.append((x, y))

        return patches_rgb, points
        # pdb.set_trace()
        # print patch.shape

    def predict_stage1(self, bbox, rgb):
        # print('Suction Point Detection')
        # pdb.set_trace()
        ttt = time.time()

        if self.model1 == None:
            self.model1 = torch.load(MODEL1_FOLDER)
            self.model1.load_state_dict(torch.load(MODEL1_PARAM))
            print 'stage1 loading'
            # pdb.set_trace()
        print 'stage1 mode loading time is :', time.time() - ttt
        print '.........', bbox
        tt = time.time()
        patches, points = self.preprocess(bbox, rgb)
        oripatches = patches

        print 'stage 1 clip box time is :', time.time() - tt
        if len(points) == 0:
            return False
            # return False,[None,None,None]
        timer = time.time()
        print 'patch number', len(patches), patches[0].shape
        # for pat in patches:
        #     cv2.imshow('pat0',pat)
        #     cv2.waitKey(0)
        transform = transforms.Compose(
            [transforms.Resize(WIN), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        compa = []
        for patch in patches:
            pa = transform(transforms.ToPILImage()(np.array(patch)))
            compa.append(pa)
        patches = torch.stack(compa, dim=0).cuda()

        # print self.model1
        # pdb.set_trace()
        pres = self.model1(patches)
        # pdb.set_trace()
        outputs = torch.squeeze(pres, 1)
        _, pres = torch.max(outputs, 1)

        print 'Grasp Detection Stage1 took {:.3f}s for {:d} patches'.format(time.time() - timer, patches.shape[0])

        # pdb.set_trace()
        # index = np.where(pres.argmax(1) == 1)[0]
        tt2 = time.time()
        index = np.where(pres[:].cpu().numpy() == 1)
        # pdb.set_trace()
        index = np.array(index)
        # pdb.set_trace()
        # index=list(index)
        points_stage1 = []
        if len(index[0]) == 0:
            return 'fail', 'fail'
            # torch.cuda.empty_cache()
            #del patches
            # del self.model1
        else:
            randi = random.choice(index[0])
            roughtarget = points[randi]
            # for i in index[0]:
            #     points_stage1.append(points[i])
            # #pdb.set_trace()
            # roughtarget=random.choice(points_stage1)
            print 'stage 1 output %d points' % len(points_stage1)
            print 'stage1 threshod time is :', time.time() - tt2
            # pdb.set_trace()
            cv2.imshow('stage1_target', oripatches[randi])
            # cv2.waitKey(0)
            q = cv2.waitKey(1000)
            # Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//datapath/test/tennis/stage:" + str(
            #     1) + str(time.strftime('  %Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ".png"
            # cv2.imwrite(Img_Name, oripatches[randi])
            print q
            cv2.destroyAllWindows()

            del patches
            # del self.model1
            return roughtarget  # ,patches[randi]

    def predict_stage2(self, point, rgb, fail2times, model2=MODEL2_FOLDER):
        ttt = time.time()
        if self.model2 is None:
            self.model2 = torch.load(model2)
            self.model2.load_state_dict(torch.load(MODEL2_PARAM))
            print 'stage2 loading'
        #pdb.set_trace()
        ninepoints = []
        # pdb.set_trace()
        for x in range(point[0] - 40, point[0] + 45, 40):
            for y in range(point[1] - 40, point[1] + 45, 40):
                ninepoints.append([x, y])
        patches_rgb = []
        points = []
        angles = []
        tt = time.time()
        for i, point in enumerate(ninepoints):
            for k in range(ROTATION_NUM):
                theta = 180 / ROTATION_NUM * k
                rgb_patch = img_rot(rgb, point, theta, WIN)

                patches_rgb.append(rgb_patch)

                points.append(point)
                angles.append(theta)
        num = len(patches_rgb)
        transform = transforms.Compose(
            [transforms.Resize(WIN), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        compa = []
        for patch in patches_rgb:
            pa = transform(transforms.ToPILImage()(np.array(patch)))

            compa.append(pa)
        patches = torch.stack(compa, dim=0).cuda()

        print 'stage2 mode loading time is :', time.time() - ttt
        print 'stage2 rotate patches time is :', time.time() - tt
        t0 = time.time()
        index = []
        # del self.model2_input
        # del patches_rgb
        # pdb.set_trace()
        # pdb.set_trace()

        pre = self.model2(patches)
        # pdb.set_trace()

        outputs = torch.squeeze(pre, 1)

        _, pre = torch.max(outputs, 1)

        print 'Grasp Detection Stage2 took {:.3f}s for {:d} patches'.format(time.time() - t0, patches[0].shape[0])
        tt1 = time.time()
        # grasp_angle = pre
        # pdb.set_trace()
        index = np.where(pre[:].cpu().numpy() == 1)
        # pdb.set_trace()
        index = np.array(index)
        # pdb.set_trace()
        if len(index[0]) == 0 & fail2times < 2:
            # torch.cuda.empty_cache()
            # del self.model2
            act_point = 'fail'
            act_angle = 'fail'
            act_patch = patches_rgb[0]
            return act_point, act_angle, act_patch
        elif len(index[0]) == 0 & fail2times >= 2:
            act_point = random.choice(points)
            act_angle = random.choice[angles]
            act_patch = random.choice[patches_rgb]
            return act_point, act_angle, act_patch
        elif len(index[0]) > 0:
            # pdb.set_trace()
            points_stage2 = []
            angles_stage2 = []
            for i in index[0]:
                points_stage2.append(points[i])
                angles_stage2.append(angles[i])

            # pdb.set_trace()
            # pdb.set_trace()
            index2 = index[0]
            act_index = random.choice(index2)
            act_point = points[act_index]
            act_angle = angles[act_index]
            act_patch = patches_rgb[act_index]

            # pdb.set_trace()
            print 'stage2 threshod time is :', time.time() - tt1
            # torch.cuda.empty_cache()
            # del self.model2
            return act_point, act_angle, act_patch


def img_rot(img, point, angle, patch):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    M = cv2.getRotationMatrix2D(center, angle, 1)
    M[0, 2] += (widthNew - width) / 2
    M[1, 2] += (heightNew - height) / 2
    rotated = cv2.warpAffine(img, M, (widthNew, heightNew))
    # x=point[0]-width//2
    # y=point[1]-height//2

    [x_new, y_new] = np.dot(M, np.array([[point[0]], [point[1]], [1]]))

    x_min = int(x_new - patch // 2)
    y_min = int(y_new - patch // 2)
    img_patch = rotated[y_min:y_min + patch, x_min:x_min + patch]
    # cv2.circle(img,point,5,(255,0,0),-1)
    return img_patch


# class PointCrop(object):
#     """Crop randomly the image in a sample.
#
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self, image, pointx,pointy):
#
#
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         cenx = np.int(pointx)
#         ceny = np.int(pointy)
#
#         image = image[cenx-new_h/2: cenx + new_h/2,
#                       ceny-new_w/2: ceny + new_w/2]
#
#
#
#         return image


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    camera = Camera()
    softgraspnet = BayesianGraspNet(model1_input='rgb')
    q = cv2.waitKey(1)

    while q != 27:
        rgb, depth = camera.get_data()

        print 'cam done'
        # cv2.imshow('rgb',rgb)
        # cv2.waitKey()
        # pdb.set_trace()
        t0 = time.time()
        # 1000.

        points_stage1, patch = softgraspnet.predict_stage1([100, 100, 500, 400], rgb)
        print 'stage 1 time is :', time.time() - t0
        # pdb.set_trace()
        if points_stage1 == 'fail':
            print 'stage1 failed, no objects detected'
            # continue
        else:
            # pdb.set_trace()
            tt1 = time.time()
            print 'stage2 random time is :', time.time() - tt1
            # pdb.set_trace()
            points_stage2, angles_stage2, actpatch = softgraspnet.predict_stage2(points_stage1, rgb)
            if points_stage2 == 'fail':
                print 'stage2 failed, grasp again'
                # continue
            else:
                print points_stage2, angles_stage2
                print 'stage 2 time is :', time.time() - tt1
                print 'detection time is: %f' % (time.time() - t0)
                points_result = []  #
                points_angles = {}  # {index:[angle1,angle2,...]},  points_result[index] is (x,y)
                tt2 = time.time()

                print 'plot time is :', time.time() - tt2
                # pdb.set_trace()
                cv2.circle(actpatch, (int(points_stage2[0]), int(points_stage2[1])), 2, (255, 255, 0), 5)
                cv2.imshow('grasp_point', actpatch)

                q = cv2.waitKey()
                print q
                cv2.destroyAllWindows()

        # torch.cuda.empty_cache()

    pdb.set_trace()
