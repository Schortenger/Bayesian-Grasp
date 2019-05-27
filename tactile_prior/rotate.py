# -*- coding:utf-8 -*-

import os
from matplotlib import pyplot as plt
from PIL import Image
import torch, torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pdb
import cv2
import glob
import math


class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split("#",1)
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

class PointCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, pointx,pointy):

        #print pointx,pointy
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h=int(new_h)
        new_w = int(new_w)
        cenx = int(pointx)
        ceny = int(pointy)



        if cenx < new_h / 2:
            lf=0
        else:
            lf = cenx - new_h / 2

        if cenx + new_h/2>640:
            rg=640
        else:
            rg=cenx + new_h/2

        if ceny - new_w/2 <0:
            dn=0
        else:
            dn=ceny - new_w/2

        if  ceny + new_w/2>480:
            tp=480
        else:
            tp = ceny + new_w / 2

        image = image[lf: rg,
                      dn: tp]

        return image

#def wrname(path):
def wrname(dir):


    fopen = open('/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/flashlight/pro_flashlight.txt', 'w')

    #dir = os.listdir(path)
    for d in dir:
        #string = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/screw/train_dataset/'+d+'#'+'\n'
        string = '/home/schortenger/Desktop/IROS/tactile_prior/data/processed/flashlight/' + d + '#' +'0.8'+ '\n'
        #string = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/screw/process_dataset/' + d + '\n'
        fopen.write(string)

    fopen.close()

def cenROI(image, pointx,pointy,size):

    #print pointx,pointy
    h, w = image.shape[:2]
    new_h= new_w = 100
    new_h=int(new_h)
    new_w = int(new_w)
    cenx = int(pointx)
    ceny = int(pointy)




    lf = cenx - new_h / 2


    rg=cenx + new_h/2


    dn=ceny - new_w/2


    tp = ceny + new_w / 2


    image = image[lf: lf+new_h,
                  dn: dn+new_w]
    #pdb.set_trace()

    return image


def cropcentre(txt_path):
    fh = open(txt_path, 'r')
    x = []
    y = []
    theta = []
    for line in fh:
        line = line.rstrip()
        words = line.split(" ",-1)
        x.append((words[0]))
        y.append((words[1]))
        theta.append((words[2]))

    return x, y, theta


def xytheta(txt_path):
    fh = open(txt_path, 'r')
    xyt = []
    for line in fh:
        line = line.rstrip()
        words = line.split(":",1)
        xyt.append((words[1]))

    xyt2 = []
    for line in xyt:
        line = line.rstrip()
        words = line.split(" ",1)
        xyt2.append((words[0]))

    xyt3 = []



    for line in xyt2:

        x = line[:3]
        y = line[3:6]
        z = line[6:]
        xyt3.append((x,y,z))
        fopen = open('/home/schortenger/Desktop/IROS/tactile_prior/data/xytheta.txt', 'a')
        fopen.write(np.str(x)+' ')
        fopen.write(np.str(y)+' ')
        fopen.write(np.str(z))
        fopen.write('\n')



    fopen.close()

def rotateimg(image,degree):

    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)


    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated






def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:

        dir_list = sorted(dir_list,  key=lambda x: os.path.getctime(os.path.join(file_path, x)))
        wrname(path)
        # print(dir_list)
        return dir_list

if __name__=='__main__':
    pathxyt = '/home/schortenger/Desktop/IROS/tactile_prior/data/original/flashlight/xytheta.txt'
    ori_datapath = '/home/schortenger/Desktop/IROS/tactile_prior/data/original/flashlight'
    pro_datapath = '/home/schortenger/Desktop/IROS/tactile_prior/data/processed/flashlight'
    #datapath = '/home/schortenger/Desktop/IROS/tactile_prior/data/processed/0327'
    path = '/home/schortenger/Desktop/IROS/tactile_prior/data/screwprocessedpath.txt'
    process_path = '/home/schortenger/Desktop/IROS/tactile_prior/figure/9*crop/screwrot.txt'
    #trainset = MyDataset(path,transform=transform)

    wholedata='done'
    #wholedata='notdone'
    if wholedata=='notdone':
        sort_file = get_file_list(pro_datapath)
        wrname(sort_file)
        pdb.set_trace()

    trainset = MyDataset(process_path)
    # xytheta(pathxyt)
    # pdb.set_trace()

    x, y, theta= cropcentre(pathxyt)
    #x, y, theta = (50,50,0)

    # for i in range(0,381):
    #     score = trainset.__getitem__(i)[1]
    #     file = open('/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/screw/train_dataset.txt', 'a')
    #     file.write('/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/screw/train_dataset/cropimg'+str(i)+'.png'+'#')
    #     file.write(str(score))
    #     file.write('\n')
    #     file.close()
    #     img = trainset.__getitem__(i)[0]
    #
    # pdb.set_trace()

    #txt_path = '/home/schortenger/Desktop/IROS/tactile_prior/data/rotatepath.txt'
    txt_path = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/screw/screw.txt'
    score_path = '/home/schortenger/Desktop/IROS/tactile_prior/data/processed/screw/score.txt'
    # fh = open(score_path, 'r')
    #
    # score1=[]
    # for line1 in fh:
    #     line1 = line1.rstrip()
    #     words = line1.split(" ", 1)
    #
    #     score1.append(words[0])



    i =0

    count=0
    cou=0
    fh = open(txt_path, 'r')
    imgs = []
    for line in fh:
        line = line.rstrip()
        words = line.split("#", 1)

        img = cv2.imread(words[0])

        deg = np.float(math.degrees(np.array(0)))



        for d in [deg,deg+np.float(180)]:

            img = np.array(img)
            size1=200
            ROIimg = PointCrop(size1)
            h = ROIimg(img, 50,50)
            roimg = rotateimg(h, d)

            h2 = cenROI(roimg, size1/2, size1/2, 100)
            Img_Name = "/home/schortenger/Desktop/IROS/tactile_prior/figure/9*crop/score" + '*'+str(cou) + ".png"
            #Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//datapath//screw//cropset/score" + str(count) + ".png"
            cv2.imwrite(Img_Name, h2)
            score = trainset.__getitem__(i)[1]
            #file = open('/home/schortenger/Desktop/IROS/tactile_prior/data//datapath//flashlight//trainset.txt', 'a')
            #file.write('/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/flashlight/trainset/score'+np.str(score1[count])+'*'+str(cou)+'.png'+'#')
            #file.write(str(score1[count]))
            #file.write('\n')
            #cou = cou+1

        count=count+1
        print count
#        print np.str(score1[count])


        #pdb.set_trace()

        for rd in np.arange(deg+30, deg+361, 30):
            if rd == deg+180:
                pass
            else:

                img = np.array(img)
                size1=200
                ROIimg = PointCrop(size1)
                h = ROIimg(img, 50, 50)
                roimg = rotateimg(h, rd)
                h2 = cenROI(roimg, size1 / 2, size1 / 2, 100)
                Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//figure/rotate/cropimg" + str(
                    count) + ".png"
                cv2.imwrite(Img_Name, h2)
                score = 0
                # file = open('/home/schortenger/Desktop/IROS/tactile_prior/data/train_dataset.txt', 'a')
                # file.write(
                #     '/home/schortenger/Desktop/IROS/tactile_prior/data/train_dataset/cropimg' + str(count) + '.png' + '#')
                # file.write(str(score))
                # file.write('\n')
                count=count+1
        i=i+1

        pdb.set_trace()
        if  i%10==0:
            print str(i)+' image rotates done'
            #pdb.set_trace()

    # for i in range(0,381):
    #     img = trainset.__getitem__(i)[0]


    #     deg = np.float(math.degrees(np.array(theta[i])))
    #     pdb.set_trace()
    #     for d in [deg,deg+180]:
    #         img = img.rotate(d)
    #         img = np.array(img)
    #
    #         ROIimg = PointCrop(100)
    #         h = ROIimg(img, y[i],x[i])
    #
    #         Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//train_dataset/cropimg" + str(i) + ".png"
    #         cv2.imwrite(Img_Name, h)
    #         score = trainset.__getitem__(i)[1]
    #         file = open('/home/schortenger/Desktop/IROS/tactile_prior/data/train_dataset.txt', 'a')
    #         file.write('/home/schortenger/Desktop/IROS/tactile_prior/data/train_dataset/cropimg'+str(i)+'.png'+'#')
    #         file.write(str(score))
    #         file.write('\n')
    #     for rd in range(deg, deg+360, 10):
    #         if i == deg+180:
    #             pass
    #         else:
    #             img = img.rotate(rd)
    #             img = np.array(img)
    #
    #             ROIimg = PointCrop(100)
    #             h = ROIimg(img, y[i], x[i])
    #
    #             Img_Name = "//home//schortenger//Desktop//IROS//tactile_prior//data//train_dataset/cropimg" + str(
    #                 i) + ".png"
    #             cv2.imwrite(Img_Name, h)
    #             score = 0
    #             file = open('/home/schortenger/Desktop/IROS/tactile_prior/data/train_dataset.txt', 'a')
    #             file.write(
    #                 '/home/schortenger/Desktop/IROS/tactile_prior/data/train_dataset/cropimg' + str(i) + '.png' + '#')
    #             file.write(str(score))
    #             file.write('\n')
    #
    #     pdb.set_trace()









