import cv2
import matplotlib.pyplot as plt
from math import *
import numpy as np

def img_rot_1(img,point,angle,patch):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    M = cv2.getRotationMatrix2D(center, angle, 1)
    M[0, 2] += (widthNew - width) / 2
    M[1, 2] += (heightNew - height) / 2
    rotated = cv2.warpAffine(img, M, (widthNew, heightNew))
    # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # plt.figure()
    # plt.imshow(cv2.cvtColor(rotated,cv2.COLOR_BGR2RGB))
    # plt.show()
    x=point[0]-width//2
    y=point[1]-height//2
    r=sqrt(x**2+y**2)
    if y>=0:
        theta = acos(x / r) - angle * pi / 180
    elif x>=0 and y<0:
        theta = asin(y / r) - angle * pi / 180
    elif x<0 and y<0:
        theta = -(asin(y / r)+pi) - angle * pi / 180
    x_new=r*cos(theta)
    y_new=r*sin(theta)
    x_min=int(x_new+widthNew//2-patch//2)
    y_min=int(y_new+heightNew//2-patch//2)
    img_patch=rotated[y_min:y_min+patch,x_min:x_min+patch]
    #cv2.circle(img,point,5,(255,0,0),-1)
    return img_patch


def img_rot(img,point,angle,patch):
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

    [x_new,y_new]=np.dot(M,np.array([[point[0]], [point[1]], [1]]))

    # r=sqrt(x**2+y**2)
    # if y>=0:
    #     theta = acos(x / r) - angle * pi / 180
    # elif x>=0 and y<0:
    #     theta = asin(y / r) - angle * pi / 180
    # elif x<0 and y<0:
    #     theta = -(asin(y / r)+pi) - angle * pi / 180
    # x_new=r*cos(theta)
    # y_new=r*sin(theta)
    # x_min=int(x_new+widthNew//2-patch//2)
    # y_min=int(y_new+heightNew//2-patch//2)
    x_min=int(x_new-patch//2)
    y_min=int(y_new-patch//2)
    img_patch=rotated[y_min:y_min+patch,x_min:x_min+patch]
    #cv2.circle(img,point,5,(255,0,0),-1)
    return img_patch
def lastline_file(file_path):
    with open(file_path,'r') as f:
        #first_line = f.readline()
        f.seek(0,1)
        lines=f.readlines()
        if len(lines)>=0:
            lastline = lines[-1]
    return lastline

# read num.txt to record number of patch
def num_file(file_path):
    with open(file_path,'r') as f:
        lines=f.readlines()
    return lines[0],lines[1]

# write 0.txt 1.txt
def list_file(file_name,rows):
    with open(file_name,'a') as f:
        for row in rows:
            f.write(row)

def list_num(file_path,nopick_num,pick_num):
    with open(file_path,'w') as f:
        f.seek(0,0)
        f.write('00'+'\t'+'%0.4d'%nopick_num+'\n')
        f.write('01'+'\t'+'%0.4d'%pick_num)

if __name__=="__main__":

    file_path='num.txt'
    a,b=num_file(file_path)
    nopick_num = int(a.split('\t')[1])
    pick_num = int(b.split('\t')[1])
    print(nopick_num, pick_num)
    nopick_num=nopick_num+1
    pick_num=pick_num+1
    list_num(file_path,nopick_num,pick_num)

    # print a.split('\t')
    # print a.split('\t')[0],a.split('\t')[1][4:8]
    # b=['1\trgb_0001.jpg\t0\t0\t1\n','2\trgb_0001.jpg\t0\t0\t1\n']
    # list_file(file_path,b)


    # img = cv2.imread('cam1.jpg')
    #
    # point=(400,200)
    # patch=120
    # angle=270
    # img_patch=img_rot(img,point,angle,patch)
    # cv2.rectangle(img,(point[0]-patch/2,point[1]-patch/2),(point[0]+patch/2,point[1]+patch/2),(255,0,0))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img_patch,cv2.COLOR_BGR2RGB))
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img[point[1]-30:point[1]+30,point[0]-30:point[0]+30], cv2.COLOR_BGR2RGB))
    # plt.show()

