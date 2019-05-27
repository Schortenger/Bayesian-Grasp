
from PIL import Image
import pdb
import cv2
import numpy as np

infile = ('/home/schortenger/Desktop/IROS/tactile_prior/figure/initial2.png')

img = Image.open(infile)
(h1,w1) = img.size #read image size
print h1
print w1
# x_s = 400 #define standard width
# y_s = 300 #calc height based on standard width
# img = img.resize((x_s,y_s),Image.ANTIALIAS) #resize image with high-quality
# print 'adjust size: ',x_s,y_s

x=y=0
w = 100/400.0*w1
print w

h = 100/500.0*h1
print h
deltah=25
deltaw=25
while  x< h1:
    #print x
    while y< w1:

        print y
        pdb.set_trace()
        #region = img.crop((x, y, x + 100, y + 100))
        #region.save("/home/schortenger/Desktop/IROS/tactile_prior/figure/crop/img" + str(x) + str(y) + '.png')
        strin = raw_input("if or not capture 9 more points")
        if strin=='1':
            print'screw'

            #for x in np.arange(0,w,deltaw):
            for xi in np.arange(1,4,1):
                print xi
                for yj in np.arange(1,4,1):
                    print yj
                    inix=x+xi*deltaw
                    iniy=y+yj*deltah
                    region = img.crop((inix-50, iniy-50, inix + 50, iniy + 50))
                    region.save("/home/schortenger/Desktop/IROS/tactile_prior/figure/crop/img"+str(inix)+str(iniy)+'.png')
        print 'img done'
        y += h
    pdb.set_trace()
    y=0
    x=x+w
    print x


