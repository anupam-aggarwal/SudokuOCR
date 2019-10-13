
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
#import matplotlib.pyplot as plt
import math
import sys

from PIL import Image as ims
from scipy.ndimage import interpolation as inter


# In[2]:


def nearestInt(x):
    f,i = math.modf(x)
    if(f<.6):
        return int(i)
    else:
        return int(i+1)


def normalise(pts, width):
    normal = [nearestInt(i[1]/width) for i in pts]
    print(normal)
    
    finalPts = []
    
    for i in range(len(normal)):
        if(normal[i] <=1):
            finalPts.append(pts[i][0])
        else:
            temp = (pts[i][0] - pts[i-1][0])/normal[i]
            for j in range(normal[i]-1):
                finalPts.append(int(finalPts[-1]+temp))
            finalPts.append(pts[i][0])
    
    print(finalPts)
    return finalPts


def getPeaks(hist):
    th = (np.average(hist)*2.5)
    bl =  hist > th
    points = [ i for i in range(bl.shape[0]) if bl[i] == True]
    
    diff = [points[i+1] - points[i] for i in range(len(points)-1)]
    dic = list(zip(points[1:],diff))
    width = (points[-1]-points[0])/10
    
    finalPts = [(points[0],0)] + [i for i in dic if (i[1] >= width-5)]
    finalPts = normalise(finalPts,width)

    return finalPts

def makeCollage(ls):
    length = 0
    if(type(ls)=='np.ndarray'):
        length = ls.shape[0]
    else:
        length = len(ls)

    col = math.floor(math.sqrt(length))
    row = length//col
    #print (row,col)

    res = ls[0]
    for i in range(1,col):
        res = np.concatenate((res,ls[i]),axis=1)

    for i in range(1,row):
        temp = ls[i*col]
        for j in range(1,col):
#             print(temp.shape,ls[i*col+j].shape)
            temp = np.concatenate((temp,ls[i*col+j]),axis=1)
        res = np.concatenate((res,temp))

#     cv2.imshow("res",res)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    rem = length - row*col
    if(rem>0):
        temp = ls[row*col]
        for i in range(1,rem):
            temp = np.concatenate((temp,ls[row*col+i]),axis=1)

        for j in range(rem,col):
            tp = np.zeros(ls[1].shape,dtype="uint8")
            temp = np.concatenate((temp,tp),axis=1)

        res = np.concatenate((res,temp))

    return res

def process(img):
    kernel = np.ones((3,3),np.uint8)
    img = cv2.resize(img,(64,64))
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    
    bd = 8
    img[:bd,:] = np.zeros((bd,img.shape[1]))
    img[-1*bd:,:] = np.zeros((bd,img.shape[1]))
    img[:,:bd] = np.zeros((img.shape[0],bd))
    img[:,-1*bd:] = np.zeros((img.shape[0],bd))
    
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.medianBlur(img,5)
    
    img = cv2.resize(img,(28,28))
#     img = pad_image(img,255)
#     img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    
    return img

def pad_image(img,val):
    if(len(img.shape)==2):
        return cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT,value=val)
    elif(len(img.shape)==3):
        return cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)


# In[32]:

# Reading the input
img = cv2.imread(str(sys.argv[1]))
# img = cv2.resize(img,(0,0), fx = .4, fy = .4)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200)
# edges = skew_fix(edges)

cv2.imshow('res',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)

# In[33]:


# Getting Points for vertical lines
histV = cv2.reduce(edges,0,cv2.REDUCE_AVG).reshape(-1)
#plt.plot(histV)
finalPtsV = getPeaks(histV)

# Getting Points for Horizontal Lines
histH = cv2.reduce(edges,1,cv2.REDUCE_AVG).reshape(-1)
#plt.plot(histH)
finalPtsH = getPeaks(histH)


# In[34]:

'''
# Displaying output for lines
copy = np.copy(img)

for i in finalPtsH:
    cv2.line(copy,(0,i),(img.shape[1],i),(255,255,0),3)
    
for i in finalPtsV:
    cv2.line(copy,(i,0),(i,img.shape[0]),(255,255,0),3)
    
cv2.imshow('res',copy)
cv2.waitKey(0)
cv2.destroyAllWindows() 

'''
# In[35]:


# Getting list of numbers
nums = []

for i in range(len(finalPtsH)-1):
    for j in range(len(finalPtsV)-1):
        im = gray[finalPtsH[i]:finalPtsH[i+1],finalPtsV[j]:finalPtsV[j+1]]
        nums.append(process(im))


# In[36]:


# Displaying a collage of black and white imgs of nums
collage = makeCollage(nums)
cv2.imshow('res',collage)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[37]:


np.save('nums',nums)

