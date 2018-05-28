# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:03:51 2018

@author: wanliang
"""
#from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
import pandas as pd
import PIL
from PIL import Image

#To load the data from the directory
def load_wikiArt_data(wikiArt_label_dir,wikiArt_pic_dir,img_width,img_height):
    wiki_label = pd.read_csv(wikiArt_label_dir,sep=',',header=0)
    wiki_label = wiki_label.apply(lambda x: x.astype(str).str.lower())
    Y = np.array(pd.factorize(wiki_label.emotion)[0])
    print(Y[0:10,])
    print(type(Y))
    num_image = wiki_label.shape[0]
    X=np.zeros((num_image,3,img_height,img_width))
    for i in range(1,num_image):
        f = os.path.join(wikiArt_pic_dir,'%d.jpg' %(i,))
        img = Image.open(f)
        img = img.resize((img_width,img_height),Image.ANTIALIAS)
        img=np.array(img,dtype="float32")
        if(len(img.shape) != 3):
            img = np.stack((img,img,img),axis=2)
        img=img.transpose((2,0,1))
        X[i-1,:,:,:] = img
    return X,Y
    
#To get the trainSet,valSet and testSet
def get_wikiArt_data(train_ratio,subtract_mean=True):
    #the label directory
    wikiArt_label_dir = 'wiki_3104_emotion_labels.csv'
    #the picture directory
    wikiArt_pic_dir = 'WikiArt_Images'
    #Load the data
    img_width=224
    img_height=224
    X,Y= load_wikiArt_data(wikiArt_label_dir,wikiArt_pic_dir,img_width,img_height)
    #Create a mask to separate the data into train,val and test set
    idx = np.arange(len(X))
    mask = np.random.permutation(idx)
    train_stop = int(train_ratio*X.size)
    X_train = X[mask[:train_stop],:,:,:]
    Y_train = Y[mask[:train_stop],]
    val_stop = train_stop + int((1-train_ratio)/2*X.size)
    X_val = X[mask[train_stop:val_stop],:,:,:]
    Y_val = Y[mask[train_stop:val_stop],]
    X_test = X[mask[val_stop:],:,:,:]
    Y_test = Y[mask[val_stop:],]
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
    return {
      'X_train': X_train, 'Y_train': Y_train,
      'X_val': X_val, 'Y_val': Y_val,
      'X_test': X_test, 'Y_test': Y_test,
    }
    
    
