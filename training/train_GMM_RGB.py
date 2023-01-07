import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM 
from sklearn.utils import shuffle 
import joblib
import os
import argparse

##  Used to train images that have already make a difference between background and object.
##  Background is black, while object is colorful.
##  This program takes all training images to train GMM.

def handleImg(imgPath, isRandom):
##  parameters: imgPath: string for image path; isRandom: int (0 or 1), 1 for is random, 0 for is not random
##  1. Rule out background black color
##  2. Scale the H,S,V components
    img = cv2.imread(imgPath)
    summation = np.sum(np.array(img, dtype=np.float64), axis=2)
    img = np.array(img, dtype=np.float64) 

    # Scaling
    img[:,:,0] = img[:,:,0] / 255
    img[:,:,1] = img[:,:,1] / 255
    img[:,:,2] = img[:,:,2] / 255
    # Emphasize R component
    img[:,:,2] = img[:,:,2] * 1.5

    isBlackOrWhite = np.bitwise_and(summation!=0, summation!=765)
    trainInput = img[isBlackOrWhite]

    rows, cols = trainInput.shape
    trainInput = np.reshape(trainInput, (rows , cols))
    if isRandom == 1:
        trainInput = shuffle(trainInput, random_state=0)

    return trainInput

def trainGMMSegmentedPicture(trainPath, clusterNum):
##  parameters:
##  trainPath: string, the directory path for training images
##  clusterNum: int, number of clusters one wants to train
    imgs = []
    post = ["PNG", "JPG", "JPEG", "png", "jpg", "jpeg"]
    for home, dirs, files in os.walk(trainPath):
        for filename in files:
            for format in range(len(post)):
                if post[format] in filename:
                    imgs.append(os.path.join(home, filename))
                    break

    allData = np.array([0, 0, 0])
    for img in imgs:
        trainedInput = handleImg(img, 1)
        allData = np.vstack([allData, trainedInput])

    allData = allData[1:]
    allData = shuffle(allData, random_state=0)
    # define the model
    model = GMM(n_components=clusterNum, random_state=0).fit(allData)
    joblib.dump(model, './model/GMMmodel_RGB.pkl')
    print("Trained model successfully dumped to './model/GMMmodel_RGB.pkl'!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # pass in the path of the directory that contains all the train images
    parser.add_argument('--train_path', type=str, default=None)
    args = parser.parse_args()
    trainGMMSegmentedPicture(args.train_path, 7)