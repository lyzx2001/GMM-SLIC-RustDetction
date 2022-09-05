import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM 
from sklearn.utils import shuffle 
import joblib
import os
import argparse

##  H 170
##  Used to get images that have already make a difference between background and object.
##  Background is black, while object is colorful.
##  This program takes all training images to train GMM.

def HandleImage(ImagePath,Random):
##  parameters: ImagePath: string for image path; Random 0/1, 1 for is random
##  1. Rule out background black color
##  2. Scale the H,S,V components
    image = cv2.imread(ImagePath)

    Sum = np.sum(np.array(image, dtype=np.float64), axis=2)
    ##  Sum != 0 is rust color

    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ##  Merge distinct HSV area into one single area
    image_HSV = np.array(image_HSV, dtype=np.float64) 

    image_HSV[:,:,0] = (image_HSV[:,:,0] + 90) % 180
    Criterion = np.bitwise_and(Sum !=0, Sum!=765)
    ## Scaling
    image_HSV[:,:,0] = image_HSV[:,:,0] / 180
    image_HSV[:,:,1] = image_HSV[:,:,1] / 255
    image_HSV[:,:,2] = image_HSV[:,:,2] / 255
    ##  Emphasize H component
    image_HSV[:,:,0] = image_HSV[:,:,0] * 1.5

    trainInput = image_HSV[Criterion]

    rows, cols = trainInput.shape
    trainInput = np.reshape(trainInput, (rows, cols))
    if Random == 1:
        trainInput = shuffle(trainInput, random_state=0)
    return trainInput

def TrainGMM_segmented_picture(ClusterNum, TrainPath):
##  FolderPath: string, training images folder path
##  ClusterNum: int, number of clusters one wants to train
    images = []
    post = ["PNG", "JPG", "JPEG", "png", "jpg", "jpeg"]
    for home, dirs, files in os.walk(TrainPath):
        for filename in files:
            for format in range(len(post)):
                if post[format] in filename:
                    images.append(os.path.join(home, filename))
                    break

    All = np.array([0, 0, 0])
    for image in images:
        trainedInput = HandleImage(image, 1)
        All = np.vstack([All, trainedInput])
    
    All = All[1:]
    All = shuffle(All, random_state=0)
    model = GMM(n_components=ClusterNum, random_state=0).fit(All) #define the model
    joblib.dump(model, './GMMmodel_HSV.pkl')
    print("Trained model successfully dumped to './GMMmodel_HSV.pkl'!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default = None) # pass in the path of the test image
    args = parser.parse_args()
    TrainGMM_segmented_picture(7, args.train_path)