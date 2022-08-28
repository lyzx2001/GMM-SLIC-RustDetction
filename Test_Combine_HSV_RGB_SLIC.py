import cv2
import numpy as np
import sys
import joblib
from pyheatmap.heatmap import HeatMap
from sklearn.mixture import GaussianMixture as GMM 
from sklearn.utils import shuffle
import os
import argparse

def TestGMM_RGB(TestImagePath):
    testImage = cv2.imread(TestImagePath)
    testImage_RGB = cv2.imread(TestImagePath)
    testImage_RGB = np.array(testImage_RGB, dtype=np.float64) 

    testImage_RGB[:,:,0] = testImage_RGB[:,:,0] / 255
    testImage_RGB[:,:,1] = testImage_RGB[:,:,1] / 255
    testImage_RGB[:,:,2] = testImage_RGB[:,:,2] / 255
    testImage_RGB[:,:,2] *= 1.5

    rows, cols,ch = testImage.shape
    testImage_HSV = np.reshape(testImage_RGB, (rows * cols, ch))

    Sum = np.sum(np.array(testImage, dtype=np.float64), axis=2)
    Sum = np.reshape(Sum,(rows * cols))
    return testImage_HSV, Sum

def TestGMM_HSV(TestImagePath):
    testImage_RGB = cv2.imread(TestImagePath)
    testImage_HSV = cv2.cvtColor(testImage_RGB, cv2.COLOR_BGR2HSV)
    testImage_HSV = np.array(testImage_HSV, dtype=np.float64) 
    testImage_HSV[:,:,0] = (testImage_HSV[:,:,0] + 90) % 180

    testImage_HSV[:,:,0] = testImage_HSV[:,:,0] / 180
    testImage_HSV[:,:,1] = testImage_HSV[:,:,1] / 255
    testImage_HSV[:,:,2] = testImage_HSV[:,:,2] / 255
    testImage_HSV[:,:,0] *= 1.5

    rows, cols,ch = testImage_HSV.shape
    testImage_HSV = np.reshape(testImage_HSV, (rows * cols, ch))

    Sum = np.sum(np.array(testImage_RGB, dtype=np.float64), axis=2)
    Sum = np.reshape(Sum, (rows * cols))
    return testImage_HSV, Sum

def drawHeatMap(probability, rows, cols): ##  probability: one dim long array (a col)
    A = np.argwhere(np.ones((rows, cols)))
    X = A[:,0]
    Y = A[:,1]
    data = []
    N = len(X)
    for i in range(N):
        tmp = [int(Y[i]), int(X[i]), probability[i]]
        data.append(tmp)
    heat = HeatMap(data)
    heat.heatmap(save_as="./Output/HeatMap_GMM_Combine.png")

def Scaling0255(probability):
    Grey = probability * 255
    Grey = np.array(Grey, dtype=np.uint8) 
    return Grey

def LoadModel(model1Path, model2Path, testimagePath, ClusterNum):
    model =  joblib.load(model1Path)
    select = []
    for i in range(ClusterNum):
        if model.weights_[i] > 1/ClusterNum:
            select.append(i)

    rows, cols, ch = cv2.imread(testimagePath).shape
    testImage_HSV, Sum = TestGMM_HSV(testimagePath)

    probability_HSV = model.predict_proba(testImage_HSV)
    probability_HSV = probability_HSV[:,select]
    probability_HSV = np.max(probability_HSV, axis=1)
    Cri = np.bitwise_or(Sum==0, Sum==765)
    probability_HSV[Cri] = 0
    drawHeatMap(probability_HSV, rows, cols)

    model =  joblib.load(model2Path)
    select = []
    for i in range(ClusterNum):
        if model.weights_[i] > 1/ClusterNum:
            select.append(i)

    rows, cols, ch = cv2.imread(testimagePath).shape
    testImage_RGB, Sum = TestGMM_RGB(testimagePath)

    probability_RGB = model.predict_proba(testImage_RGB)
    probability_RGB = probability_RGB[:,select]
    probability_RGB = np.max(probability_RGB, axis=1)
    Cri = np.bitwise_or(Sum==0, Sum==765)
    probability_RGB[Cri] = 0
    ## take average
    probability = 0.4 * probability_RGB + 0.6 * probability_HSV
    drawHeatMap(probability, rows, cols)
    
    ##  Scale probability to 0-255:
    grey = np.reshape(Scaling0255(np.copy(probability)), (rows, cols))
    probability2D = np.reshape(probability, (rows, cols))
    return grey, probability2D

def classify(img_path, npy, ret = 255):
    img = cv2.imread(img_path)
    # initialize the parameters for SLIC
    # region_size refers to the size of every piece of superpixel segmentation, ruler refers to the smooth factor
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=5, ruler=20.0)
    slic.iterate(10)
    mask_slic = slic.getLabelContourMask()

    cv2.imshow('mask_slic', mask_slic)
    cv2.imwrite('./Output/SLIC_Black_HSV.jpg', mask_slic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mask_inv_slic = cv2.bitwise_not(mask_slic)

    cv2.imshow('mask_inv_slic',mask_inv_slic)
    cv2.imwrite('./Output/SLIC_White_HSV.jpg', mask_inv_slic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic)
    cv2.imshow("img_slic", img_slic)
    cv2.waitKey(0)
    cv2.imwrite('./Output/SLIC_HSV.jpg', img_slic)
    cv2.destroyAllWindows()
    
    prob_value = npy
    length = np.size(slic.getLabels(), axis=0)
    width = np.size(slic.getLabels(), axis=1)
    label_array = slic.getLabels()
    label_num = np.max(label_array)
    score = []
    for i in range(label_num):
        super_score = []
        for j in range(length):
            for k in range(width):
                if (label_array[j][k] == i + 1):
                    super_score.append(prob_value[j][k])
        mean_score = np.array(super_score).mean()
        score.append(mean_score)
    
    steps = [ret / 255.0]
    for h in range(len(steps)):
        bin_score = []

        for i in range(len(score)):
            if (score[i] < steps[h]):
                bin_score.append(0)
            else:
                bin_score.append(1)    

        final_img = np.zeros_like(img)
        binary_img = np.zeros_like(img)
        for i in range(length):
            for j in range(width):
                b_score = bin_score[label_array[i][j] - 1]
                if (b_score == 0):
                    final_img[i][j] = img[i][j]
                    binary_img[i][j][0] = 255
                    binary_img[i][j][1] = 255
                    binary_img[i][j][2] = 255
                else:
                    final_img[i][j][0] = 0
                    final_img[i][j][1] = 0
                    final_img[i][j][2] = 255
                    binary_img[i][j][0] = 0
                    binary_img[i][j][1] = 0
                    binary_img[i][j][2] = 255
        
        return final_img, binary_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_img_path', type=str, default = None) # pass in the path of the test image
    args = parser.parse_args()
    grey, prob = LoadModel('./GMMmodel_HSV.pkl', './GMMmodel_RGB.pkl', args.test_img_path, 7) 
    ret, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    final_img, binary_img = classify(args.test_img_path, prob, ret)
    print("Successfully generated test output!")
    cv2.imshow('Final_Img_Combine', final_img)
    cv2.imwrite('./Output/Final_Img_Combine.jpg', final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Binary_Img_Combine', binary_img)
    cv2.imwrite('./Output/Binary_Img_Combine.jpg', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()