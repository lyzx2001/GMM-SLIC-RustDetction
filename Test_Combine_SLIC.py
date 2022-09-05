import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM 
from sklearn.utils import shuffle 
from pyheatmap.heatmap import HeatMap
import joblib
import argparse

def HandleImage_HSV(ImagePath, isRandom):
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
    Criterion = np.bitwise_and(Sum!=0, Sum!=765)
    ## Scaling
    image_HSV[:,:,0] = image_HSV[:,:,0] / 180
    image_HSV[:,:,1] = image_HSV[:,:,1] / 255
    image_HSV[:,:,2] = image_HSV[:,:,2] / 255
    ##  Emphasize H component
    image_HSV[:,:,0] = image_HSV[:,:,0] * 1.5

    trainInput = image_HSV[Criterion]

    rows, cols = trainInput.shape
    trainInput = np.reshape(trainInput, (rows, cols))
    if isRandom == 1:
        trainInput = shuffle(trainInput, random_state=0)
    return trainInput

def HandleImage_RGB(ImagePath, isRandom):
##  parameters: ImagePath: string for image path; Random 0/1, 1 for is random
##  1. Rule out background black color
##  2. Scale the H,S,V components
    image = cv2.imread(ImagePath)

    Sum = np.sum(np.array(image, dtype=np.float64),axis = 2)

    image = np.array(image, dtype=np.float64) 

    ## Scaling
    image[:,:,0] = image[:,:,0] / 255
    image[:,:,1] = image[:,:,1] / 255
    image[:,:,2] = image[:,:,2] / 255
    ##  Emphasize R component
    image[:,:,2] = image[:,:,2] * 1.5

    Criterion = np.bitwise_and(Sum!=0, Sum!=765)
    trainInput = image[Criterion]

    rows, cols = trainInput.shape
    trainInput = np.reshape(trainInput, (rows, cols))
    if isRandom == 1:
        trainInput = shuffle(trainInput, random_state=0)
    return trainInput

def TestGMM_RGB(TestImagePath):
    testImage = cv2.imread(TestImagePath)
    testImage_BGR = cv2.imread(TestImagePath)
    testImage_BGR = np.array(testImage_BGR, dtype=np.float64) 

    testImage_BGR[:,:,0] = testImage_BGR[:,:,0] / 255
    testImage_BGR[:,:,1] = testImage_BGR[:,:,1] / 255
    testImage_BGR[:,:,2] = testImage_BGR[:,:,2] / 255
    testImage_BGR[:,:,2] *= 1.5

    rows, cols,ch = testImage.shape
    testImage_HSV = np.reshape(testImage_BGR, (rows * cols, ch))

    return testImage_HSV

def TestGMM_HSV(TestImagePath):
    testImage_BGR = cv2.imread(TestImagePath)
    testImage_HSV = cv2.cvtColor(testImage_BGR, cv2.COLOR_BGR2HSV)
    testImage_HSV = np.array(testImage_HSV, dtype=np.float64) 
    testImage_HSV[:,:,0] = (testImage_HSV[:,:,0] + 90) % 180

    testImage_HSV[:,:,0] = testImage_HSV[:,:,0] / 180
    testImage_HSV[:,:,1] = testImage_HSV[:,:,1] / 255
    testImage_HSV[:,:,2] = testImage_HSV[:,:,2] / 255
    testImage_HSV[:,:,0] *= 1.5

    rows, cols,ch = testImage_HSV.shape
    testImage_HSV = np.reshape(testImage_HSV, (rows * cols, ch))

    Sum = np.sum(np.array(testImage_BGR, dtype=np.float64), axis=2)
    Sum = np.reshape(Sum,(rows * cols))
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
    ##  Draw a heat map
    heat.heatmap(save_as="./Output/HeatMap_GMM_Combine.png")

def Scaling0255(probability):
    Grey = probability * 255
    Grey = np.array(Grey, dtype=np.uint8) 
    return Grey

def LoadModel(modelPath, testimagePath, ClusterNum):
    model =  joblib.load(modelPath)
    select = []
    for i in range(ClusterNum):
        if model.weights_[i] > 0.122:
            select.append(i)

    rows, cols, ch = cv2.imread(testimagePath).shape
    testImage_HSV, Sum = TestGMM_HSV(testimagePath)
    testImage_RGB = TestGMM_RGB(testimagePath)
    testImage = np.hstack((testImage_HSV, testImage_RGB))
    probability_combine = model.predict_proba(testImage)
    probability_combine = probability_combine[:,select]
    probability = np.max(probability_combine, axis=1)
    Cri = np.bitwise_or(Sum==0, Sum==765)
    probability_combine[Cri] = 0
    
    grey = np.reshape(Scaling0255(np.copy(probability)), (rows, cols))
    probability2D = np.reshape(probability, (rows, cols))
    print("Probability prediction completed!")
    drawHeatMap(probability, rows, cols)

    return grey, probability2D

def classify(img_path, npy, ret=255):
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
        if len(super_score) != 0:
            mean_score = np.array(super_score).mean()
            score.append(mean_score)
        else:
            score.append(0)
    
    bin_score = []
    for i in range(len(score)):
        if (score[i] < ret / 255.0):
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
    grey, prob = LoadModel('./GMMmodel_Combine.pkl', args.test_img_path, 7) 
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