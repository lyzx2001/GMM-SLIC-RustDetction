import cv2
import numpy as np
import joblib
from pyheatmap.heatmap import HeatMap
import argparse

def drawHeatMap(probability, rows, cols):
##  probability: one dim long array (a col)
    A = np.argwhere(np.ones((rows, cols)))
    X = A[:,0]
    Y = A[:,1]
    data = []
    N = len(X)
    for i in range(N):
        tmp = [int(Y[i]), int(X[i]), probability[i]]
        data.append(tmp)
    heat = HeatMap(data)
    heat.heatmap(save_as="./Output/HeatMap_GMM_HSV.png")

def testGMM(testImgPath):
    testImgRGB = cv2.imread(testImgPath)
    testImgHSV = cv2.cvtColor(testImgRGB, cv2.COLOR_BGR2HSV)
    testImgHSV = np.array(testImgHSV, dtype=np.float64) 
    testImgHSV[:,:,0] = (testImgHSV[:,:,0] + 90) % 180

    testImgHSV[:,:,0] = testImgHSV[:,:,0] / 180
    testImgHSV[:,:,1] = testImgHSV[:,:,1] / 255
    testImgHSV[:,:,2] = testImgHSV[:,:,2] / 255
    testImgHSV[:,:,0] *= 1.5

    rows, cols, ch = testImgHSV.shape
    testImgHSV = np.reshape(testImgHSV, (rows * cols, ch))

    summation = np.sum(np.array(testImgRGB, dtype=np.float64), axis=2)
    summation = np.reshape(summation, (rows * cols))
    return testImgHSV, summation

def scaling(probability):
    grey = probability * 255
    grey = np.array(grey, dtype=np.uint8) 
    return grey

def loadModel(modelPath, testImgPath, clusterNum):
##  modelPath: .pkl path
##  testImgPath: test image path
    model = joblib.load(modelPath)

    select = []
    for i in range(clusterNum):
        if model.weights_[i] > 1 / clusterNum:
            select.append(i)

    rows, cols, ch = cv2.imread(testImgPath).shape
    testImgHSV, summation = testGMM(testImgPath)
    probability = model.predict_proba(testImgHSV)
    probability = probability[:,select]
    probability = np.max(probability, axis=1)
    isBlackOrWhite = np.bitwise_or(summation==0, summation==765)
    probability[isBlackOrWhite] = 0
    drawHeatMap(probability, rows, cols)
    
    # Scale probability from 0-1 to 0-255
    grey = np.reshape(scaling(np.copy(probability)), (rows, cols)) 
    probability2D = np.reshape(probability, (rows, cols))
    print("Probability prediction completed!")
    return grey, probability2D

def classify(imgPath, npy, ret=255):
    img = cv2.imread(imgPath)
    # initialize the parameters for SLIC
    # region_size refers to the size of every piece of superpixel segmentation, ruler refers to the smooth factor
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=5, ruler=20.0)
    slic.iterate(10)
    maskSlic = slic.getLabelContourMask()

    cv2.imwrite('./Output/SLIC_Black_HSV.png', maskSlic)
    # display image (could be commented out)
    cv2.imshow('maskSlic', maskSlic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    maskInvSlic = cv2.bitwise_not(maskSlic)
    cv2.imwrite('./Output/SLIC_White_HSV.png', maskInvSlic)
    # display image (could be commented out)
    cv2.imshow('maskInvSlic', maskInvSlic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgSlic = cv2.bitwise_and(img, img, mask=maskInvSlic)
    cv2.imwrite('./Output/SLIC_HSV.png', imgSlic)
    # display image (could be commented out)
    cv2.imshow("imgSlic", imgSlic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    probValue = npy
    length = np.size(slic.getLabels(), axis=0)
    width = np.size(slic.getLabels(), axis=1)
    labelArray = slic.getLabels()
    labelNum = np.max(labelArray)
    score = []
    for i in range(labelNum):
        superScore = []
        for j in range(length):
            for k in range(width):
                if (labelArray[j][k] == i + 1):
                    superScore.append(probValue[j][k])
        if len(superScore) != 0:
            meanScore = np.array(superScore).mean()
            score.append(meanScore)
        else:
            score.append(0)
    
    binScore = []
    for i in range(len(score)):
        if (score[i] < ret / 255.0):
            binScore.append(0)
        else:
            binScore.append(1)    

    finalImg = np.zeros_like(img)
    binaryImg = np.zeros_like(img)
    for i in range(length):
        for j in range(width):
            bScore = binScore[labelArray[i][j] - 1]
            if (bScore == 0):
                finalImg[i][j] = img[i][j]
                binaryImg[i][j][0] = 255
                binaryImg[i][j][1] = 255
                binaryImg[i][j][2] = 255
            else:
                finalImg[i][j][0] = 0
                finalImg[i][j][1] = 0
                finalImg[i][j][2] = 255
                binaryImg[i][j][0] = 0
                binaryImg[i][j][1] = 0
                binaryImg[i][j][2] = 255
        
    return finalImg, binaryImg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # pass in the path of the test image
    parser.add_argument('--test_img_path', type=str, default=None)
    args = parser.parse_args()
    grey, prob = loadModel('./Model/GMMmodel_HSV.pkl', args.test_img_path, 7) 
    ret, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    finalImg, binaryImg = classify(args.test_img_path, prob, ret)
    print("Successfully generated test output!")
    cv2.imwrite('./Output/finalImg_HSV.png', finalImg)
    # display image (could be commented out)
    cv2.imshow('finalImg_HSV', finalImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('./Output/binaryImg_HSV.png', binaryImg)
    # display image (could be commented out)
    cv2.imshow('binaryImg_HSV', binaryImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()