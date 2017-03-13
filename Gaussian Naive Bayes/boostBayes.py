import numpy as np
import pandas as pd
import math

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_data = df_train.values
test_data = df_test.values

def gaussian(mu, st, x):
    ex = np.exp(-(np.power(x-mu,2)/(2*np.power(st,2))))
    res = (1 / (np.sqrt(2*math.pi) * st)) * ex
    return res

def testing(testData, trainData):
    r, c = np.shape(testData)
    r1, c1 = np.shape(trainData)
    mean, std = build(trainData)

    prC = (np.bincount(trainData[:,0]) * 1.0) / r1
    for i in range(r):
        gaus = gaussian(mean,std,testData[i])
        #print(gaus)
        #print(gaus.shape)
        #exit(0)
        pr = np.transpose(np.transpose(gaus) * prC)
        #print("pr",pr)
        #exit(0)
        sumL = np.sum(np.log(pr), axis=1)
        print(sumL)
        # print("preid", products)
        #print(np.argmax(sumL))

def build(input):
    r, c = np.shape(input)
    mean = np.zeros((10, c-1)).astype(float)
    std = np.zeros((10, c-1)).astype(float)


    for i in range(10):
        indices = np.where(input[:,0] == i)
        temp = input[indices]
        mean[i] = np.mean(temp[:,1:], axis=0)
        #print(mean[i])
        std[i] = np.nanstd(temp[:,1:], axis=0)
        indices = np.where(std == 0)
        std[indices] = 0.001

    return mean, std

testing(test_data, train_data)
