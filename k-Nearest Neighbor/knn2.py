import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def distM(inputTr, inputT):
    #print("I'm here ")
    dotP = -2 * np.dot(inputT, np.transpose(inputTr))
    sumSquaredTr = np.square(inputTr).sum(axis = 1)
    sumSquaredT = np.square(inputT).sum(axis = 1)
    return np.sqrt(dotP + sumSquaredTr + np.transpose(np.matrix(sumSquaredT)))
#
# def main(inputTr, inputT, yTr, k):
#     distanceM = distM(inputTr[:,1:], inputT)
#     r,c = np.shape(inputT)
#     yHat = np.zeros(r)
#     for i in range(1000):
#         sLabels = np.copy(yTr[np.argsort(distanceM[i,:])].flatten())
#         yHat[i] = np.argmax(np.bincount(sLabels[0:k]))
#         print (yHat[i])

def main1(inputTr, inputT, yTr, distanceM, k, num_data, true_test_label):
    r,c = np.shape(inputT)
    yHat = np.zeros(r)
    count = 0
    print("k = ", k)
    # use to predict the accuracy of each digit
    digit_count = {0:0,1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    mistake_count = {0:0,1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for i in range(num_data):
        #print(true_test_label[i][0])
        #exit(0)
        if (true_test_label[i][0] == 8):
            digit_count[8] += 1
        #digit_count[true_test_label[i][0]] += 1
        sLabels = np.copy(yTr[np.argsort(distanceM[i,:])].flatten())
        yHat[i] = np.argmax(np.bincount(sLabels[0:k]))
        if (yHat[i] != true_test_label[i][0] and true_test_label[i][0] == 8):
            mistake_count[yHat[i]] += 1
            #count+=1
    for j in range(10):
        if (j != 8):
            print(mistake_count[j]/digit_count[8])
    print()
    #     if (yHat[i] != true_test_label[i][0]): #for each incorrect prediction
    #         mistake_count[true_test_label[i][0]] += 1
    #         count += 1
    #
    # for j in range(10):
    #     print(mistake_count[j]/digit_count[j])
    # print (count / num_data)
    # print()


def analyze_knn(num_data):
    """
    Top-level wrapper to iterate over a bunch of values of k and plot the
    distortions and misclassification rates.
    """
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_label = pd.read_csv("submission_softmax_100.csv")

    train_data = df_train.values
    test_data = df_test.values
    test_label = df_label.values

    #print("here")
    k = range(1,11)
    errs = []
    inputTr = train_data
    inputT = test_data
    distanceM = distM(inputTr[:, 1:], inputT[:num_data, :])
    #print("I'm here1")
    #distanceM = distM(inputTr[:, :], inputT[:num_data, :])
    for i in range (10):
        #print(i+1)
        #k.append(i+1)
        #errs.append(main1(inputTr, inputTr[:, 0], i+1, distanceM, num_data))
        main1(inputTr, inputT, train_data[:,:1],distanceM, i + 1,num_data, test_label[:, 1:])
        #print(errs)



# inputTr = np.array([[8,1,2,3,4,5,6,7,8],[7,9,8,7,6,5,4,3,2],[3,4,5,6,2,1,2,3,9]])
# inputT = np.array([[4,6,8,9,10,11,12,13],[7,7,7,8,8,8,9,9]])
#
# main(inputTr, inputT, inputTr[:,0],2)
analyze_knn(1000)

