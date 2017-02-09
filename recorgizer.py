import numpy as np
import pandas as pd
import bisect as bisect


df_train = pd.read_csv("train.csv")

arr_train = df_train.values;
train_rows, train_cols = arr_train.shape;

train_label = arr_train[:, :1]
train_digits= arr_train[:train_rows, 1:]

#print(train_pixels[0])
imgQ = [2,2]
data_label = np.array([1,2,3,4])
data = np.array([[1,1], [2,2], [3,3], [4,4]])


def distance (q_dig, data_dig):
    return np.sqrt(np.sum(((data_dig - q_dig)/255)**2))

def kernel(distance,lambda_):
    return np.exp(-distance/lambda_)

def kNNAlg(train_label,train_digits, q_dig, k, row_num):
    init_dist2kNN = []
    for i in range (k):
        #print(i)
        dis = distance(q_dig, train_digits[i])
        # distance tuple which contain (distance, label, 728 data pixels)
        dis_tup = (dis, train_label[i],train_digits[i])
        #print(dis_tup[1])
        init_dist2kNN.append(dis_tup)
        #print(len(init_dist2kNN))
    init_dist2kNN.sort(key=lambda tup:tup[0])

    for j in range (k+1, row_num):
        dis = distance(train_digits[j], q_dig)
        if (dis < init_dist2kNN[k-1][0]):
            for b in range(k-1):
                left_el = init_dist2kNN[b][0]
                right_el = init_dist2kNN[b+1][0]
                #right_el = init_dist2kNN[b][0]
                if (dis > left_el and dis < right_el):
                    init_dist2kNN[b+2:k] = init_dist2kNN[b+1:k-1]
                    init_dist2kNN[b+1] = (dis, train_label[j], train_digits[j])
                #print(train_label[j])
    return init_dist2kNN



def predict_dig(lambda_, q_digit):
    kNN = kNNAlg(train_label, train_digits, q_digit, 10, train_rows)
    #print(train_digits[0])
    sum_kernel = 0.0
    sum_kernel_dig = 0.0
    for i in range (len(kNN)):
        kern = kernel(kNN[i][0], lambda_)
        #print(kNN[i][0])
        sum_kernel_dig += kern * kNN[i][1]
        sum_kernel += kern
    return sum_kernel_dig/sum_kernel

def perc_err(label, digits, rows, lambda_):

    count = 0.0
    for i in range (10):
        predict_y = predict_dig(lambda_, digits[i])
        actual_y = label[i]
        if (np.abs(actual_y - predict_y) < 0.00001):
            count += 1
    return count/10


print(perc_err(train_label, train_digits, train_rows, 0.4))