import numpy as np
import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_data = df_train.values
test_data = df_test.values
def findCoeffW(wr):
    return 0.5 * np.log((1-wr)/float(wr))

def findCWAlpha(wr, flag):
    if (flag):
        return np.exp(-wr)
    else:
        return np.exp(wr)

def sigmoid(w, x):
    score = np.dot(w, np.transpose(x))
    return 1 / (1 + np.exp(-score))

def polynomial_kernel(u,v,d):
    return (np.dot(u,v) + 1)**d

def exponential_kernel(u, v):
    sigma = 119
    return np.exp(-np.linalg.norm(u-v)/(2*sigma**2))

def compute_y_hat2(w, x):
    def sign(x): return 1 if x >= 0 else -1
    return sign(np.dot(w,x))

def compute_y_hat(x_t, y_mistake, X_mistake, kernel):
    def sign(x): return 1 if x >= 0 else -1
    n_mistake = len(y_mistake)
    if not n_mistake:
        return sign(0)
    sum = 0
    for i in range (n_mistake):
        sum+=y_mistake[i] * kernel(X_mistake[i], x_t)
    return sign(sum)

def boosting(inputData, cSize, classes):
    y = inputData[:, 0]
    X = inputData[:, 1:].astype(np.float)
    N, D = X.shape

    w = np.zeros(D).astype(np.float)
    newW = np.copy(w)
    res = np.zeros(D).astype(np.float)
    start = 0

    flag = (np.any(classes==3) and np.any(classes==5)) or (np.any(classes==3) and np.any(classes==8)) or (np.any(classes==3) and np.any(classes==9)) \
           or (np.any(classes == 8) and np.any(classes == 9)) or (np.any(classes == 8) and np.any(classes == 5))
    if (flag):
        cSize*=8

    WeakL = np.empty((cSize,D,))
    size = int(0.80 * (N/cSize))
    count = 0

    for i in range(cSize):
        for m in range(start,start+size,1):
            yHat = sigmoid(w, X[count])
            for j in range(D):
                partial = X[m, j] * (max(0,y[count]) - yHat)
                newW[j] = w[j] + 0.8 * partial
            w = np.copy(newW)
            count+=1
        WeakL[i] = w
        start+=size

    alpha = np.zeros((cSize, N))
    alpha.fill(1.0 / N)

    for t in range(30):
        Rer = 100000
        Rwrong = np.repeat(False, N)
        Rcorrect = np.repeat(False, N)
        index = 0

        for g in range(cSize):
            wrong = np.repeat(False, N)
            correct = np.repeat(False, N)
            er = 0.0

            for l in range(N):
                y_Hat = compute_y_hat2(WeakL[g],X[l])
                if y_Hat != y[l]:
                    wrong[l] = True
                    er+=alpha[g,l]
                else:
                    correct[l] = True

            if (er < Rer):
                Rer = er
                Rwrong = wrong
                Rcorrect = correct
                index = g

        wr = findCoeffW(Rer)
        alpha[index,Rwrong] = alpha[index,Rwrong] * findCWAlpha(wr, 0)
        alpha[index,Rcorrect] = alpha[index,Rcorrect] * findCWAlpha(wr, 1)
        alpha[index] = alpha[index] / float(alpha[index].sum())
        res+=wr*WeakL[index]

    return res


def buildClassifier(inputData):
    r, c = np.shape(inputData)
    classifier = {}
    classes = np.zeros(2)
    for i in range(9):
        classes[0] = i
        for j in range(i+1,10):
            m = np.repeat(False, r)
            temp = np.copy(inputData);
            for k in range(r):
                if (temp[k,0] == i):
                    m[k] = True
                    temp[k,0] = 1
                elif (temp[k,0] == j):
                    m[k] = True
                    temp[k,0] = -1
            classes[1] = j
            classifier[i*10+j] = boosting(temp[m],50,classes)
    return classifier

def testing(testData, classifier):
    r, c = np.shape(testData)
    for i in range(r):
        res = []
        for key in classifier:
            sign = compute_y_hat2(classifier[key], testData[i])
            if sign > 0:
                if key < 10:
                    res.append(0)
                else:
                    res.append(key/10)
            elif sign < 0:
                res.append(key%10)
        yHat = np.argmax(np.bincount(res))
        print(yHat)


def normalize(input):
    r, c = np.shape(input)
    for i in range(c):
        val = np.sqrt(np.sum(input[:,i]**2))
        if (val != 0):
            input[:,i] = input[:,i]*1.0/val
    return input

def normalize2(input):
    r, c = np.shape(input)
    for i in range(r):
        input[i] = input[i]/np.linalg.norm(input[i])
    return input


# c = np.array([[0,1,2],[0,1,3],[3,2,1],[4,5,2],[1,1,3],[1,1,2],[2,2,3],[2,2,5]]).astype(np.float)
# n = normalize(c)
# print c

# print (c)
# print (c[:,0])
# print (c[:,1:])
#buildClassifier(c)
#print c

s_test = test_data[:,:]
s_train = train_data[:1,:]

#normalize(train_data[:,1:].astype(np.float))
#Ntest = normalize(s_test.astype(np.float))
#testing(Ntest, buildClassifier(train_data))

testing(test_data, buildClassifier(train_data))