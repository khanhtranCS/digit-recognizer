multi-class perceptron.py
import bisect

def distance(imgQ, imgC, m):
    return np.sqrt(np.sum((imgQ-imgC)**2)) / m

def kernel(distance, lambda_):
    return np.exp(-distance**2/lambda_)

def yHat(list):
    numerator = 0
    denominator = 0
    for i in range(len(list)):
        partK = kernel(list[i][1], 0.04)
        print(partK, "--",list[i][1])
        numerator += list[i][0] * partK
        denominator += partK
    return numerator/denominator

def yHat1(list):
    label = [x[0] for x in list]
    counts = np.bincount(label)
    print(np.argmax(counts))

def knn(input, imgQ, k):
    i = 0
    r, c = np.shape(input)
    list = []

    for j in range(r):
        if(len(list) == k):
            break
        dis = distance(input[j:j+1,1:], imgQ[:,0:], c-1)
        if(dis != 0):
            #list.append((input[j:j+1,0:1], dis))
            list.append((input[j,0], dis))
    list = sorted(list, key=lambda list:list[1])

    for m in range(j, r):

        currD = distance(input[m:m+1,1:], imgQ[:,0:], c-1)
        if (currD != 0 and currD < list[k-1][1]):
            index = bisect.bisect([x[1] for x in list], currD)
            list[index+1:k] = list[index:k-1]
            #list[index] = (input[m:m+1,0:1], currD)
            list[index] = (input[m,0], currD)
    return list

def main(inputTr, inputT, k):
    r,c = np.shape(inputT)
    for i in range(r):
        list = knn(inputTr, inputT[i:i+1, 0:], 10)
        yHat1(list)



df_train = pd.read_table("train.csv", delimiter = ",")
df_test = pd.read_table("test.csv", delimiter = ",")
inputTr = df_train.values
inputT = df_test.values
main(inputTr,inputT, 3)
# imgQ = np.array([[1,0,0]])
# input = np.array([[7,4,4],[9,2,2],[5,0,0],[5,6,6],[3,1,1]])
#
# print(knn(input, imgQ, 3))


