from datasketch import MinHashLSHForest, MinHash
from datasketch import MinHash, MinHashLSH
import pandas as pd
import numpy as np
import base64

# best is 0.2 and 512
threshold = 0.2
perm = 600
def distance(imgQ, imgC):
    return np.sqrt(np.sum((imgQ-imgC)**2))

def main():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_label = pd.read_csv("submission_softmax_100.csv")

    train_data = df_train.values
    test_data = df_test.values
    test_label = df_label.values

    #actual test label
    test_actual_label = test_label[:, 1:]

    #print('hellow')
    r, c = np.shape(train_data)
    #print("")
    rT, cT = np.shape(test_data[:1000,:])
    rowh = []
    lsh = MinHashLSH(threshold=threshold, num_perm=perm)

    for i in range(r):
        #print(i)
        rowh.append(MinHash(num_perm=perm))
        for j in range(c-1):
         rowh[i].update(base64.b64encode(train_data[i, j+1]))
        lsh.insert(i, rowh[i])

    # count = 0
    # for i in range(r):
    #     query = MinHash(num_perm=256)
    #     for j in range(c - 1):
    #         query.update(train_data[i, j+1])
    #     result = lsh.query(query)
    #     #print("result ", train_data[result[0],0])
    #     if (train_data[result[0],0] == train_data[i, 0]):
    #         count+=1
    # print(count)
    # print(count*1.0/r)

    # for test data
    count = 0
    for i in range(rT):
        #print(i)
        query = MinHash(num_perm=perm)
        for j in range(cT):
            query.update(base64.b64encode(test_data[i, j]))
        result = lsh.query(query)
        index = []
        dis = []
        for k in range(len(result)):
            index.append(train_data[result[k],0])
            dis.append(distance(test_data[i,:], train_data[result[k],1:]))
        if (len(index) != 0):
            index = np.array(index)
            sortedI = np.copy(index[np.argsort(dis)].flatten())
            res = -1
            if (len(sortedI) >= 5):
                res = np.argmax(np.bincount(sortedI[0:5]))
            else:
                res = np.argmax(np.bincount(sortedI))
            if (res != test_actual_label[i][0]):
                count += 1

        else:
            print("?")
    print("mistake Rate is ", count/rT)

# for test data
main()
#for i in range(len(result)):
#    print(train_data[result[i],0])






# data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
#         'estimating', 'the', 'similarity', 'between', 'datasets']
# data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
#         'estimating', 'the', 'similarity', 'between', 'documents']
# data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
#         'estimating', 'the', 'similarity', 'between', 'documents']
#
# # Create MinHash objects
# m1 = MinHash(num_perm=128)
# m2 = MinHash(num_perm=128)
# m3 = MinHash(num_perm=128)
# for d in data1:
#     m1.update(d.encode('utf8'))
# for d in data2:
#     m2.update(d.encode('utf8'))
# for d in data3:
#     m3.update(d.encode('utf8'))
#
# # Create a MinHash LSH Forest with the same num_perm parameter
# forest = MinHashLSHForest(num_perm=128)
#
# # Add m2 and m3 into the index
# forest.add("m2", m2)
# forest.add("m3", m3)
#
# # IMPORTANT: must call index() otherwise the keys won't be searchable
# forest.index()
#
# # Check for membership using the key
# print("m2" in forest)
# print("m3" in forest)
#
# # Using m1 as the query, retrieve top 2 keys that have the higest Jaccard
# result = forest.query(m1, 2)
# print("Top 2 candidates", result)