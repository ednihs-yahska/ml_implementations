import sys
from collections import defaultdict, Counter
import itertools
import numpy as np
import time

path="hw2-data/"

def process_data(filename):
    X, Y = [], []
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:-1]): # last one is target
            if (i, fv) in feature_map: # ignore unobserved features
                feat_vec[feature_map[i, fv]] = 1
        X.append(feat_vec)
        Y.append(1 if features[-1] == ">50K" else -1) # fake for testdata
    return np.array(X), np.array(Y)

def perceptron(data, maxIter=5):
    X = data[0]
    Y = data[1]
    W = np.append([0], np.zeros(X.shape[1]))
    stats = [{"updates": 0, "weights": []} for x in range(maxIter)]
    for iter in range(maxIter):
        for i, x in enumerate(X):
            x = np.append([1], x)
            y = Y[i]
            a = np.dot(W, np.transpose(x))
            if y*a <= 0:
                W = W + (y*x)
                stats[iter]["updates"]+=1
        stats[iter]["weights"]=W
    return (W, stats)

def average_perceptron(train, maxIter=5):
    X = train[0]
    Y = train[1]
    W = np.append([0], np.zeros(X.shape[1]))
    Wa = np.append([0], np.zeros(X.shape[1]))
    c = 0
    stats = [{"updates": 0, "weights": []} for x in range(maxIter)]
    for iter in range(maxIter):
        for i, x in enumerate(X):
            x = np.append([1], x)
            y = Y[i]
            a = np.dot(W, np.transpose(x))
            if y*a <= 0:
                W = W + (y*np.transpose(x))
                Wa = Wa + (c*y*np.transpose(x))
                stats[iter]["updates"]+=1
            c += 1
        stats[iter]["weights"]=(c*W - Wa)
    return (c*W - Wa, stats)

def perceptron_test(test, W):
    X = test[0]
    Y = test[1]
    errors = 0
    ppr = 0
    for i, x in enumerate(X):
        x = np.append([1], x)
        y = Y[i]
        a = np.dot(W, np.transpose(x))
        ppr = ppr + 1 if a > 0 else ppr
        if y*a <= 0:
            errors += 1  
    return ((errors/X.shape[0])*100, ppr)

def print_perceptron_results(stats, data):
    X = data[0]
    Y = data[1]
    for index, stat in enumerate(stats):
        num_updates = stat["updates"]
        _weights = stat["weights"]
        err, ppr = perceptron_test(data, _weights)
        print("epoch ", index+1, "updates ", num_updates, "(", (num_updates/X.shape[0])*100,"%) dev_err ", err,"(+", (ppr/X.shape[0])*100,"%)" )

def get_centered_features(data):
    X = data[0]
    X_bin = X[:, :-2]
    X_nums = X[:, -2:]
    return np.append(X_bin-0.5, X_nums - (np.max(X_nums[:,0]))/2, axis=1)

def _42a_feature_engineering(file, data):
    X = data[0]
    nums = np.array(get_numeric_features(file))
    X_nums = np.append(X, nums, axis=1)
    return X_nums

def get_numeric_features(file):
    num_features = []
    for line in open(file):
        line = line.strip()
        features = [float(x) for i, x in enumerate(line.split(", ")) if i in [0, 7]]
        num_features.append(features)
    return num_features

def test_prediction():
    W = a_W
    X = test_data[0]
    Y = test_data[1]
    predictions = []
    errors = 0
    ppr = 0
    for i, x in enumerate(X):
        x = np.append([1], x)
        y = Y[i]
        a = np.dot(W, np.transpose(x))
        predictions.append(np.sign(a))
        if np.sign(a) > 0:
            ppr+=1
    return (predictions,ppr)


if __name__ == "__main__":
    print("\n\n>>>>> Assuming data files are stored in a folder called hw2-data <<<<<<<< \n\n")
    field_value_freqs = defaultdict(lambda : defaultdict(int))
    for line in open(path+"income.train.txt.5k"):
        line = line.strip()
        features = line.split(", ")[:-1]
        for i, fv in enumerate(features):
            field_value_freqs[i][fv] += 1
    feature_map = {}
    feature_remap = {}
    for i, value_freqs in field_value_freqs.items():
        for v in value_freqs:
            k = len(feature_map) # bias
            feature_map[i, v] = k
            feature_remap[k] = i, v
    dimension = len(feature_map) # bias
    train_data = process_data(path+"income.train.txt.5k") 
    dev_data = process_data(path+"income.dev.txt")
    test_data = process_data(path+"income.test.blind")

    _42a_train_data = ([],[])
    _0 = _42a_feature_engineering(path+"income.train.txt.5k", train_data)
    _1 = train_data[1]
    _42a_train_data = (_0, _1)

    _42a_dev_data = ([],[])
    _d0 = _42a_feature_engineering(path+"income.dev.txt", dev_data)
    _d1 = dev_data[1]
    _42a_dev_data = (_d0, _d1)

    _42b_train_data = ([], [])
    _0 = get_centered_features(_42a_train_data)
    _1 = train_data[1]
    _42b_train_data = (_0, _1)

    _42b_dev_data = ([], [])
    _d0 = get_centered_features(_42a_dev_data)
    _d1 = dev_data[1]
    _42b_dev_data = (_d0, _d1)

    X_unit_train = _42b_train_data[0]
    X_unit_train = (X_unit_train - np.min(X_unit_train)) / (np.max(X_unit_train) - np.min(X_unit_train))

    X_unit_dev = _42b_dev_data[0]
    X_unit_dev = (X_unit_dev - np.min(X_unit_dev)) / (np.max(X_unit_dev) - np.min(X_unit_dev))

    _42c_train_data = ([],[])
    _0 = X_unit_train
    _1 = train_data[1]
    _42c_train_data = (_0, _1)

    _42c_dev_data = ([],[])
    _0 = X_unit_dev
    _1 = dev_data[1]
    _42c_dev_data = (_0, _1)

    ans=True
    while ans:
        print ("""
        1.Vanilla Perceptron
        2.Averaged Perceptron
        3.Experiment 4.1
        4.Experiment 4.2.a
        5.Experiment 4.2.b
        6.Experiment 4.2.c
        7.Experiment 4.2.d
        8.Experiment 3.2
        9.Exit/Quit
        """)
        ans=input("What would you like to do? ") 
        if ans=="1": 
            print("\n Vanilla Perceptron") 
            W, updates = perceptron(train_data, maxIter=5)
            perceptron_test(dev_data, W)
            print_perceptron_results(updates, dev_data)
        elif ans=="2":
            print("\nAveraged Perceptron") 
            a_W, average_updates = average_perceptron(train_data, maxIter=5)
            perceptron_test(dev_data, a_W)
            print_perceptron_results(average_updates, dev_data)
        elif ans=="3":
            print("\nExperiment 4.1") 
            train_data2 = np.append(train_data[0], np.array(np.transpose([train_data[1]])), axis=1)
            train_data2 = train_data2[train_data2[:,-1].argsort()]
            W, average_updates = average_perceptron((train_data2[:, 0:230], train_data2[:,-1]))
            print("Average Perceptron ", perceptron_test(dev_data, W))
            W, updates = perceptron((train_data2[:, 0:230], train_data2[:,-1]))
            print("Vanilla Perceptron ", perceptron_test(dev_data, W))
        elif ans=="4":
            print("\nExperiment 4.2.a") 
            W, updates = average_perceptron(_42a_train_data)
            perceptron_test(_42a_dev_data, W)
            print_perceptron_results(updates, _42a_dev_data)
        elif ans=="5":
            print("\nExperiment 4.2.b")
            W, updates = average_perceptron(_42b_train_data)
            perceptron_test(_42b_dev_data, W)
            print_perceptron_results(updates, _42b_dev_data)
        elif ans=="6":
            print("\nExperiment 4.2.c")
            W, updates = average_perceptron(_42c_train_data)
            perceptron_test(_42c_dev_data, W)
            print_perceptron_results(updates, _42c_dev_data)
        elif ans=="7":
            print("\nExperiment 4.2.d")
            combinations = [(149, 202),(149, 84),(149, 91),(149, 86), (202, 84), (202, 91), (202, 86), (84, 91), (84, 86), (91, 86)]
            c_dev_data = dev_data[0]
            c_train_data = train_data[0]
            for combination in combinations:
                da = dev_data[0][:, combination[0]]
                db = dev_data[0][:, combination[1]]
                ta = train_data[0][:, combination[0]]
                tb = train_data[0][:, combination[1]]
                cd = np.logical_and(da, db)*1
                ct = np.logical_and(ta, tb)*1
                c_dev_data = np.append(c_dev_data, np.transpose([cd]), axis=1)
                c_train_data = np.append(c_train_data, np.transpose([ct]), axis = 1)
            W, c_updates = average_perceptron((c_train_data, train_data[1]))
            perceptron_test((c_dev_data, dev_data[1]), W)
            print_perceptron_results(c_updates, (c_dev_data, dev_data[1]))
        elif ans=="8":
            print("\nExperiment 3.2")
            print("KNN time for same data set was : 1.17 secs" )
            start = time.time()
            a_W, average_updates = average_perceptron(train_data, maxIter=5)
            perceptron_test(dev_data, a_W)
            p = test_prediction()
            end = time.time()
            print("Perceptron time with same dataset \n\n", end-start, "secs << 1.17secs \n")
            print("------Weights------ \n", a_W)
        elif ans=="9":
            ans=False
        elif ans !="":
            print("\n Not Valid Choice Try again")