from numpy import *
import time

euclidean = lambda source, row: pow(sum(pow(source - row,2), axis=1), 0.5)
manhattan = lambda source, row: sum(absolute(source-row), axis=1)

def create_mapping(dev, train, test):
    all_split_dev = hsplit(dev, dev.shape[1])
    all_split_train = hsplit(train, train.shape[1])
    all_split_test = hsplit(test, test.shape[1])
    cat_data_dev = hstack(append(all_split_dev[1:7], all_split_dev[8:9], axis=0))
    cat_data_train = hstack(append(all_split_train[1:7], all_split_train[8:9], axis=0))
    cat_data_test = hstack(append(all_split_test[1:7], all_split_test[8:9], axis=0))
    all_cat = append(append(cat_data_dev, cat_data_train, axis=0), cat_data_test, axis=0)
    all_mapping = {}
    for row in all_cat:
        new_row = []
        for i,x in enumerate(row):
            feature = (i,x)
            if feature not in all_mapping:
                all_mapping[feature] = len(all_mapping)
    return all_mapping

def binarize(data, all_mapping):
    _mapping = {}
    _binarized_list = []
    for row in data:
        new_row = []
        for i,x in enumerate(row):
            feature = (i,x)
            new_row.append(all_mapping[feature])
        _binarized_list.append(new_row)
    _bindata = zeros((data.shape[0], len(all_mapping)))
    for i, row in enumerate(_binarized_list):
        for x in row:
            _bindata[i][x] = 1
    return _bindata
    
def binarize_and_normalize(data, all_mapping):
    all_split = hsplit(data, data.shape[1])
    cat_data = hstack(append(all_split[1:7], all_split[8:9], axis=0))
    numeric_data = array(hstack(append(array([all_split[0]]), array([all_split[7]]), axis=0)), dtype="f")
    numeric_data -= amin(numeric_data, axis=0)
    numeric_data /= ptp(numeric_data, axis=0)
    _bindata = binarize(cat_data, all_mapping)
    return append(numeric_data, _bindata, axis=1)

def kNN(source, target, source_labels, target_labels, K, distance_measure=lambda source, row: pow(sum(pow(source - row,2), axis=1), 0.5)):
    predictions = []
    stats = {}
    for k in K:
        k = min(k, source.shape[0])
        distances = []
        success = 0
        positive = 0
        for i, row in enumerate(target):
            row_distances = []
            #distances.append(append([row], sum(pow(source - row, 2)), axis=1))
            row_distances = distance_measure(source, row)
            closest = sort(row_distances)[:k]
            closest_indexes = []
            possible_labels = []
            prediction = ""
            vote = 0
            majority = {}
            for closest_distance in closest:
                closest_indexes.append(where(row_distances == closest_distance)[0])
                #print(where(row_distances == closest_distance)[0])
            closest_indexes = array(concatenate(closest_indexes))[0:k]
            for closest_index in closest_indexes:
                possible_labels.append(source_labels[closest_index])
            for possible_label in possible_labels:
                if possible_label in majority:
                    majority[possible_label] += 1
                else:
                    majority[possible_label] = 1
            for key, value in majority.items():
                if value > vote:
                    prediction = key
                    vote = value
            if len(target_labels) and target_labels[i] == prediction:
                success += 1  
            else:
                success += 1
                predictions.append(append(row, array([prediction])))
            if ">" in prediction:
                positive += 1
        stats[k] = {"error": 1-(success/target.shape[0]), "ppr": positive/target.shape[0]}
        #print("k= ", k, "Error ", 1-(success/target.shape[0]), "Predicted Positive Rate ", positive/target.shape[0])
    if len(predictions):
        return (array(predictions), stats)
    return stats

dev_data = genfromtxt("hw1-data/income.dev.txt", delimiter=", ", dtype="<U20")
dev_train = genfromtxt("hw1-data/income.train.txt.5k", delimiter=", ", dtype="<U20")
dev_test = genfromtxt("hw1-data/income.test.blind", delimiter=", ", dtype="<U20")

all_mapping = create_mapping(dev_data, dev_train, dev_test)

dev = binarize_and_normalize(dev_data, all_mapping)
test = binarize_and_normalize(dev_test, all_mapping)
train = binarize_and_normalize(dev_train, all_mapping)

def predict_blind_test():
    test = binarize_and_normalize(dev_test, all_mapping)
    source = train
    target = test
    target_labels = []
    source_labels = dev_train[:, dev_train.shape[1]-1]
    K = [51]
    result = kNN(source, target, source_labels, target_labels, K)
    test_prediction=result[0]
    test_prediction_final = append(dev_test, test_prediction[:,93].reshape(1000,1), axis=1)
    savetxt("hw1-data/income.test.predicted.4.txt", test_prediction_final, fmt="%s, %s, %s, %s, %s, %s, %s, %s, %s, %s")

def dev_euclidean():
    source = train
    target = dev
    target_labels = dev_data[:, dev_data.shape[1]-1]
    source_labels = dev_train[:, dev_train.shape[1]-1]
    K=[1, 3, 5, 7, 9, 99, 999, 9999]
    return kNN(source, target, source_labels, target_labels, K)

def train_euclidean():
    source = train
    target = train
    target_labels = dev_train[:, dev_train.shape[1]-1]
    source_labels = dev_train[:, dev_train.shape[1]-1]
    K=[1, 3, 5, 7, 9, 99, 999, 9999]
    return kNN(source, target, source_labels, target_labels, K)

def dev_manhattan():
    source = train
    target = dev
    target_labels = dev_data[:, dev_data.shape[1]-1]
    source_labels = dev_train[:, dev_train.shape[1]-1]
    K=[1, 3, 5, 7, 9, 99, 999, 9999]
    return kNN(source, target, source_labels, target_labels, K, distance_measure=manhattan)

def train_manhattan():
    source = train
    target = train
    target_labels = dev_train[:, dev_train.shape[1]-1]
    source_labels = dev_train[:, dev_train.shape[1]-1]
    K=[1, 3, 5, 7, 9, 99, 999, 9999]
    return kNN(source, target, source_labels, target_labels, K, distance_measure=manhattan)

def ball_create_mapping(dev, train, test):
    all_split_dev = hsplit(dev, dev.shape[1])
    all_split_train = hsplit(train, train.shape[1])
    all_split_test = hsplit(test, test.shape[1])
    cat_data_dev = hstack(append(all_split_dev[0:8], all_split_dev[8:9], axis=0))
    cat_data_train = hstack(append(all_split_train[0:8], all_split_train[8:9], axis=0))
    cat_data_test = hstack(append(all_split_test[0:8], all_split_test[8:9], axis=0))
    return append(append(cat_data_dev, cat_data_train, axis=0), cat_data_test, axis=0)

ball_all_mapping = {}

def ball_binarize(data):
    _binarized_list = []
    for row in data:
        new_row = []
        for i,x in enumerate(row):
            feature = (i,x)
            new_row.append(ball_all_mapping[feature])
        _binarized_list.append(new_row)
    _bindata = zeros((data.shape[0], len(ball_all_mapping)))
    for i, row in enumerate(_binarized_list):
        for x in row:
            _bindata[i][x] = 1
    return _bindata





def dev_all_binarized():
    ball_all_cat = ball_create_mapping(dev_data, dev_train, dev_test)
    for row in ball_all_cat:
        new_row = []
        for i,x in enumerate(row):
            feature = (i,x)
            if feature not in ball_all_mapping:
                ball_all_mapping[feature] = len(ball_all_mapping)
    bin_data = ball_binarize(dev_data[:,0:9])
    bin_train = ball_binarize(dev_train[:,0:9])
    bin_test = ball_binarize(dev_test[:,0:9])
    source = bin_train
    target = bin_data
    target_labels = dev_data[:, dev_data.shape[1]-1]
    source_labels = dev_train[:, dev_train.shape[1]-1]
    euclidean = lambda source, row: pow(sum(pow(source - row,2), axis=1), 0.5)
    manhattan = lambda source, row: sum(absolute(source-row), axis=1)
    K=[1, 3, 5, 7, 9, 99, 999, 9999]
    return kNN(source, target, source_labels, target_labels, K)

def train_all_binarized():
    ball_all_cat = ball_create_mapping(dev_data, dev_train, dev_test)
    for row in ball_all_cat:
        new_row = []
        for i,x in enumerate(row):
            feature = (i,x)
            if feature not in ball_all_mapping:
                ball_all_mapping[feature] = len(ball_all_mapping)
    bin_data = ball_binarize(dev_data[:,0:9])
    bin_train = ball_binarize(dev_train[:,0:9])
    bin_test = ball_binarize(dev_test[:,0:9])
    source = bin_train
    target = bin_train
    target_labels = dev_train[:, dev_train.shape[1]-1]
    source_labels = dev_train[:, dev_train.shape[1]-1]
    euclidean = lambda source, row: pow(sum(pow(source - row,2), axis=1), 0.5)
    manhattan = lambda source, row: sum(absolute(source-row), axis=1)
    K=[1, 3, 5, 7, 9, 99, 999, 9999]
    return kNN(source, target, source_labels, target_labels, K)

def predict_blind():
    start = time.time()
    source = train
    target = test
    target_labels = []
    source_labels = dev_train[:, dev_train.shape[1]-1]
    K = [51]
    result = kNN(source, target, source_labels, target_labels, K)
    test_prediction = result[0]
    test_prediction_final = append(dev_test, test_prediction[:,93].reshape(1000,1), axis=1)
    savetxt("hw1-data/income.test.predicted", test_prediction_final, fmt="%s, %s, %s, %s, %s, %s, %s, %s, %s, %s")
    end = time.time()
    return "Prediction done. Time Taken "+str(end-start)


ans=True
while ans:
    print ("""
    1.KNN on dev set
    2.KNN on train set
    3.KNN on dev set (Manhattan)
    4.KNN on train set (Manhattan)
    5.KNN all binarized dev (Euclidean)
    6.KNN all binarized train (Euclidean)
    7.Predict Test
    8.Exit/Quit
    """)
    ans=input("What would you like to do? ") 
    if ans=="1": 
        print("\n KNN on dev set") 
        print(dev_euclidean())
    elif ans=="2":
        print("\n KNN on train set") 
        print(train_euclidean())
    elif ans=="3":
        print("\n KNN on dev set (Manhattan)") 
        print(dev_manhattan())
    elif ans=="4":
        print("\n KNN on train set (Manhattan)") 
        print(train_manhattan())
    elif ans=="5":
        print("\n KNN all binarized dev (Euclidean)")
        print(dev_all_binarized())
    elif ans=="6":
        print("\n KNN all binarized train (Euclidean)")
        print(train_all_binarized())
    elif ans=="7":
        print("\n Predicting for test")
        print(predict_blind())
    elif ans=="8":
        ans=False
    elif ans !="":
        print("\n Not Valid Choice Try again")
