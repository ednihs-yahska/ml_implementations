from __future__ import division

import sys
import time
from svector import svector
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import collections

path_prefix = "hw4-data/"
avg_model = None
train_cache = []
dev_cache = []
best_model = None

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def read_cache(cache):
    for (label, words) in cache:
        yield (label, words)

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    return v

def bias_test(devfile, model): 
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
            
def bias_train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            sent += {"bias": 1}
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
        dev_err = bias_test(devfile, model)
        best_err = min(best_err, dev_err)
        print ("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print ("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

def avg_test(devfile, model): 
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
            
def avg_train(trainfile, devfile, epochs=5, threshold=0):
    t = time.time()
    best_err = 1.
    model = svector()
    model_a = svector()
    c = 0
    bag = svector()
    for i, (label, words) in enumerate(read_from(trainfile), 1):
        bag += make_vector(words)
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            words = [w for w in words if bag[w] > threshold]
            sent = make_vector(words)
            sent += {"bias": -1}
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_a += c * label * sent
            c+=1
        dev_err = avg_test(devfile, (c*model) - model_a)
        if dev_err < best_err:
            avg_model = model
        best_err = min(best_err, dev_err)
        print ("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print ("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

def cache_test(devfile, model): 
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_cache(dev_cache), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
            
def cache_train(trainfile, devfile, epochs=5, threshold=0):
    t = time.time()
    best_err = 1.
    model = svector()
    model_a = svector()
    c = 0
    bag = svector()
    for i, (label, words) in enumerate(read_cache(train_cache), 1):
        bag += make_vector(words)
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            words = [w for w in words if bag[w] > threshold]
            sent = make_vector(words)
            sent += {"bias": -1}
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_a += c * label * sent
            c+=1
        dev_err = avg_test(devfile, (c*model) - model_a)
        if dev_err < best_err:
            avg_model = model
        best_err = min(best_err, dev_err)
        print ("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print ("best dev err %.1f%%, |w|=%d, time: %f secs" % (best_err * 100, len(model), time.time() - t))

def exp_test(devfile, model, word_hash): 
    tot, err = 0, 0
    m=0
    labels = []
    bag = svector()
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        labels.append(label)
        m+=1
        bag += make_vector(words)
    #word_hash = {word: i for i, (word, count) in enumerate(bag.items())}
    dense_dev = np.zeros((m, len(word_hash)+1))
    dense_dev[:, -1] = -1
    for i, (label, words) in enumerate(read_from(devfile), 1):
        for w in words:
            if w in word_hash.keys():
                dense_dev[i-1, word_hash[w]] = 1
    y_Hat = model.predict(dense_dev)
    return accuracy_score(np.array(labels).reshape(-1,1), y_Hat)  # i is |D| now

def exp_train(trainfile, devfile, epochs=1, option=1):
    t = time.time()
    best_err = 1.
    model = svector()
    model_a = svector()
    m = 0
    c = 0
    bag = svector()
    for i, (label, words) in enumerate(read_from(trainfile), 1):
        m+=1
        bag += make_vector(words)
    word_hash = {word: i for i, (word, count) in enumerate(bag.items())}
    dense_train = np.zeros((m, len(word_hash)+1))
    dense_train[:, -1] = -1
    threshold = 1
    labels = []
    for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
        labels.append(label)
        words = [w for w in words if bag[w] > threshold]
        for w in words:
            dense_train[i-1, word_hash[w]] = 1
    for it in range(1, epochs+1):
        if option == 1:
            clf = DecisionTreeClassifier(random_state=0, max_depth=100)
        elif option == 2:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-8, hidden_layer_sizes=(8, 6, 4, 2), random_state=1)
        else:
            clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=300)
            best_model = clf
        clf.fit(dense_train, np.array(labels).reshape(-1,1).ravel())
        dev_err = exp_test(devfile, clf, word_hash)
        best_err = min(best_err, dev_err)
        print ("dev %.1f%%" % (100-(dev_err * 100)))
    print ("best dev err %.1f%%" % (100-(best_err * 100)))

stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

def dep_test(devfile, model, word_hash): 
    tot, err = 0, 0
    m=0
    labels = []
    bag = svector()
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        labels.append(label)
        m+=1
        words = [w for w in words if w not in stop_words]
        bag += make_vector(words)
    #word_hash = {word: i for i, (word, count) in enumerate(bag.items())}
    dense_dev = np.zeros((m, len(word_hash)+1))
    dense_dev[:, -1] = -1
    for i, (label, words) in enumerate(read_from(devfile), 1):
        for w in words:
            if w in word_hash.keys():
                dense_dev[i-1, word_hash[w]] = 1
    y_Hat = model.predict(dense_dev)
    return accuracy_score(np.array(labels).reshape(-1,1), y_Hat)  # i is |D| now

def dep_train(trainfile, devfile, epochs=1, option=1):
    t = time.time()
    best_err = 1.
    model = svector()
    model_a = svector()
    m = 0
    c = 0
    bag = svector()
    for i, (label, words) in enumerate(read_from(trainfile), 1):
        m+=1
        words = [w for w in words if w not in stop_words]
        bag += make_vector(words)
    word_hash = {word: i for i, (word, count) in enumerate(bag.items())}
    dense_train = np.zeros((m, len(word_hash)+1))
    dense_train[:, -1] = -1
    threshold = 1
    labels = []
    for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
        labels.append(label)
        words = [w for w in words if bag[w] > threshold]
        for w in words:
            dense_train[i-1, word_hash[w]] = 1
    for it in range(1, epochs+1):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-8, hidden_layer_sizes=(8, 6, 4), random_state=1)
        clf.fit(dense_train, np.array(labels).reshape(-1,1).ravel())
        dev_err = exp_test(devfile, clf, word_hash)
        best_err = min(best_err, dev_err)
        print ("dev %.1f%%" % (100-(dev_err * 100)))
    print ("best dev err %.1f%%" % (100-(best_err * 100)))

def bi_test(devfile, model): 
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        bigrams = [t[0]+" "+t[1] for t in [b for l in [" ".join(words)] for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]]
        words = bigrams
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
            
def bi_train(trainfile, devfile, epochs=5, threshold=0):
    t = time.time()
    best_err = 1.
    model = svector()
    model_a = svector()
    c = 0
    bag = svector()
    for i, (label, words) in enumerate(read_from(trainfile), 1):
        bigrams = [t[0]+" "+t[1] for t in [b for l in [" ".join(words)] for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]]
        words = bigrams
        bag += make_vector(words)
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            bigrams = [t[0]+" "+t[1] for t in [b for l in [" ".join(words)] for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]]
            words = bigrams
            words = [w for w in words if bag[w] > threshold]
            sent = make_vector(words)
            sent += {"bias": -1}
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_a += c * label * sent
            c+=1
        dev_err = avg_test(devfile, (c*model) - model_a)
        best_err = min(best_err, dev_err)
        print ("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print ("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

if __name__ == "__main__":
    for i, (label, words) in enumerate(read_from(path_prefix+sys.argv[1])):
        train_cache.append((label, words))
    
    for i, (label, words) in enumerate(read_from(path_prefix+sys.argv[2])):
        dev_cache.append((label, words))

    ans=True
    while ans==True:
        print ("""
        1.Adding bias
        2.Average Perceptron
        3.Average Perceptron 10 epochs
        4.Pruning 1 occurence words
        5.Pruning 2 occurence words
        6.Experiments
        7.Deployment
        8.Exit/Quit
        """)
        ans=input("What would you like to do? ") 
        if ans=="1": 
            print("\n Adding bias") 
            bias_train(path_prefix+"train.txt", path_prefix+"dev.txt")
            break
        elif ans=="2":
            print("\n Average Perceptron") 
            avg_train(path_prefix+"train.txt", path_prefix+"dev.txt")
            print("\n Cached Average Perceptron") 
            cache_train(path_prefix+"train.txt", path_prefix+"dev.txt")
        elif ans=="3":
            print("\n Average Perceptron 10 epochs") 
            avg_train(path_prefix+"train.txt", path_prefix+"dev.txt", epochs=10)
        elif ans=="4":
            print("\n Pruning 1 occurence words") 
            avg_train(path_prefix+"train.txt", path_prefix+"dev.txt", epochs=10, threshold=1)
        elif ans=="5":
            print("\n Pruning 2 occurence words")
            avg_train(path_prefix+"train.txt", path_prefix+"dev.txt", epochs=10, threshold=2)
        elif ans=="6":
            print("\n Experiments Decision tree")
            exp_train(path_prefix+"train.txt", path_prefix+"dev.txt", option=1)
            print("\n Experiments Multi-layer Perceptron")
            exp_train(path_prefix+"train.txt", path_prefix+"dev.txt", option=2)
            print("\n Experiments Logistic Regression")
            exp_train(path_prefix+"train.txt", path_prefix+"dev.txt", option=3)
        elif ans=="7":
            print("\n Deployment (Removing stop words)")
            dep_train(path_prefix+"train.txt", path_prefix+"dev.txt")
            print("\n Deployment (Bigrams)")
            bi_train(path_prefix+"train.txt", path_prefix+"dev.txt")
        elif ans=="8":
            ans=False
        elif ans !="":
            print("\n Not Valid Choice Try again")

        