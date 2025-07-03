#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node
import numpy as np


train = None
varnames = None
test = None
testvarnames = None
root = None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<

    #no entropy if data is pure
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<

    #return no gain if the data or attribute is pure
    if pxi == 0 or total == 0 or total == pxi:
        return 0
    
    return entropy(py/total) - (pxi/total) * entropy(py_pxi/pxi) - (((total-pxi)/total)) * entropy((py-py_pxi)/(total-pxi)) 


# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<

    #initialize stuff
    total = len(data)
    best_gain = -1
    best_attr = None

    #Add up 0s and 1s in the last column
    count = [0, 0]
    for row in data:
        count[row[-1]] += 1
    py = count[1]

    #for each attribute, add up the cases of "attribute = 1 & poisonous = 1" and "attribute = 1", then calculate gain
    for attr in range(len(varnames) - 1):
        py_pxi = sum(1 for row in data if row[attr] == 1 and row[-1] == 1)
        pxi = sum(1 for row in data if row[attr] == 1)
        gain = infogain(py_pxi, pxi, py, total)

        #if the gain for this attribute is better than the best gain so far, set the new best gain and best attribute
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    #if the best_gain is no gain, make a new leaf with a prediction based on the total number of poisonous/non-poisonous
    if best_gain <= 0:
        prediction = 1 if count[1] > count[0] else 0
        return node.Leaf(varnames, prediction)

    #split on the new data sets that are the "left" and "right" branch below attr, with attribute=0 on the left and attribute=1 on the right
    left_data = [row for row in data if row[best_attr] == 0]
    right_data = [row for row in data if row[best_attr] == 1]

    #build_tree again based on the new data sets acquired after splitting on that attribute
    left = build_tree(left_data, varnames)
    right = build_tree(right_data, varnames)

    #once we've exhausted the info gain we can get, save the split.
    return node.Split(varnames, best_attr, left, right)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
