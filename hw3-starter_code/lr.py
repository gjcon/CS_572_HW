#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt
import numpy as np

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = np.array([0.0] * numvars)
    b = 0.0
    #
    # YOUR CODE HERE
    #

    for i in range(0, MAX_ITERS):
        loss = 0
        dw = np.zeros_like(w)
        db = 0
        for row in data:                    #look at each row of the data (i.e., each test mushroom)
            x = np.array(row[:-1])          #x is all the data in the row, excluding the classification column
            x = x.flatten()
            y = row[-1]

            wx = np.dot(w,x)

            #loss += np.log10(1 + np.exp(-y*(wx+b)))
            dw += -(y*x/(1+np.exp(y*(wx+b))))
            db += -(y/(1+np.exp(y*(wx+b))))

        #loss += l2_reg_weight / 2 * np.dot(w, w)

        gradient = np.sqrt(np.sum(dw**2) + db**2)
        if gradient < 0.0001:
            print('Converged after '+str(i+1)+' iterations.')
            break

        w = w - eta*(dw + l2_reg_weight * w)
        b = b - eta*db

    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model

    #
    # YOUR CODE HERE
    #
    Py1 = 1/(1+np.exp(-np.dot(w,x)-b))

    return Py1 # This is an random probability, fix this according to your solution


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
