# imports
import random
import numpy as np
import matplotlib.pyplot as plt

# Training data
def sine(a,b,c,x):
    return a * np.sin(b * x + c)

# Regression
def train_lr(x, eta):
    a = 0
    b = 0
    c = 0

    for iter in range(0, MAX_ITERS):
        da = 0
        db = 0
        dc = 0
        for i in x:                    
            my_x = x          
            my_y = sine(1,1,1,my_x) + np.random.normal(-a/10,a/10,1) #signal plus noise, actual a,b,c=1,1,1

            #loss = (my_y - ypred)**2 = (my_y - a * np.sin(b * x + c))**2
            da += 2*(my_y - sine(a,b,c,x))*(-np.cos(b*x+x))
            db += 2*(my_y - sine(a,b,c,x))*(-x*a*np.cos(b*x+c))
            dc += 2*(my_y - sine(a,b,c,x))*(-a*np.cos(b*x+c))

        gradient = np.sqrt(da**2 + db**2 + dc**2)
        if gradient < 0.0001:
            print('Converged after '+str(iter+1)+' iterations.')
            break

        a = a - eta*da
        b = b - eta*db
        c = c - eta*dc

    return (a, b, c)

# Generate data
x = np.arange(0,10,0.001)
a_true = 1
b_true = 1
c_true = 1
ytrue = sine(a_true, b_true, c_true, x) + np.random.normal(-a/10,a/10,1)
plt.plot(x, ytrue)

# Set hyperparameters
eta = 1
MAX_ITERS = 100

# Run model
pred_a, pred_b, pred_c = train_lr(x, eta)
print(pred_a, pred_b, pred_c)

