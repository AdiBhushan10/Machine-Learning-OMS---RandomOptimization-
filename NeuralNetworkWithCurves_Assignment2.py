from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import time
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

randomSeed = 11

MyFrame = pd.read_csv('C:/Users/User2/Desktop/GATECH/Machine Learning/Datasets/Dataset1 - Assignment 1/ExoPlanetData.csv')
MyFrame['Detected_ExoPlanet'] = MyFrame['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)
MyFrame.drop(columns=['Object_Name','DispositionUsingKeplerData'], inplace=True)  # Dropping columns that are not required for classification such as name of the observed celestial object 
MyFrame.fillna(MyFrame.median(), inplace=True)  # Handling missing values via median()
print(MyFrame.shape)

# Independent features and target feature
features = MyFrame.drop(columns=['Detected_ExoPlanet'])
target = MyFrame['Detected_ExoPlanet']
print(target.value_counts())

#Test Train Split: 75-25
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=randomSeed, test_size=0.25)
print("Shape of test set:", X_test.shape)
print("Shape of training set:", X_train.shape)
#Feature Scaling
X_train = scale(X_train)
X_test = scale(X_test)

#Neural Network performance using Backpropagation and Schoastic Gradient Descent
from sklearn.neural_network import MLPClassifier
MyTree = MLPClassifier(random_state=randomSeed, hidden_layer_sizes=70, activation='logistic', solver = 'adam') 
t1 = time.time()
MyTree.fit(X_train,y_train)
predicting = MyTree.predict(X_test)
from sklearn.metrics import f1_score
score_test=f1_score(y_test,predicting)
print(f'Accuracy using SGD(w/ BP) optimization: {score_test}')
dt_SGD = time.time()-t1
print("Time Taken: {}".format(dt_SGD))

train_sizes = range(1,111,20)
_, train_scores, test_scores = learning_curve(MyTree, X_train, y_train, train_sizes=train_sizes)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score',c='brown')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score',c='green')
title = "Learning Curve for NN: ExoPlanet Identification"
plt.title(title)
plt.xlabel('Percentage of training data')
plt.ylabel("Model Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()

# Weights Optimization
method = mlrose_hiive.NeuralNetwork(hidden_nodes=[70], activation='sigmoid',
                                    algorithm='random_hill_climb', max_iters=1000,
                                    bias=True, is_classifier=True, learning_rate=0.1,
                                    early_stopping=True, clip_max=5, max_attempts=100,
                                    random_state=randomSeed)
print("Starting with Random Restart Hill Climbing method to optimize weights")
t = time.time()
method.fit(X_train, y_train)
y_test_pred = method.predict(X_test)
y_test_accuracy = f1_score(y_test, y_test_pred)
print("Time Taken: {}".format(time.time()-t))
print("Accuracy Score: {}".format(y_test_accuracy))

method = mlrose_hiive.NeuralNetwork(hidden_nodes=[70], activation='sigmoid',
                                   algorithm='simulated_annealing', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=0.1,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=randomSeed)
print("Starting with Simulated Annealing method  to optimize weights")
t = time.time()
method.fit(X_train, y_train)
y_test_pred = method.predict(X_test)
y_test_accuracy = f1_score(y_test, y_test_pred)
print("Time Taken: {}".format(time.time()-t))
print("Accuracy Score: {}".format(y_test_accuracy))

method = mlrose_hiive.NeuralNetwork(hidden_nodes=[70], activation='sigmoid',
                                   algorithm='genetic_alg', max_iters=500,
                                   bias=True, is_classifier=True, learning_rate=0.1,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=randomSeed)
print("Starting with Genetic Algorithm method to optimize weights")
t = time.time()
method.fit(X_train, y_train)
y_test_pred = method.predict(X_test)
y_test_accuracy = f1_score(y_test, y_test_pred)
print("Time Taken: {}".format(time.time()-t))
print("Accuracy Score: {}".format(y_test_accuracy))

NN_list = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
max_iter = range(1,1001,100)

for alg in NN_list:
    print('\nPreparing to plot curves for the current Algorithm = {}'.format(alg))
    train_losses, train_times =[],[]
    for iter in max_iter:
        method = mlrose_hiive.NeuralNetwork(hidden_nodes=[70], activation='sigmoid',
                                        algorithm=alg, max_iters=iter,
                                        bias=True, is_classifier=True, learning_rate=1,
                                        early_stopping=True, clip_max=5, max_attempts=100,
                                        random_state=randomSeed)
        #Compute fit time
        start_time = time.time()
        method.fit(X_train, y_train)
        train_times.append(time.time() - start_time)
        #Compute Loss
        y_calc = method.predict(X_train)  # already calc above
        cross_entropy_loss = log_loss(y_train, y_calc, eps=1e-10)
        train_losses.append(cross_entropy_loss)

    loss = train_losses
    fiTime = train_times
    plt.figure()
    plt.plot(max_iter, loss, 'r', label='Cross Entropy Loss')
    plt.plot(max_iter, fiTime, 'g', label='Fit Time')
    #plt.plot()
    plt.title("Loss Function & Fit Times plotted against Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Function/Fit Time")
    plt.legend(loc='best')
    plt.grid()
    plt.show()
print("That's all for this part; thanks for your patience!!!")

"""
Output:
(9564, 19)
0    4847
1    4717
Name: Detected_ExoPlanet, dtype: int64
Shape of test set: (2391, 18)    
Shape of training set: (7173, 18)
Accuracy using SGD(w/ BP) optimization: 0.8854122621564482
Time Taken: 13.6223726272583
Starting with Random Restart Hill Climbing method to optimize weights
Time Taken: 50.66680598258972
Accuracy Score: 0.8857985672144963
Starting with Simulated Annealing method  to optimize weights
Time Taken: 68.44307398796082
Accuracy Score: 0.7824773413897281
Starting with Genetic Algorithm method to optimize weights
Time Taken: 739.7300188541412
Accuracy Score: 0.7677008750994432

Preparing to plot curves for the current Algorithm = random_hill_climb

Preparing to plot curves for the current Algorithm = simulated_annealing

Preparing to plot curves for the current Algorithm = genetic_alg
"""
   