import numpy as np
import sklearn.neural_network as sknn
from sknn.mlp import Classifier , Layer
import csv

file = open('train.csv' ,'r')
train_data = csv.reader(file , delimiter=',')
x=[np.delete(k,0).astype(int) for k in train_data if k[0]!='label']
x=np.array(x)
x= np.array([k.reshape(784,1) for k in x])

file = open('train.csv' ,'r')
train_data = csv.reader(file , delimiter=',')
y = np.array([int(k[0]) for k in train_data if k[0]!='label'])

file = open('test.csv' , 'r')
test_data = csv.reader(file , delimiter=',')
x_test = np.array([np.array(k).astype(int) for k in test_data if k[0]!='pixel0'])
## Please normalize the test data 

nn = Classifier(layers=[Layer("Softmax" , units=1000),Layer("Softmax",units=10)] , n_iter=1000 , learning_rate=0.01)