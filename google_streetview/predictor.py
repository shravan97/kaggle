import datasplit
from sklearn.preprocessing import normalize , LabelEncoder
import sklearn.neural_network as sknn
from sknn.mlp import Classifier , Layer

def fit_network():
	x,y = datasplit.data()
	x_normalized = normalize(x,norm='l2')
	nn = Classifier(layers=[Layer("Softmax" , units=1000),Layer("Softmax" , units=62)],learning_rate=0.02,n_iter=1)
	le= LabelEncoder()
	le.fit(y)
	y = le.transform(y)
	nn.fit(x_normalized , y)
	return nn
