import numpy as np
import os
import timeit
import matplotlib.pyplot as plt
import sys   
import pprint 
import pandas as pd
import math


# load data1  (Huge data)
mnist = pd.read_csv("mnist_test.csv")
labels = mnist.label
labels = labels.to_numpy()
#print(labels)
input_data = mnist.loc[:,mnist.columns != 'label']
#print(input_data)
X_huge = input_data.to_numpy()
#print(X_huge)



# load data2 (lesser size data)
X = pd.read_csv("X.csv")
#print(X)
X = X.to_numpy()
y = pd.read_csv("y.csv")
y = y.to_numpy()
#print(np.shape(y))

# change y in range of 0 to 9
for i in range(len(y)):
	if y[i] == 10 :
		y[i] = 0



## Data variables 
def Data_var(X):
	m = len(X) #no.of examples
	n_f = len(X[0])  #no.of features
	return m , n_f

## NN architecture - should be configurable
def NN_arch(X,no_of_classes,):
	input_units = len(X[0])
	output_units = no_of_classes
	return input_units , output_units

def sigmoid(z):
	g = 1 / (1 + np.exp(-z))
	return g

def sigmoid_diff(z):
	a = sigmoid(z)
	g = a.dot(1-a)
	return g


## weight(Theta) initilisation
def random_init(s , s1):
	eps = math.sqrt(6) / math.sqrt(s + s1)
	W = np.random.rand(s1, s+1) * (2 * eps) - eps
	return W

def weight_init(input_units,output_units,hidden_layers,hidden_units) :
	Theta_count = 1 + hidden_layers
	Theta = [ [] for i in range(Theta_count) ]

	s = input_units
	s1 = hidden_units
	Theta[0] = random_init(s,s1)

	for l in range(1 , Theta_count-1):
		s = hidden_units
		s1 = s
		Theta[l] = random_init(s,s1)
	
	s = hidden_units
	s1 = output_units
	last = Theta_count - 1
	Theta[last] = random_init(s,s1)

	return Theta


## forward propogation calculation
def forward(X,Theta,input_units,output_units,hidden_layers,hidden_units,m):
	#m = len(X) #no.of examples
	total_layers = 1 + hidden_layers + 1
	a = [[] for i in range(total_layers)]
	z = [[] for i in range(total_layers)]
	# Theta size consits of thetas between different layers (3 dimensional matrix)

	#input layer
	a[0] = X #a1
	z[0] = a[0]
	if(m != 1):
		a[0] = np.append( np.ones((m,1)) , a[0] ,axis = 1)  # adding ones at start of matrix
	if(m == 1):
		a[0] = np.append( np.ones(m) , a[0] ,axis = 0) 


	# hidden layers
	for l in range(1 , total_layers-1):
		z[l] = a[l-1].dot( np.transpose(Theta[l-1]) )
		a[l] = sigmoid(z[l])
		if(m != 1):
			a[l] = np.append( np.ones((m,1)) , a[l] ,axis = 1)
		if(m == 1):
			a[l] = np.append( np.ones(m) , a[l] ,axis = 0)

	# output layer
	last = total_layers-1
	z[last] = a[last-1].dot( np.transpose(Theta[last-1]))
	a[last] = sigmoid(z[last])
	return a

def vector_labels(y,num_labels):
	m = len(y)
	Y = np.zeros((m,num_labels))
	for i in range(m):
		a = y[i]
		Y[i][a] = 1
	return Y


def cost_func(X,Y,init_Theta, input_units,output_units,hidden_layers,hidden_units) :
	last = 1 + hidden_layers 
	m = len(X)
	a = forward(X,init_Theta,input_units,output_units,hidden_layers,hidden_units,m)
	h = a[last] 

	f1 = ( np.transpose(np.log(h)) ).dot(Y )
	J1 = np.sum(f1)
	f2 = ( np.transpose(np.log(1-h)) ).dot(1-Y)
	J2 = np.sum(f2)
	J = (-1/m) * (J1 + J2)
	return J


def back_propogation(X , Y , Theta, input_units,output_units,hidden_layers,hidden_units):
	m = len(X)
	total_layers = 1 + hidden_layers + 1
	last = 1 + hidden_layers
	
	# gradient initialisation
	DEL = [[] for i in range(len(Theta)) ]
	for i in range(len(Theta)) :
		dim = np.shape(Theta[i])
		DEL[i] = np.zeros(dim)
	
	for t in range(m):
		error = [[] for i in range(total_layers)]

		# forward prop
		x = X[t]
		a = forward(x,Theta,input_units,output_units,hidden_layers,hidden_units,1)
		h = a[last] 

		#output layer error
		#print(np.shape(Y[t]))
		error[last] = a[last] - Y[t]

		# hidden layer errors
		for l in reversed(range(1,last)):
			# errors
			der = (1-a[l]).dot(a[l]) # sigmoid gradient
			error[l] = error[l+1].dot(Theta[l])
			error[l] = error[l].dot(np.transpose(der)) 
			c = error[l]
			d = np.delete(c,0 , axis = 0)
			error[l] = d

		# gradients
		for l in reversed(range(0,last)):
			e_r = (error[l+1]).reshape( len(error[l+1]) , 1 )
			a_r = (a[l]).reshape( 1 , len(a[l]))
			er = e_r.dot(a_r)
			DEL[l] =DEL[l] + er

	for i in range(len(Theta)) :
		DEL[i] = (1/m) * DEL[i]

	return DEL


def gradient_descent(X,Y,Theta , input_units,output_units,hidden_layers,hidden_units,epochs,alpha ) :
	for t in range(epochs):
		J_prev = cost_func(X,Y,Theta, input_units,output_units,hidden_layers,hidden_units)
		Grad = back_propogation(X , Y , Theta, input_units,output_units,hidden_layers,hidden_units)
		#print(Grad)
		Theta = Theta - (alpha * np.array(Grad))
		J = cost_func(X,Y,Theta, input_units,output_units,hidden_layers,hidden_units)
		if(J > J_prev):
			break
		print("epoch no " , t ," :" , "cost functn = ",J)

	print("Final cost function = ",J_prev,"--> obtained at epoch",t)
	return Theta

def predictions(X,Theta , input_units,output_units,hidden_layers,hidden_units) : 
	m = len(X)
	a = forward(X,Theta,input_units,output_units,hidden_layers,hidden_units,m)
	last = 1 + hidden_layers
	h = a[last]
	pred = np.argmax(h , axis = 1) # max probability
	return pred



def main() :

	# NN and hidden layer - configuration
	k = 10 # no.of classes
	input_units , output_units = NN_arch(X,k)
	hidden_layers = 1   #should be changed accordingly 
	hidden_units = 25   #should be changed accordingly

	
	total_layers = 1 + hidden_layers + 1
	last = total_layers-1

	m ,n_f = Data_var(X)
	print(m)
	print(n_f)

	# intial forward propgation and predict values
	init_Theta =  weight_init(input_units,output_units,hidden_layers,hidden_units)
	a = forward(X,init_Theta,input_units,output_units,hidden_layers,hidden_units,len(X))
	h = a[last] # predict values
	print(h)
	print(np.shape(h))
	#print(y)
	#print(np.shape(y))
	print(init_Theta)


	#vector form of labels
	Y = vector_labels(y,k)
	#print(Y)
	print(np.shape(Y))


	#initial cost functn
	J = cost_func(X,Y,init_Theta, input_units,output_units,hidden_layers,hidden_units) 
	print(J)


	#BP - grad
	Grad = back_propogation(X , Y , init_Theta, input_units,output_units,hidden_layers,hidden_units)
	#print(Grad)


	#Gradient descent
	epochs = 150 #configurable
	alpha = 0.01 #configurable
	start = timeit.default_timer()
	final_Theta = gradient_descent(X,Y,init_Theta , input_units,output_units,hidden_layers,hidden_units,epochs,alpha )
	print("------------")
	#print(final_Theta)
	stop = timeit.default_timer()
	print('Time taken for gradient descent', stop - start , "secs")


	# Predictions
	predict = predictions(X,final_Theta , input_units,output_units,hidden_layers,hidden_units)
	print(predict)
	print(np.shape(predict))

	# Accuracy
	check = np.zeros(m)
	for i in range(m):
		if(predict[i] == y[i]):
			check[i] = 1
	Accuracy = np.mean(check) * 100.0
	print("Accuracy percentage on training set = ",Accuracy)


if __name__ == '__main__':
	main()