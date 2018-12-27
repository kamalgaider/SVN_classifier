#Working of Support Vector Machine for classification
import numpy as np
from matplotlib import pyplot as plt

def svn_set_weights(X,Y):

	#Weights
	w = np.zeros(len(X[0]))
	#learning rate
	eta =1
	#training iterations
	#Also later, regulator(λ) would be used as 1/epoch 
	epochs = 100000

	#training
	for epoch in range(1, epochs):
		for i,x in enumerate(X):
			#misclassification
			if (Y[i]*np.dot(X[i], w)) < 1:
				#misclassified update for ours weights
				w = w + eta * ((X[i] * Y[i]) + (-2 *(1/epoch)* w))
			else:
				#correct classification, update our weights
				w = w + eta * (-2  *(1/epoch)* w)
	return w


def main():

	#input data points in the form : [ x, y, bias/weight ]
	X = np.array([
	[-2,4,-1],
	[4,1,-1],
	[1,6,-1],
	[2,4,-1],
	[6,2,-1]
	])

	#Associated output labels/classes for the above input points
	Y = np.array([-1, -1, 1, 1, 1])

	#Calling method to get optimum weight(used to generate hyperplane)
	w = svn_set_weights(X,Y)

	#plot the input points on 2D graph
	for d, x in enumerate(X):
		#points in class -1
		if Y[d] == -1:
			#syntax - scatter(x,y,size,symbol,linewidth used to draw)
			plt.scatter(x[0], x[1], s = 120, marker ='^', linewidths =1)
		#points in class 1
		else:
			plt.scatter(x[0], x[1], s = 120, marker ='+', linewidths =1)


	# Print the hyperplane calculated by svn_set_weights()
	x2=[w[0],w[1],-w[1],w[0]]
	x3=[w[0],w[1],w[1],-w[0]]

	x2x3 =np.array([x2,x3])
	X1,Y1,U,V = zip(*x2x3)
	ax = plt.gca()
	ax.quiver(X1,Y1,U,V,scale=1, color='blue')
	plt.xlabel('X-cordinate')
	plt.ylabel('Y-cordinate')
	plt.show()

if __name__ == '__main__':
	main()