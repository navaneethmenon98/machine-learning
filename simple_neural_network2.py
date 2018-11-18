import numpy as py

def nonlinearfunc (x, d=False):
	if (d==True):
		return x*(1-x)
	return (1/(1+py.exp(-x)))

z= py.array ([[0,0,0],
	      [0,0,1],
	      [0,1,0],
	      [0,1,1],
	      [1,0,0],
	      [1,0,1],
	      [1,1,0],
	      [1,1,1]])

y= py.array([[0,1,0,1,0,1,0,1]]).T

py.random.seed(1)

synapse0 = 2*py.random.random((3,1)) - 1

for i in range(100000):
	l0 = z
	l1 = nonlinearfunc ( py.dot(l0, synapse0))

	l1_error = y -l1

	l1_delta = l1_error* nonlinearfunc( l1,True)

	synapse0+= py.dot(l0.T,l1_delta)

print (" Output after training ")

print (l1)

