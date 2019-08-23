import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as rn

max_iteration = 10000

#Function for calculating summation
def calc_val(n, cur_n, X_val, Y_val, w):
	val = 0
	for i in range(len(X_val)):
		temp_val = w[0]
		power_val1 = 1
		power_val2 = 1
		for j in range(1,n+1):
			power_val1 = power_val1*X_val[i]
			temp_val = temp_val + w[j]*power_val1
		for j in range(1,cur_n+1):
			power_val2 = power_val2*X_val[i]
		temp_val = temp_val - Y_val[i]
		val = val + temp_val*power_val2
	return val

#Function fo calculating error value
def error(n, X_val, Y_val, w):
	val = 0
	for i in range(len(X_val)):
		temp_val = w[0]
		power_val1 = 1
		for j in range(1,n+1):
			power_val1 = power_val1*X_val[i]
			temp_val = temp_val + w[j]*power_val1
		temp_val = temp_val - Y_val[i]
		temp_val = temp_val*temp_val
		val = val + temp_val
	return val/(2*len(X_val))

#Function fo generating the coefficients
def linear_reg(n, alpha, X_val, Y_val, iterations):
	w = np.zeros(n+1)
	temp_w = np.zeros(n+1)
	m = len(X_val)
	itr = 0
	prev_err = error(n, X_val,Y_val,w)
	while itr < iterations:
		for i in range(n+1):
			val = calc_val(n, i, X_val, Y_val, w)
			temp_w[i] = w[i] - (alpha/m)*(val)
		for i in range(n+1):
			w[i] = temp_w[i]
		err = error(n, X_val,Y_val,w)
		if itr%200 == 0:
			print(itr,err)
		if abs(err - prev_err) < 0.0000001:
			break
		else:
			prev_err = err
			itr = itr + 1
	return w

train_error = []
test_error = []


################Dataset size = 10##################
data_set_size = 100

#Generate the data set
X_val = np.random.uniform(size=data_set_size)
Y_val = []
noise = np.random.normal(0, 0.3, data_set_size)

#Split the dataset into training set and testing set
for i in range(data_set_size):
	Y_val.append(m.sin(2*m.pi*X_val[i]) + noise[i])

rand_arr = np.arange(data_set_size)
rn.shuffle(rand_arr)

train_X_val = []
train_Y_val = []

test_X_val = []
test_Y_val = []

limit = 0.8*data_set_size
limit = int(limit)

for i in rand_arr[0:limit]:
	train_X_val.append(X_val[i])
	train_Y_val.append(Y_val[i])

for i in rand_arr[limit:data_set_size]:
	test_X_val.append(X_val[i])
	test_Y_val.append(Y_val[i])

#Plot the datasets
plt.figure(0)
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.title('Dataset size = 100')
plt.legend(loc='upper right')

#Run the linear regression function with polynomial degree 3
a3 = []
a3 = linear_reg(3,0.05,train_X_val,train_Y_val,max_iteration)
print(a3)

#Generate polynomial plot points
x = np.linspace(-2.0,2.0,1000000)
y3 = []


for i in x:
	y3.append(a3[0]+a3[1]*i+a3[2]*i*i+a3[3]*i*i*i)

#Calculate the training and test error values
train_error.append(error(3,train_X_val,train_Y_val,a3))
test_error.append(error(3,test_X_val,test_Y_val,a3))

#Plot the polynomial graph
plt.figure(1)
plt.plot(x,y3,label='n=3')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.title('Dataset size = 100')
plt.legend(loc='best')

################Dataset size = 1000##################
data_set_size = 1000

X_val = np.random.uniform(size=data_set_size)
Y_val = []
noise = np.random.normal(0, 0.3, data_set_size)


for i in range(data_set_size):
	Y_val.append(m.sin(2*m.pi*X_val[i]) + noise[i])

rand_arr = np.arange(data_set_size)
rn.shuffle(rand_arr)

train_X_val = []
train_Y_val = []

test_X_val = []
test_Y_val = []

limit = 0.8*data_set_size
limit = int(limit)

for i in rand_arr[0:limit]:
	train_X_val.append(X_val[i])
	train_Y_val.append(Y_val[i])

for i in rand_arr[limit:data_set_size]:
	test_X_val.append(X_val[i])
	test_Y_val.append(Y_val[i])


plt.figure(2)
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.title('Dataset size = 1000')
plt.legend(loc='upper right')

a3 = []
a3 = linear_reg(3,0.05,train_X_val,train_Y_val,max_iteration)
print(a3)

x = np.linspace(-2.0,2.0,1000000)
y3 = []


for i in x:
	y3.append(a3[0]+a3[1]*i+a3[2]*i*i+a3[3]*i*i*i)

train_error.append(error(3,train_X_val,train_Y_val,a3))
test_error.append(error(3,test_X_val,test_Y_val,a3))

plt.figure(3)
plt.plot(x,y3,label='n=3')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.title('Dataset size = 1000')
plt.legend(loc='best')

################Dataset size = 10000##################
data_set_size = 10000

X_val = np.random.uniform(size=data_set_size)
Y_val = []
noise = np.random.normal(0, 0.3, data_set_size)


for i in range(data_set_size):
	Y_val.append(m.sin(2*m.pi*X_val[i]) + noise[i])

rand_arr = np.arange(data_set_size)
rn.shuffle(rand_arr)

train_X_val = []
train_Y_val = []

test_X_val = []
test_Y_val = []

limit = 0.8*data_set_size
limit = int(limit)

for i in rand_arr[0:limit]:
	train_X_val.append(X_val[i])
	train_Y_val.append(Y_val[i])

for i in rand_arr[limit:data_set_size]:
	test_X_val.append(X_val[i])
	test_Y_val.append(Y_val[i])


plt.figure(4)
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.title('Dataset size = 10000')
plt.legend(loc='upper right')

a3 = []
a3 = linear_reg(3,0.05,train_X_val,train_Y_val,max_iteration)
print(a3)

x = np.linspace(-2.0,2.0,1000000)
y3 = []


for i in x:
	y3.append(a3[0]+a3[1]*i+a3[2]*i*i+a3[3]*i*i*i)

train_error.append(error(3,train_X_val,train_Y_val,a3))
test_error.append(error(3,test_X_val,test_Y_val,a3))

plt.figure(5)
plt.plot(x,y3,label='n=3')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.title('Dataset size = 10000')
plt.legend(loc='best')

x_axis = [100,1000,10000]


plt.figure(6)
plt.plot(x_axis,train_error,label='training error')
plt.plot(x_axis,test_error,label='test error')
plt.title('Error vs Dataset Size')
plt.legend(loc='best')

plt.show()
