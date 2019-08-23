import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as rn

max_iteration = 10000

def get_summation_value(n, cur_n, X_val, Y_val, w):
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

def get_error_value(n, X_val, Y_val, w):
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

def linear_regression_fit(n, alpha, X_val, Y_val, iterations):
	w = np.zeros(n+1)
	temp_w = np.zeros(n+1)
	m = len(X_val)
	itr = 0
	prev_err = get_error_value(n, X_val,Y_val,w)
	while itr < iterations:
		for i in range(n+1):
			val = get_summation_value(n, i, X_val, Y_val, w)
			temp_w[i] = w[i] - (alpha/m)*(val)
		for i in range(n+1):
			w[i] = temp_w[i]
		err = get_error_value(n, X_val,Y_val,w)
		if itr%200 == 0:
			print(itr,err)
		if abs(err - prev_err) < 0.0000001:
			break
		else:
			prev_err = err
			itr = itr + 1
	return w


data_set_size = int(input("Enter the size of the data set: "))

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
print(len(train_X_val))
print(len(test_X_val))

plt.figure(0)
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')

a1 = []
a1 = linear_regression_fit(1,0.05,train_X_val,train_Y_val,max_iteration)
print(a1)

a2 = []
a2 = linear_regression_fit(2,0.05,train_X_val,train_Y_val,max_iteration)
print(a2)

a3 = []
a3 = linear_regression_fit(3,0.05,train_X_val,train_Y_val,max_iteration)
print(a3)

a4 = []
a4 = linear_regression_fit(4,0.05,train_X_val,train_Y_val,max_iteration)
print(a4)

a5 = []
a5 = linear_regression_fit(5,0.05,train_X_val,train_Y_val,max_iteration)
print(a5)

a6 = []
a6 = linear_regression_fit(6,0.05,train_X_val,train_Y_val,max_iteration)
print(a6)

a7 = []
a7 = linear_regression_fit(7,0.05,train_X_val,train_Y_val,max_iteration)
print(a7)

a8 = []
a8 = linear_regression_fit(8,0.05,train_X_val,train_Y_val,max_iteration)
print(a8)

a9 = []
a9 = linear_regression_fit(9,0.05,train_X_val,train_Y_val,max_iteration)
print(a9)

x = np.linspace(-2.0,2.0,1000000)
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []
y7 = []
y8 = []
y9 = []

for i in x:
	y1.append(a1[0]+a1[1]*i)
	y2.append(a2[0]+a2[1]*i+a2[2]*i*i)
	y3.append(a3[0]+a3[1]*i+a3[2]*i*i+a3[3]*i*i*i)
	y4.append(a4[0]+a4[1]*i+a4[2]*i*i+a4[3]*i*i*i+a4[4]*i*i*i*i)
	y5.append(a5[0]+a5[1]*i+a5[2]*i*i+a5[3]*i*i*i+a5[4]*i*i*i*i+a5[5]*i*i*i*i*i)
	y6.append(a6[0]+a6[1]*i+a6[2]*i*i+a6[3]*i*i*i+a6[4]*i*i*i*i+a6[5]*i*i*i*i*i+a6[6]*i*i*i*i*i*i)
	y7.append(a7[0]+a7[1]*i+a7[2]*i*i+a7[3]*i*i*i+a7[4]*i*i*i*i+a7[5]*i*i*i*i*i+a7[6]*i*i*i*i*i*i+a7[7]*i*i*i*i*i*i*i)
	y8.append(a8[0]+a8[1]*i+a8[2]*i*i+a8[3]*i*i*i+a8[4]*i*i*i*i+a8[5]*i*i*i*i*i+a8[6]*i*i*i*i*i*i+a8[7]*i*i*i*i*i*i*i+a8[8]*i*i*i*i*i*i*i*i)
	y9.append(a9[0]+a9[1]*i+a9[2]*i*i+a9[3]*i*i*i+a9[4]*i*i*i*i+a9[5]*i*i*i*i*i+a9[6]*i*i*i*i*i*i+a9[7]*i*i*i*i*i*i*i+a9[8]*i*i*i*i*i*i*i*i+a9[9]*i*i*i*i*i*i*i*i*i)

train_error = []
test_error = []
x_axis = []

for i in range(9):
	x_axis.append(i+1)

train_error.append(get_error_value(1,train_X_val,train_Y_val,a1))
test_error.append(get_error_value(1,test_X_val,test_Y_val,a1))

train_error.append(get_error_value(2,train_X_val,train_Y_val,a2))
test_error.append(get_error_value(2,test_X_val,test_Y_val,a2))

train_error.append(get_error_value(3,train_X_val,train_Y_val,a3))
test_error.append(get_error_value(3,test_X_val,test_Y_val,a3))

train_error.append(get_error_value(4,train_X_val,train_Y_val,a4))
test_error.append(get_error_value(4,test_X_val,test_Y_val,a4))

train_error.append(get_error_value(5,train_X_val,train_Y_val,a5))
test_error.append(get_error_value(5,test_X_val,test_Y_val,a5))

train_error.append(get_error_value(6,train_X_val,train_Y_val,a6))
test_error.append(get_error_value(6,test_X_val,test_Y_val,a6))

train_error.append(get_error_value(7,train_X_val,train_Y_val,a7))
test_error.append(get_error_value(7,test_X_val,test_Y_val,a7))

train_error.append(get_error_value(8,train_X_val,train_Y_val,a8))
test_error.append(get_error_value(8,test_X_val,test_Y_val,a8))

train_error.append(get_error_value(9,train_X_val,train_Y_val,a9))
test_error.append(get_error_value(9,test_X_val,test_Y_val,a9))

plt.figure(1)
plt.plot(x_axis,train_error, label = 'train error')
plt.plot(x_axis,test_error, label = 'test error')
plt.legend(loc='upper right')

plt.figure(1)
plt.plot(x,y1,label='n=1')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(2)
plt.plot(x,y2,label='n=2')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(3)
plt.plot(x,y3,label='n=3')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(4)
plt.plot(x,y4,label='n=4')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(5)
plt.plot(x,y5,label='n=5')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(6)
plt.plot(x,y6,label='n=6')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(7)
plt.plot(x,y7,label='n=7')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(8)
plt.plot(x,y8,label='n=8')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')
plt.figure(9)
plt.plot(x,y9,label='n=9')
plt.scatter(train_X_val, train_Y_val, label = 'training data')
plt.scatter(test_X_val, test_Y_val, label = 'test data')
plt.legend(loc='upper right')


plt.show()
