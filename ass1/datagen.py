import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as rn

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
			print("Iteration =",itr,"Error =",err)
		if abs(err - prev_err) < 0.000000001:
			break
		else:
			prev_err = err
		itr = itr+1
	return w

def p(w, x):
	val = w[0]
	for i in range(1,len(w)):
		val = val + w[i]*(x**i)
	return val

#Part 1a
X_val = np.random.uniform(size=1000)
Y_val = []
noise = np.random.normal(0, 0.3, 1000)

for i in range(1000):
	Y_val.append(m.sin(2*m.pi*X_val[i]) + noise[i])

# print(X_val)
# print(Y_val)

plt.scatter(X_val, Y_val)
# plt.show()

#Part 1b
rand_arr = np.arange(1000)
rn.shuffle(rand_arr)

train_X_val = []
train_Y_val = []

for i in rand_arr[0:800]:
	train_X_val.append(X_val[i])
	train_Y_val.append(Y_val[i])

test_X_val = []
test_Y_val = []

for i in rand_arr[800:1000]:
	test_X_val.append(X_val[i])
	test_Y_val.append(Y_val[i])
# X_val = [1,2,3,4,5]
# Y_val = [1,1.2,3,7,7]
a = []
a = linear_regression_fit(5,0.05,train_X_val,train_Y_val,10000)

print(a)

train_error = get_error_value(5,train_X_val,train_Y_val,a)
test_error = get_error_value(5,test_X_val,test_Y_val,a)
print("Train Error =",train_error)
print("Test Error =",test_error)
# x = np.linspace(-2.0,2.0,1000000)
# y = []
# for i in x:
# 	y.append(a[0] + a[1]*i + a[2]*i*i + a[3]*i*i*i + a[4]*i*i*i*i + a[5]*i*i*i*i*i)
	 # + a[6]*i*i*i*i*i*i + a[7]*i*i*i*i*i*i*i + a[8]*i*i*i*i*i*i*i*i + a[9]*i*i*i*i*i*i*i*i*i)
# y = p(a,x)
# plt.plot(x,y)
# plt.show()
