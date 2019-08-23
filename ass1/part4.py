import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as rn

max_iteration = 1000

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

def get_summation_value_mod(n, cur_n, X_val, Y_val, w):
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
		if temp_val < 0:
			temp_val = -1
		else:
			temp_val = 1
		val = val + temp_val*power_val2
	return val

def get_error_value_mod(n, X_val, Y_val, w):
	val = 0
	for i in range(len(X_val)):
		temp_val = w[0]
		power_val1 = 1
		for j in range(1,n+1):
			power_val1 = power_val1*X_val[i]
			temp_val = temp_val + w[j]*power_val1
		temp_val = temp_val - Y_val[i]
		temp_val = abs(temp_val)
		val = val + temp_val
	return val/(2*len(X_val))

def linear_regression_fit_mod(n, alpha, X_val, Y_val, iterations):
	w = np.random.uniform(0,1,n+1)
	temp_w = np.zeros(n+1)
	m = len(X_val)
	itr = 0
	prev_err = get_error_value_mod(n, X_val,Y_val,w)
	while itr < iterations:
		for i in range(n+1):
			val = get_summation_value_mod(n, i, X_val, Y_val, w)
			# print("VAL = ",val)
			# print(temp_w[i])
			temp_w[i] = w[i] - (alpha/2*m)*(val)
			# print(temp_w[i])
		# print("Before",w)
		for i in range(n+1):
			w[i] = temp_w[i]
		# print("After",w)
		err = get_error_value_mod(n, X_val,Y_val,w)
		# if itr%200 == 0:
			# print(w)
			# print(itr,err)
		if abs(err - prev_err) < 0.000001:
			break
		else:
			prev_err = err
			itr = itr + 1
	return w

def get_summation_value_4(n, cur_n, X_val, Y_val, w):
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
		temp_val = temp_val*temp_val*temp_val
		val = val + temp_val*power_val2
	return val

def get_error_value_4(n, X_val, Y_val, w):
	val = 0
	for i in range(len(X_val)):
		temp_val = w[0]
		power_val1 = 1
		for j in range(1,n+1):
			power_val1 = power_val1*X_val[i]
			temp_val = temp_val + w[j]*power_val1
		temp_val = temp_val - Y_val[i]
		temp_val = temp_val*temp_val*temp_val*temp_val
		val = val + temp_val
	return val/(2*len(X_val))

def linear_regression_fit_4(n, alpha, X_val, Y_val, iterations):
	w = np.random.uniform(0,1,n+1)
	temp_w = np.zeros(n+1)
	m = len(X_val)
	itr = 0
	prev_err = get_error_value_4(n, X_val,Y_val,w)
	while itr < iterations:
		for i in range(n+1):
			val = get_summation_value_4(n, i, X_val, Y_val, w)
			temp_w[i] = w[i] - (2*alpha/m)*(val)
		for i in range(n+1):
			w[i] = temp_w[i]
		err = get_error_value_4(n, X_val,Y_val,w)
		if itr%200 == 0:
			print(itr,err)
		if abs(err - prev_err) < 0.000001:
			break
		else:
			prev_err = err
			itr = itr + 1
	return w

def rmse(n,X_val,Y_val,w):
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
	val = val/len(X_val)
	val = m.sqrt(val)
	return val

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

normal_cost_function = []
mod_cost_function = []
power4_cost_function = []

learning_rates = [0.025, 0.05, 0.1, 0.2, 0.5]
a = []

# Normal
a = linear_regression_fit(3,0.025,train_X_val,train_Y_val,max_iteration)
normal_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit(3,0.05,train_X_val,train_Y_val,max_iteration)
normal_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit(3,0.1,train_X_val,train_Y_val,max_iteration)
normal_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit(3,0.2,train_X_val,train_Y_val,max_iteration)
normal_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit(3,0.5,train_X_val,train_Y_val,max_iteration)
normal_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

# Mod
a = linear_regression_fit_mod(3,0.025,train_X_val,train_Y_val,max_iteration)
mod_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_mod(3,0.05,train_X_val,train_Y_val,max_iteration)
mod_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_mod(3,0.1,train_X_val,train_Y_val,max_iteration)
mod_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_mod(3,0.2,train_X_val,train_Y_val,max_iteration)
mod_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_mod(3,0.5,train_X_val,train_Y_val,max_iteration)
mod_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

# Power 4
a = linear_regression_fit_4(3,0.025,train_X_val,train_Y_val,max_iteration)
power4_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_4(3,0.05,train_X_val,train_Y_val,max_iteration)
power4_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_4(3,0.1,train_X_val,train_Y_val,max_iteration)
power4_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_4(3,0.2,train_X_val,train_Y_val,max_iteration)
power4_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

a = linear_regression_fit_4(3,0.5,train_X_val,train_Y_val,max_iteration)
power4_cost_function.append(rmse(3,test_X_val,test_Y_val,a))
print(a)

plt.plot(learning_rates,normal_cost_function,label='Normal')
plt.plot(learning_rates,mod_cost_function,label='Mod')
plt.plot(learning_rates,power4_cost_function,label='Power 4')
plt.legend(loc='best')
plt.show()
