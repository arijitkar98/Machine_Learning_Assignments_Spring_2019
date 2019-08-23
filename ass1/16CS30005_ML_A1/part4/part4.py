import numpy as np
import math
import matplotlib.pyplot as plt
import random as rn

max_iteration = 1000

def calc_val(n, cur_n, X_val, Y_val, w, method):
	val = 0
	for i in range(len(X_val)):
		y_pred = w[0]
		X_power_1 = 1
		X_power_2 = 1
		for j in range(1,n+1):
			X_power_1 = X_power_1*X_val[i]
			y_pred = y_pred + w[j]*X_power_1
		for j in range(1,cur_n+1):
			X_power_2 = X_power_2*X_val[i]
		y_pred = y_pred - Y_val[i]
		if (method == 1):
			if y_pred < 0:
				y_pred = -1/((len(X_val)))
			else:
				y_pred = 1/((len(X_val)))
		elif (method == 2):
			y_pred = y_pred*y_pred*(y_pred/len(X_val))/200
		else:
			y_pred = y_pred/len(X_val)
		val = val + y_pred*X_power_2
	return val

def error(n, X_val, Y_val, w, method):
	val = 0
	for i in range(len(X_val)):
		y_pred = w[0]
		X_power_1 = 1
		for j in range(1,n+1):
			X_power_1 = X_power_1*X_val[i]
			y_pred = y_pred + w[j]*X_power_1
		y_pred = y_pred - Y_val[i]
		if (method == 0):
			y_pred = y_pred*(y_pred/(2*len(X_val)))
		elif (method == 1):
			y_pred = math.fabs(y_pred)/((2*len(X_val)))
		else:
			y_pred = y_pred*y_pred*y_pred*(y_pred/(2*len(X_val)))/200		
		val = val + y_pred
	return val

def linear_reg(n, alpha, X_val, Y_val, iterations, method):
	w = np.random.uniform(0,1,n+1)
	temp_w = np.zeros(n+1)
	m = len(X_val)
	itr = 0
	prev_err = error(n,X_val,Y_val,w,method)
	while itr < iterations:
		for i in range(n+1):
			val = calc_val(n,i,X_val,Y_val,w,method)
			if (method == 0):
				temp_w[i] = w[i] - (alpha)*(val)
			elif (method == 1):
				temp_w[i] = w[i] - (alpha)*(val)
			else:
				temp_w[i] = w[i] - (2*alpha)*(val)
		for i in range(n+1):
			w[i] = temp_w[i]
		err = error(n,X_val,Y_val,w,method)
		if itr%200 == 0:
			print("Iteration =",itr,"Train Error =",err)
		if abs(err - prev_err) < 0.00001:
			break
		else:
			prev_err = err
			itr = itr + 1
	return w

def rms_error(n, X_val, Y_val, w):
	val = 0
	for i in range(len(X_val)):
		y_pred = w[0]
		X_power_1 = 1
		for j in range(1,n+1):
			X_power_1 = X_power_1*X_val[i]
			y_pred = y_pred + w[j]*X_power_1
		y_pred = y_pred - Y_val[i]
		y_pred = y_pred*y_pred
		val = val + y_pred
	val = val/len(X_val)
	val = math.sqrt(val)
	return val

data_set_size = int(input("Enter the size of the data set: "))

X_val = np.random.uniform(0,1,size=data_set_size)
Y_val = []
noise = np.random.normal(0, 0.3, data_set_size)

for i in range(data_set_size):
	Y_val.append(math.sin(2*math.pi*X_val[i]) + noise[i])

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

MSE_cost_function = []
MAE_cost_function = []
Power4_cost_function = []

learning_rates = [0.025, 0.05, 0.1, 0.2, 0.5]
a = []

# MSE
for i in learning_rates:
	a = linear_reg(5,i,train_X_val,train_Y_val,max_iteration,0)
	MSE_cost_function.append(rms_error(5,test_X_val,test_Y_val,a))
	print(a)

# MAE
for i in learning_rates:
	a = linear_reg(5,i,train_X_val,train_Y_val,max_iteration,1)
	MAE_cost_function.append(rms_error(5,test_X_val,test_Y_val,a))
	print(a)

# Power 4
for i in learning_rates:
	a = linear_reg(5,i,train_X_val,train_Y_val,max_iteration,2)
	Power4_cost_function.append(rms_error(5,test_X_val,test_Y_val,a)/2)
	print(a)

plt.plot(learning_rates,MSE_cost_function,label='MSE')
plt.plot(learning_rates,MAE_cost_function,label='MAE')
plt.plot(learning_rates,Power4_cost_function,label='Power 4')
plt.legend(loc='best')
plt.show()
