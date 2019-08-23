import numpy as np
import math as m
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
		if abs(err - prev_err) < 0.0000001:
			break
		else:
			prev_err = err
			itr = itr + 1
	return w


data_set_size = 10

#Generate the data set
X_val = np.random.uniform(size=data_set_size)
Y_val = []
noise = np.random.normal(0, 0.3, data_set_size)

for i in range(data_set_size):
	Y_val.append(m.sin(2*m.pi*X_val[i]) + noise[i])


print("The generated data set is:")

for i in range(data_set_size):
    print(X_val[i],Y_val[i])
print('\n')


#Split the dataset into training set and teting set
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

print("The training data set is:")
for i in range(len(train_X_val)):
    print(train_X_val[i],train_Y_val[i])
print('\n')

for i in rand_arr[limit:data_set_size]:
	test_X_val.append(X_val[i])
	test_Y_val.append(Y_val[i])

print("The test data set is:")
for i in range(len(test_X_val)):
    print(test_X_val[i],test_Y_val[i])
print('\n')

#Run the linear regression function with polynomial degree 1
a1 = []
a1 = linear_reg(1,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 1 polynomial are:")
print(a1)
print('\n')

#Run the linear regression function with polynomial degree 2
a2 = []
a2 = linear_reg(2,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 2 polynomial are:")
print(a2)
print('\n')

#Run the linear regression function with polynomial degree 3
a3 = []
a3 = linear_reg(3,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 3 polynomial are:")
print(a3)
print('\n')

#Run the linear regression function with polynomial degree 4
a4 = []
a4 = linear_reg(4,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 4 polynomial are:")
print(a4)
print('\n')

#Run the linear regression function with polynomial degree 5
a5 = []
a5 = linear_reg(5,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 5 polynomial are:")
print(a5)
print('\n')

#Run the linear regression function with polynomial degree 6
a6 = []
a6 = linear_reg(6,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 6 polynomial are:")
print(a6)
print('\n')

#Run the linear regression function with polynomial degree 7
a7 = []
a7 = linear_reg(7,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 7 polynomial are:")
print(a7)
print('\n')

#Run the linear regression function with polynomial degree 8
a8 = []
a8 = linear_reg(8,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 8 polynomial are:")
print(a8)
print('\n')

#Run the linear regression function with polynomial degree 9
a9 = []
a9 = linear_reg(9,0.05,train_X_val,train_Y_val,max_iteration)
print("The learned coefficients for degree 9 polynomial are:")
print(a9)
print('\n')


train_error = []
test_error = []

#Calculate the training and test error values for different degree polynolmials

train_error.append(error(1,train_X_val,train_Y_val,a1))
test_error.append(error(1,test_X_val,test_Y_val,a1))

train_error.append(error(2,train_X_val,train_Y_val,a2))
test_error.append(error(2,test_X_val,test_Y_val,a2))

train_error.append(error(3,train_X_val,train_Y_val,a3))
test_error.append(error(3,test_X_val,test_Y_val,a3))

train_error.append(error(4,train_X_val,train_Y_val,a4))
test_error.append(error(4,test_X_val,test_Y_val,a4))

train_error.append(error(5,train_X_val,train_Y_val,a5))
test_error.append(error(5,test_X_val,test_Y_val,a5))

train_error.append(error(6,train_X_val,train_Y_val,a6))
test_error.append(error(6,test_X_val,test_Y_val,a6))

train_error.append(error(7,train_X_val,train_Y_val,a7))
test_error.append(error(7,test_X_val,test_Y_val,a7))

train_error.append(error(8,train_X_val,train_Y_val,a8))
test_error.append(error(8,test_X_val,test_Y_val,a8))

train_error.append(error(9,train_X_val,train_Y_val,a9))
test_error.append(error(9,test_X_val,test_Y_val,a9))

for i in range(9):
    print("The train error on train data set for n =",i+1,'is:',train_error[i])
    print("The test error on test data set for n =",i+1,'is:',test_error[i],'\n')
