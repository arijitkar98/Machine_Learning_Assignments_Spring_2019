import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

def relu(x, mode):
    if mode == 'forward':
        ans = x
        for i in range(len(x)):
            for j in range(len(x[i])):
                ans[i][j] = max(x[i][j],0)
        return ans
    else:
        ans = x
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] <= 0:
                    ans[i][j] = 0
                else:
                    ans[i][j] = 1
        return ans

def init_layers(nn_architecture):
    np.random.seed(12345)
    number_of_layers = len(nn_architecture)
    params_values = {}
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['Weights' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['Biases' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

def forward_nn(Weights,m,Biases,X):
    predicted_Y=list()
    threshold=0.2
    map1=dict()
    map2=dict()

    for i in range(1,(len(architecture)-1)+1):
        map1[i]=np.empty(shape=[1,architecture[i]])
        map2[i]=np.empty(shape=[1,architecture[i]])


    for n in range(len(X)):
        if(m == 0):
            continue
        else:
            map1[0]=X[n]
            for l in range(1,len(architecture)):
                map2[l]=map1[l-1].dot(Weights[l])+Biases[l]
                map1[l]=relu(map2[l],'forward')
            map2[(len(architecture)-1)]-= np.max(map2[(len(architecture)-1)])
            map1[(len(architecture)-1)] = np.exp(map2[(len(architecture)-1)]) / np.sum(np.exp(map2[(len(architecture)-1)]))
            map1[(len(architecture)-1)]=map1[(len(architecture)-1)].reshape(-1)
            if (map1[(len(architecture)-1)][0]>=map1[(len(architecture)-1)][1]):
                map1[(len(architecture)-1)]=[1,0]
            else:
                map1[(len(architecture)-1)]=[0,1]
            predicted_Y.append(map1[(len(architecture)-1)])

    predicted_Y=np.asarray(predicted_Y)
    return predicted_Y

def update_nn(params_values, grads_values, nn_architecture):

    for layer_idx, layer in enumerate(nn_architecture, 1):
        try:
            params_values["Weights" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
            params_values["Biases" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
        except:
            continue
    return params_values;

def backward_nn(delta,L,Weights,Biases,z,y,a):
    delta[L]=(a[L]-y).T
    for l in range(L-1,0,-1):
        tmp=np.asarray(np.dot(Weights[l+1],delta[l+1]))
        delta[l]=np.asarray(np.asarray(relu(z[l],'backward')).T*tmp)

    params = [Weights,Biases]
    params = update_nn(params, delta, NN_ARCHITECTURE)

    for l in range(1,L+1):
        Weights[l]=Weights[l]-learning_rate*(delta[l]*a[l-1]).T
        Biases[l]-=learning_rate*(delta[l]).T
    return delta, Weights, Biases

def weight_initilizer(Weights,Biases,L):
    for i in range(1,L+1):
        Weights[i]=np.random.randn(architecture[i-1],architecture[i])/np.sqrt(architecture[i-1])
        Biases[i]=np.random.randn(1,architecture[i])/np.sqrt(architecture[i])

    return Weights,Biases

def CE_loss(Y_val,Y_pred):
    loss = 0.0
    for i in range(len(Y_val)):
        if np.argmax(Y_val[i]) != 0:
            try:
                val=(-1.0)*np.log(Y_pred[i])
            except:
                val = 0
            loss += val
        else:
            loss+=(-1.0)*np.log(Y_pred[i])
    return (loss/len(Y_val))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def predict_nn(Weights,Biases,X):
    predicted_Y=list()
    map1=dict()
    map2=dict()

    for i in range(1,(len(architecture)-1)+1):
        map1[i]=np.empty(shape=[1,architecture[i]])
        map2[i]=np.empty(shape=[1,architecture[i]])


    for n in range(len(X)):
        map1[0]=X[n]
        forward_nn(Weights,0,Biases,X)
        for l in range(1,len(architecture)):
            map2[l]=map1[l-1].dot(Weights[l])+Biases[l]
            map1[l]=relu(map2[l],'forward')

        map2[(len(architecture)-1)]-= np.max(map2[(len(architecture)-1)])
        map1[(len(architecture)-1)] = np.exp(map2[(len(architecture)-1)]) / np.sum(np.exp(map2[(len(architecture)-1)]))
        map1[(len(architecture)-1)]=map1[(len(architecture)-1)].reshape(-1)
        if (map1[(len(architecture)-1)][0]>=map1[(len(architecture)-1)][1]):
            map1[(len(architecture)-1)]=[1,0]
        else:
            map1[(len(architecture)-1)]=[0,1]
        predicted_Y.append(map1[(len(architecture)-1)])

    predicted_Y=np.asarray(predicted_Y)
    return predicted_Y

def train_nn(architecture,X_train,Y_train,X_test,Y_test):
    Weights=dict()
    Biases=dict()
    map1=dict()
    map2=dict()
    delta=dict()

    map1[0]=np.empty(shape=[1,architecture[0]])
    Weights,Biases = weight_initilizer(Weights,Biases,(len(architecture)-1))

    train_errors=list()
    test_errors=list()
    epoch_list=list()

    for epoch in range(num_epochs):
        print("Epoch",epoch)
        index = []
        for i in range(len(X_train)):
            index.append(i)
        random.shuffle(index)
        temp_train_X = X_train
        temp_train_Y = Y_train
        for i in range(len(index)):
            temp_train_X[i] = X_train[index[i]]
            temp_train_Y[i] = Y_train[index[i]]
        X_train = temp_train_X
        Y_train = temp_train_Y

        for n in range(len(X_train)):
            map1[0]=X_train[n]
            y=Y_train[n]
            new_y = forward_nn(Weights,0,Biases,X_train)
            for l in range(1,(len(architecture)-1)+1):
                map2[l]=map1[l-1].dot(Weights[l])+Biases[l]
                map1[l]=relu(map2[l],'forward')
                mask=(np.random.rand(*map1[l].shape) < persist)/persist
                if(l!=(len(architecture)-1)):
                    map1[l]*= mask
            if np.sum(np.exp(map2[(len(architecture)-1)])) != 0:
                map1[(len(architecture)-1)] = np.exp(map2[(len(architecture)-1)]) / np.sum(np.exp(map2[(len(architecture)-1)]))

            delta, Weights, Biases = backward_nn(delta,(len(architecture)-1),Weights,Biases,map2,y,map1)

        train_cost=CE_loss(Y_train,forward_nn(Weights,1,Biases,X_train))
        len_vec=len(Y_train)
        J = np.sum((Y_train - forward_nn(Weights,1,Biases,X_train)) ** 2)/(2 * len_vec)
        train_cost = J
        print("Train Error =",train_cost)
        test_cost=CE_loss(Y_test,forward_nn(Weights,1,Biases,X_test))
        len_vec=len(Y_test)
        J = np.sum((Y_test - forward_nn(Weights,1,Biases,X_test)) ** 2)/(2 * len_vec)
        test_cost = J
        print("Test Error =",test_cost)
        train_errors.append(train_cost)
        test_errors.append(test_cost)
        epoch_list.append(epoch)

    return train_errors,test_errors,epoch_list,Weights,Biases

def load_data():
    X_train = []
    with open('train_X.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            temp_row = [float(a) for a in row]
            X_train.append(temp_row)
    X_train = np.array(X_train)

    X_test = []
    with open('test_X.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            temp_row = [float(a) for a in row]
            X_test.append(temp_row)
    X_test = np.array(X_test)

    Y_train = []
    with open('train_Y.txt', 'r') as f:
        a = f.readlines()
        for i in range(len(a)):
            a[i] = a[i].split()
            a[i] = a[i][0]
            a[i] = float(a[i])
            if(a[i] == 1):
                Y_train.append([0.0,1.0])
            else:
                Y_train.append([1.0,0.0])
    Y_train = np.array(Y_train)

    Y_test = []
    with open('test_Y.txt', 'r') as f:
        a = f.readlines()
        for i in range(len(a)):
            a[i] = a[i].split()
            a[i] = a[i][0]
            a[i] = float(a[i])
            if(a[i] == 1):
                Y_test.append([0.0,1.0])
            else:
                Y_test.append([1.0,0.0])
    Y_test = np.array(Y_test)

    total_words = len(X_train[0])

    print("Data Loaded")
    print("X_train =",X_train)
    print("Y_train =",Y_train)
    print("X_test =",X_test)
    print("Y_test =",Y_test)

    return X_train, Y_train, X_test, Y_test , total_words

num_epochs = 21
learning_rate = 0.1
threshold=0.6
persist=0.9

print("Neurons in first hidden layer")
layer1 = int(input())

print("Neurons in second hidden layer")
layer2 = int(input())

print("Loading Data")
X_train, Y_train, X_test, Y_test, total_words = load_data()
architecture = [total_words,layer1,layer2,2]
NN_ARCHITECTURE = [
    {"input_dim": total_words, "output_dim": layer1, "activation": "sigmoid"},
    {"input_dim": layer1, "output_dim": layer2, "activation": "sigmoid"},
    {"input_dim": layer2, "output_dim": 1}
]

print("Training Started")
l1,l2,l3,Weights,Biases=train_nn(architecture,X_train,Y_train,X_test,Y_test)
print("Training Ended")

print("Training Accuracy")
train_Y_pred = predict_nn(Weights,Biases,X_train)
print("Train Accuracy = ",accuracy_score(Y_train,train_Y_pred,normalize=True))

print("Testing Accuracy")
test_Y_pred = predict_nn(Weights,Biases,X_test)
print("Test Accuracy = ",accuracy_score(Y_test,test_Y_pred,normalize=True))

f = plt.figure()
plt.plot(l3, l1, color='r', label='Train Cross Entropy Error')
plt.plot(l3, l2, color='g', label='Test Cross Entropy Error')
plt.xlabel('Epoch No.')
plt.ylabel('Cross Entropy Error')
plt.title('Plot of Training & Testing Error vs Epoch')
legend = plt.legend(loc='upper left', shadow=False, fontsize='x-small')
f.savefig("temp.pdf", bbox_inches='tight')
