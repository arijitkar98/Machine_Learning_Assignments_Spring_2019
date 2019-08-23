import pandas
from csv import reader
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

##Scikit-learn##

print("****With scikit-learn****\n")

##Training Part##

##Load Training Data
print("****Loading Training Data****\n")
train_filename = 'traindata.csv'
file = open(train_filename, "r")
lines = reader(file)
train_dataset = list(lines)
train_dataset[0][1] = '1'

total_train_samples = int(train_dataset[len(train_dataset)-1][0])

dataset = []
cur_index = 1
i = 0
while i < len(train_dataset)-1:
    arr = []
    index = int(train_dataset[i][0])
    while cur_index == index:
        arr.append(int(train_dataset[i][1]))
        if i+1 < len(train_dataset):
            i += 1
            index = int(train_dataset[i][0])
        else:
            break
    dataset.append(arr)
    cur_index += 1

dataset_data = np.empty((0,3566), dtype=int)
for i in range(len(dataset)):
	temp = np.empty((1,3566), dtype=int)
	for j in range(3566):
		if j+1 in dataset[i]:
			temp[0][j] = 1 #YES
		else:
			temp[0][j] = 0 #NO
	dataset_data = np.concatenate((dataset_data, temp))

train_docID_to_wordID_map = dataset_data

##Load Training Labels
print("****Loading Training Labels****\n")
train_filename = 'trainlabel.txt'
file = open(train_filename, "r")
lines = file.readlines()
dataset_label = []

for i in range(len(lines)):
	dataset_label.append(int(lines[i].split('\n')[0]) - 1)
train_docID_to_label_map = np.array(dataset_label)

##Load Testing Data
print("****Loading Testing Data****\n")
test_filename = 'testdata.csv'
file = open(test_filename, "r")
lines = reader(file)
test_dataset = list(lines)

total_test_samples = int(test_dataset[len(test_dataset)-1][0])

dataset = []
cur_index = 1
i = 0
while i < len(test_dataset)-1:
    arr = []
    index = int(test_dataset[i][0])
    while cur_index == index:
        arr.append(int(test_dataset[i][1]))
        if i+1 < len(test_dataset):
            i += 1
            index = int(test_dataset[i][0])
        else:
            break
    dataset.append(arr)
    cur_index += 1

dataset_data = np.empty((0,3566), dtype=int)
for i in range(len(dataset)):
	temp = np.empty((1,3566), dtype=int)
	for j in range(3566):
		if j+1 in dataset[i]:
			temp[0][j] = 1 #YES
		else:
			temp[0][j] = 0 #NO
	dataset_data = np.concatenate((dataset_data, temp))

test_docID_to_wordID_map = dataset_data

##Load Testing Labels
print("****Loading Testing Labels****\n")
test_filename = 'testlabel.txt'
file = open(test_filename, "r")
lines = file.readlines()
dataset_label = []

for i in range(len(lines)):
	dataset_label.append(int(lines[i].split('\n')[0])-1)
test_docID_to_label_map = np.array(dataset_label)

train_accuracy = []
test_accuracy = []

for maxDepth in range(1,41):
	print("\n\n\n**** Starting Training for Max Depth =", maxDepth, "****")

	clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=maxDepth)
	clf = clf.fit(train_docID_to_wordID_map, train_docID_to_label_map)

	##Training Accuracy
	count = 0
	for i in range(len(train_docID_to_wordID_map)):
		predicted_label = clf.predict(np.array([train_docID_to_wordID_map[i]]))
		if predicted_label == 0:
			predicted_label = 'alt.atheism'
		else:
			predicted_label = 'comp.graphics'
		true_label = train_docID_to_label_map[i]
		if true_label == 0:
			true_label = 'alt.atheism'
		else:
			true_label = 'comp.graphics'

		if predicted_label == true_label:
			count += 1

	accuracy = count*100/1061
	print("\n\nTraining Accuracy =", accuracy)
	train_accuracy.append(accuracy)

	count = 0
	for i in range(len(test_docID_to_wordID_map)):
		predicted_label = clf.predict(np.array([test_docID_to_wordID_map[i]]))
		if predicted_label == 0:
			predicted_label = 'alt.atheism'
		else:
			predicted_label = 'comp.graphics'
		true_label = test_docID_to_label_map[i]
		if true_label == 0:
			true_label = 'alt.atheism'
		else:
			true_label = 'comp.graphics'

		if predicted_label == true_label:
			count += 1

	accuracy = count*100/707
	print("\n\nTesting Accuracy =", accuracy)
	test_accuracy.append(accuracy)

maxdepth = [i for i in range(1,41)]
f = plt.figure()
plt.plot(maxdepth, train_accuracy, color='b', label='Training Accuracy')
plt.plot(maxdepth, test_accuracy, color='y', label='Testing Accuracy')
plt.xlabel('Max Allowed Depth of Tree')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy for different Max Depths')
legend = plt.legend(loc='upper left', shadow=False, fontsize='x-small')
f.savefig("sklearn_accuracy_plot.pdf", bbox_inches='tight')
