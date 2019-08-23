from csv import reader
import numpy as np
from sklearn import tree
import math

class DT(object):
	def __init__(self):
		self.is_root = False
		self.num_children = 0
		self.left = None
		self.center = None
		self.right = None
		self.parent_attribute = None
		self.parent_attribute_val = None
		self.attribute = None
		self.is_terminal = False
		self.terminal_class = None

def split_current_node(attribute, dataset, attribute_values):
    if attribute == 3:
        left_child = np.empty((0,5), dtype='<U4')
        right_child = np.empty((0,5), dtype='<U4')
        for row in dataset:
            if row[attribute] == attribute_values[attribute][0]:
                left_child = np.concatenate((left_child, np.array([row])))
            elif row[attribute] == attribute_values[attribute][1]:
                right_child = np.concatenate((right_child, np.array([row])))
        return left_child, right_child
    else:
        left_child = np.empty((0,5), dtype='<U4')
        middle_child = np.empty((0,5), dtype='<U4')
        right_child = np.empty((0,5), dtype='<U4')
        for row in dataset:
            if row[attribute] == attribute_values[attribute][0]:
                left_child = np.concatenate((left_child, np.array([row])))
            elif row[attribute] == attribute_values[attribute][1]:
                middle_child = np.concatenate((middle_child, np.array([row])))
            elif row[attribute] == attribute_values[attribute][2]:
                right_child = np.concatenate((right_child, np.array([row])))
        return left_child, middle_child, right_child

def calc_gini_index(groups, classes):
    total_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        score = 0.0
        size = float(len(group))
        if size == 0:
            continue
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/size
            score += p*p
        gini += (1.0 - score)*(size/total_instances)
    return gini

def calc_info_gain(parent_group, groups, classes):
    total_instances = float(sum([len(group) for group in groups]))
    entropy = 0.0
    for group in groups:
        score = 0.0
        size = float(len(group))
        if size == 0:
            continue
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/size
            if p != 0:
                score += -(p*math.log(p,2))
        entropy += score*(size/total_instances)
    parent_entropy = 0.0
    size = float(len(parent_group))
    for class_val in classes:
        p = [row[-1] for row in parent_group].count(class_val)/size
        parent_entropy += -(p*math.log(p,2))
    return parent_entropy - entropy

def get_best_split(dataset, classes, attribute_values, attributes_left, mode):
	min_gini = 10000
	max_info_gain = -10000
	best_attr = -1
	best_split = []
	for attr in attributes_left:
		groups = split_current_node(attr, dataset, attribute_values)

		if mode == 'gini':
			gini_val = calc_gini_index(groups, classes)
			if gini_val < min_gini:
				min_gini = gini_val
				best_attr = attr
				best_split = groups
		else:
			info_gain = calc_info_gain(dataset, groups, classes)
			if info_gain > max_info_gain:
				max_info_gain = info_gain
				best_attr = attr
				best_split = groups	    	

	return best_attr, best_split

def build_tree(node, group, classes, attribute_values, attributes_unused, mode):
	count = []
	cur_count = 0
	for class_val in classes:
		cur_count = 0
		for row in group:
			if row[-1] == class_val:
				cur_count += 1
		count.append(cur_count)
	if count[0] == len(group) or count[1] == len(group):
		node.is_terminal = True
		if count[0] == len(group):
			node.terminal_class = 'yes'
		else:
			node.terminal_class = 'no'
	else:
		best_attr, best_split_groups = get_best_split(group, classes, attribute_values, attributes_unused, mode)
		if node.is_root == True:
			if mode == 'gini':
				print("\n**** Gini Index at root node =", calc_gini_index(best_split_groups, classes),"****")
			else:
				print("\n**** Information Gain at root node =", calc_info_gain(group, best_split_groups, classes), "****")
		attributes_unused.remove(best_attr)
		node.attribute = best_attr
		if len(best_split_groups) == 2:
			node.left = DT()
			node.left.parent_attribute = best_attr
			node.left.parent_attribute_val = 0
			build_tree(node.left, best_split_groups[0], classes, attribute_values, attributes_unused, mode)
			node.right = DT()
			node.right.parent_attribute = best_attr
			node.right.parent_attribute_val = 1
			build_tree(node.right, best_split_groups[1], classes, attribute_values, attributes_unused, mode)
			node.num_children = 2

		else:
			node.left = DT()
			node.left.parent_attribute = best_attr
			node.left.parent_attribute_val = 0
			build_tree(node.left, best_split_groups[0], classes, attribute_values, attributes_unused, mode)
			node.center = DT()
			node.center.parent_attribute = best_attr
			node.center.parent_attribute_val = 1
			build_tree(node.center, best_split_groups[1], classes, attribute_values, attributes_unused, mode)
			node.right = DT()
			node.right.parent_attribute = best_attr
			node.right.parent_attribute_val = 2
			build_tree(node.right, best_split_groups[2], classes, attribute_values, attributes_unused, mode)
			node.num_children = 3

def print_tree(node, attributes_list, attribute_values, level):
	if node.is_root == True:
		if node.num_children == 2:
			level = 1
			print_tree(node.left, attributes_list, attribute_values, level)
			print_tree(node.right, attributes_list, attribute_values, level)
		else:
			level = 1
			print_tree(node.left, attributes_list, attribute_values, level)
			print_tree(node.center, attributes_list, attribute_values, level)
			print_tree(node.right, attributes_list, attribute_values, level)
	else:
		if node.is_terminal	== True:
			for i in range(level-2):
				print('\t', end=" ")
			if(level > 1):
				print("|", attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val], ":", node.terminal_class)
			else:
				print(attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val], ":", node.terminal_class)
		else:
			for i in range(level-2):
				print('\t', end=" ")
			if(level > 1):
				print("|", attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val])
			else:
				print(attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val])
			if node.num_children == 2:
				level += 1
				print_tree(node.left, attributes_list, attribute_values, level)
				print_tree(node.right, attributes_list, attribute_values, level)
			else:
				level += 1
				print_tree(node.left, attributes_list, attribute_values, level)
				print_tree(node.center, attributes_list, attribute_values, level)
				print_tree(node.right, attributes_list, attribute_values, level)

def test_tree(node, data, attribute_values):
	if node.is_terminal == True:
		return node.terminal_class
	else:
		if node.num_children == 3:
			if data[node.attribute] == attribute_values[node.attribute][0]:
				return test_tree(node.left, data, attribute_values)
			elif data[node.attribute] == attribute_values[node.attribute][1]:
				return test_tree(node.center, data, attribute_values)
			else:
				return test_tree(node.right, data, attribute_values)
		else:
			if data[node.attribute] == attribute_values[node.attribute][0]:
				return test_tree(node.left, data, attribute_values)
			else:
				return test_tree(node.right, data, attribute_values)

##Without scikit-learn##

print("****Without scikit-learn****\n")

##Training Part##

print("****Training with Gini-Index****\n")

train_filename = 'train_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Training dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

dataset = np.array(dataset)

attributes_list = ['price','maintenance','capacity','airbag']
attribute_values = [['low','med','high'], ['low','med','high'], ['2','4','5'], ['yes','no']]
classes = ['yes', 'no']
attributes_left = [0,1,2,3]

root = DT()
root.is_root = True
build_tree(root, dataset, classes, attribute_values, attributes_left, 'gini')

print("\n\n\nDecision Tree: \n")
print_tree(root, attributes_list, attribute_values, 0)

##Testing Part##

print("\n\n\n****Testing with Gini-Index****\n")

train_filename = 'test_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Testing dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

data1 = dataset[0]
data2 = dataset[1]

count = 0
print("\n\nFor test case 1:\n")
predicted_label = test_tree(root, data1, attribute_values)
true_label = data1[4]
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

print("\n\nFor test case 2:\n")
predicted_label = test_tree(root, data2, attribute_values)
true_label = data2[4]
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

accuracy = count*100/2
print("\n\nGini-Index Accuracy =", accuracy)

##Training Part##

print("\n\n\n****Training with Information Gain****\n")

train_filename = 'train_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Training dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

dataset = np.array(dataset)

attributes_list = ['price','maintenance','capacity','airbag']
attribute_values = [['low','med','high'], ['low','med','high'], ['2','4','5'], ['yes','no']]
classes = ['yes', 'no']
attributes_left = [0,1,2,3]

root = DT()
root.is_root = True
build_tree(root, dataset, classes, attribute_values, attributes_left, 'info')

print("\n\n\nDecision Tree: \n")
print_tree(root, attributes_list, attribute_values, 0)

##Testing Part##

print("\n\n\n****Testing with Information Gain****\n")

train_filename = 'test_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Testing dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

data1 = dataset[0]
data2 = dataset[1]

count = 0
print("\n\nFor test case 1:\n")
predicted_label = test_tree(root, data1, attribute_values)
true_label = data1[4]
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

print("\n\nFor test case 2:\n")
predicted_label = test_tree(root, data2, attribute_values)
true_label = data2[4]
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

accuracy = count*100/2
print("\n\nInformation Gain Accuracy =", accuracy)



##Scikit-learn##

print("\n\n\nWith scikit-learn\n")

##Training Part##

print("****Training with Gini-Index****\n")

train_filename = 'train_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Training dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

dataset = np.array(dataset)
x = np.empty((0,4), dtype='<U4')
y = []

for i in range(len(dataset)):
	row = dataset[i]
	data = row[0:4]
	label = row[4]
	for j in range(4):
		if data[j] == 'low' or data[j] == '2' or data[j] == 'yes':
			data[j] = 0
		elif data[j] == 'med' or data[j] == '4' or data[j] == 'no':
			data[j] = 1
		else:
			data[j] = 2
	x = np.concatenate((x, np.array([data])))
	if label == 'yes':
		y.append([0])
	else:
		y.append([1])

y = np.array(y)

clf = tree.DecisionTreeClassifier(criterion="gini")
clf = clf.fit(x, y)

impurity = clf.tree_.impurity
nodes = clf.tree_.n_node_samples
SL_GI = (nodes[1]*impurity[1] + nodes[2]*impurity[2])/nodes[0]
print("**** Gini Index at root node =", SL_GI, "****\n")


##Testing Part##

print("\n\n\n****Testing with Gini-Index****\n")

train_filename = 'test_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Testing dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

data = dataset[0]
data1 = data[0:4]
for j in range(4):
	if data1[j] == 'low' or data1[j] == '2' or data1[j] == 'yes':
		data1[j] = 0
	elif data1[j] == 'med' or data1[j] == '4' or data1[j] == 'no':
		data1[j] = 1
	else:
		data1[j] = 2
data1 = np.array([data1])
label1 = data[4]

data = dataset[1]
data2 = data[0:4]
for j in range(4):
	if data2[j] == 'low' or data2[j] == '2' or data2[j] == 'yes':
		data2[j] = 0
	elif data2[j] == 'med' or data2[j] == '4' or data2[j] == 'no':
		data2[j] = 1
	else:
		data2[j] = 2
data2 = np.array([data2])
label2 = data[4]

count = 0
print("\n\nFor test case 1:\n")
predicted_label = clf.predict(data1)
true_label = label1
if predicted_label == 0:
	predicted_label = 'yes'
else:
	predicted_label = 'no'
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

print("\n\nFor test case 2:\n")
predicted_label = clf.predict(data2)
true_label = label2
if predicted_label == 0:
	predicted_label = 'yes'
else:
	predicted_label = 'no'
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

accuracy = count*100/2
print("\n\nAccuracy =", accuracy, "\n")

##Training Part##

print("****Training with Information Gain****\n")

train_filename = 'train_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Training dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

dataset = np.array(dataset)
x = np.empty((0,4), dtype='<U4')
y = []

for i in range(len(dataset)):
	row = dataset[i]
	data = row[0:4]
	label = row[4]
	for j in range(4):
		if data[j] == 'low' or data[j] == '2' or data[j] == 'yes':
			data[j] = 0
		elif data[j] == 'med' or data[j] == '4' or data[j] == 'no':
			data[j] = 1
		else:
			data[j] = 2
	x = np.concatenate((x, np.array([data])))
	if label == 'yes':
		y.append([0])
	else:
		y.append([1])

y = np.array(y)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x, y)

impurity = clf.tree_.impurity
nodes = clf.tree_.n_node_samples
SL_IG = impurity[0] - (nodes[1]*impurity[1] + nodes[2]*impurity[2])/nodes[0]
print("**** Information Gain at root node =", SL_IG, "****\n")

##Testing Part##

print("\n\n\n****Testing with Information Gain****\n")

train_filename = 'test_1.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

print("Testing dataset:\n")
for row in dataset:
    print(row)

del dataset[0]

data = dataset[0]
data1 = data[0:4]
for j in range(4):
	if data1[j] == 'low' or data1[j] == '2' or data1[j] == 'yes':
		data1[j] = 0
	elif data1[j] == 'med' or data1[j] == '4' or data1[j] == 'no':
		data1[j] = 1
	else:
		data1[j] = 2
data1 = np.array([data1])
label1 = data[4]

data = dataset[1]
data2 = data[0:4]
for j in range(4):
	if data2[j] == 'low' or data2[j] == '2' or data2[j] == 'yes':
		data2[j] = 0
	elif data2[j] == 'med' or data2[j] == '4' or data2[j] == 'no':
		data2[j] = 1
	else:
		data2[j] = 2
data2 = np.array([data2])
label2 = data[4]

count = 0
print("\n\nFor test case 1:\n")
predicted_label = clf.predict(data1)
true_label = label1
if predicted_label == 0:
	predicted_label = 'yes'
else:
	predicted_label = 'no'
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

print("\n\nFor test case 2:\n")
predicted_label = clf.predict(data2)
true_label = label2
if predicted_label == 0:
	predicted_label = 'yes'
else:
	predicted_label = 'no'
print("Predicted Label =", predicted_label)
print("True Label =", true_label)
if predicted_label == true_label:
	count += 1

accuracy = count*100/2
print("\n\nAccuracy =", accuracy)
