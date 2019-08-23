import pandas
from csv import reader
import math
import matplotlib.pyplot as plt

class DT(object):
    def __init__(self):
        self.is_root = False
        self.num_children = 2
        self.left = None
        self.right = None
        self.parent_attribute = None
        self.parent_attribute_val = None
        self.attribute = None
        self.is_terminal = False
        self.terminal_class = None
        self.depth = 0

def calc_info_gain(parent, dataset_groups, classes):
    total_instances = float(sum([len(group) for group in dataset_groups]))
    entropy = 0.0
    for group in dataset_groups:
        score = 0.0
        size = float(len(group))
        if size == 0:
            continue
        for class_val in classes:
            p = [train_docID_to_label_map[sample-1] for sample in group].count(class_val)/size
            if p != 0:
                score += -(p*math.log(p,2))
        entropy += score*(size/total_instances)
    parent_entropy = 0.0
    size = float(len(parent))
    for class_val in classes:
        p = [train_docID_to_label_map[sample-1] for sample in parent].count(class_val)/size
        parent_entropy += -(p*math.log(p,2))
    return parent_entropy - entropy

def split_current_node(attribute, dataset_group):
    left_child, right_child = [], []
    for sample in dataset_group:
        if attribute in train_docID_to_wordID_map[sample]:
            left_child.append(sample)
        else:
            right_child.append(sample)
    return left_child, right_child

def get_best_split(dataset, classes, attributes_used):
    max_info_gain = -10000
    best_word = -1
    best_split = []
    for i in range(total_words):
        if i+1 not in attributes_used:
            split_groups = split_current_node(i+1, dataset)
            info_gain = calc_info_gain(dataset, split_groups, classes)
            # print("gini_val =",gini_val,' ',i+1)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_word = i+1
                best_split = split_groups
    return best_word, best_split

def build_tree(node, dataset, attributes_used, max_depth):
    if node.is_root == True:
        print("Training time Depth of Tree:")
        print(node.depth, end=" ")
    else:
        print("-->",node.depth, end=" ")
    count = []
    cur_count = 0
    for class_val in classes:
        cur_count = 0
        for sample in dataset:
            if train_docID_to_label_map[sample-1] == class_val:
                cur_count += 1
        count.append(cur_count)
    if node.depth >= max_depth:
        node.is_terminal = True
        if count[0] > count[1]:
            node.terminal_class = 1
        else:
            node.terminal_class = 2
    elif count[0] == len(dataset) or count[1] == len(dataset):
        node.is_terminal = True
        if count[0] == len(dataset):
            node.terminal_class = 1
        else:
            node.terminal_class = 2
    else:
        best_attr, best_split_groups = get_best_split(dataset, classes, attributes_used)
        node.attribute = best_attr
        attributes_used.append(best_attr)

        node.left = DT()
        node.left.parent_attribute = best_attr
        node.left.parent_attribute_val = 1
        node.left.depth = node.depth + 1
        build_tree(node.left, best_split_groups[0], attributes_used, max_depth)
        print(" ")
        if node.depth == 1:
            print(" ", end=" ")
            for i in range(node.depth):
                print("     ", end=" ")
        else:
            for i in range(node.depth):
                print("     ", end=" ")
        node.right = DT()
        node.right.parent_attribute = best_attr
        node.right.parent_attribute_val = 0
        node.right.depth = node.depth + 1
        build_tree(node.right, best_split_groups[1], attributes_used, max_depth)

def print_tree(node, attributes_list, attribute_values, classes_values, level):
    if node.is_root == True:
        level = 1
        print_tree(node.left, attributes_list, attribute_values, classes_values, level)
        print_tree(node.right, attributes_list, attribute_values, classes_values, level)
    else:
        if node.is_terminal == True:
            for i in range(level-2):
                print('\t', end=" ")
            if(level > 1):
                print("|", attributes_list[node.parent_attribute-1], "=", attribute_values[node.parent_attribute_val], ":", classes_values[node.terminal_class-1])
            else:
                print(attributes_list[node.parent_attribute-1], "=", attribute_values[node.parent_attribute_val], ":", classes_values[node.terminal_class-1])
        else:
            for i in range(level-2):
                print('\t', end=" ")
            if(level > 1):
                print("|", attributes_list[node.parent_attribute-1], "=", attribute_values[node.parent_attribute_val])
            else:
                print(attributes_list[node.parent_attribute-1], "=", attribute_values[node.parent_attribute_val])

            level += 1
            print_tree(node.left, attributes_list, attribute_values, classes_values, level)
            print_tree(node.right, attributes_list, attribute_values, classes_values, level)

def test_tree(node, test_instance, mode):
    if node.is_terminal == True:
        return node.terminal_class
    else:
        if mode == 'train':
            if node.attribute in train_docID_to_wordID_map[test_instance]:
                return test_tree(node.left, test_instance, 'train')
            else:
                return test_tree(node.right, test_instance, 'train')
        else:
            if node.attribute in train_docID_to_wordID_map[test_instance]:
                return test_tree(node.left, test_instance, 'train')
            else:
                return test_tree(node.right, test_instance, 'train')

##Without Scikit-learn##

print("****Without scikit-learn****\n")

##Load Training Data
print("****Loading Training Data****\n")
train_filename = 'traindata.csv'
file = open(train_filename, "r")
lines = reader(file)
train_dataset = list(lines)
train_dataset[0][1] = '1'

total_train_samples = int(train_dataset[len(train_dataset)-1][0])

dataset = [[]]
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
train_docID_to_wordID_map = dataset

##Load Training Labels
print("****Loading Training Labels****\n")
f = open('trainlabel.txt', 'r')
train_docID_to_label_map = [int(line.strip()) for line in f]
f = open('words.txt', 'r')
words = [line.strip() for line in f]
total_words = len(words)

train_docID_list = []
for i in range(total_train_samples):
    train_docID_list.append(i+1)

##Load Testing Data
print("****Loading Testing Data****\n")
test_filename = 'testdata.csv'
file = open(test_filename, "r")
lines = reader(file)
test_dataset = list(lines)

total_test_samples = int(test_dataset[len(test_dataset)-1][0])

dataset = [[]]
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
test_docID_to_wordID_map = dataset

##Load Testing Labels
print("****Loading Testing Labels****\n")
f = open('testlabel.txt', 'r')
test_docID_to_label_map = [int(line.strip()) for line in f]

test_docID_list = []
for i in range(total_test_samples):
    test_docID_list.append(i+1)

##Other Training Parameters
classes = [1 ,2]
attributes_list = words
attribute_values = ['no', 'yes']
classes_values = ['alt.atheism', 'comp.graphics']

train_accuracy = []
test_accuracy = []

max_depth = 1
while max_depth < 26:
    print("\n\n\n**** Starting Training for Max Depth =", max_depth,"****")
    attributes_used = []
    root = DT()
    root.is_root = True
    root.depth = 0

    build_tree(root, train_docID_list, attributes_used, max_depth)
    print(" ")
    print("Decision Tree:\n")
    print_tree(root, attributes_list, attribute_values, classes_values, 0)

    print("\n****Testing on Train Data****\n")

    count = 0
    for i in range(len(train_docID_list)):
        predicted_label = test_tree(root, train_docID_list[i], 'train')
        if predicted_label == 1:
            predicted_label = 'alt.atheism'
        else:
            predicted_label = 'comp.graphics'
        true_label = train_docID_to_label_map[i]
        if true_label == 1:
            true_label = 'alt.atheism'
        else:
            true_label = 'comp.graphics'

        if predicted_label == true_label:
            count += 1

    accuracy = count*100/1061
    print("Accuracy =", accuracy)
    train_accuracy.append(accuracy)

    print("\n****Testing on Test Data****\n")

    count = 0
    for i in range(total_test_samples):
        predicted_label = test_tree(root, test_docID_list[i], 'test')
        if predicted_label == 1:
            predicted_label = 'alt.atheism'
        else:
            predicted_label = 'comp.graphics'
        true_label = test_docID_to_label_map[i]
        if true_label == 1:
            true_label = 'alt.atheism'
        else:
            true_label = 'comp.graphics'

        if predicted_label == true_label:
            count += 1

    accuracy = count*100/707
    print("Accuracy =", accuracy)
    test_accuracy.append(accuracy)

    max_depth += 1

maxdepth = [i for i in range(1,26)]
f = plt.figure()
plt.plot(maxdepth, train_accuracy, color='b', label='Training Accuracy')
plt.plot(maxdepth, test_accuracy, color='y', label='Testing Accuracy')
plt.xlabel('Max Allowed Depth of Tree')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy for different Max Depths')
legend = plt.legend(loc='upper left', shadow=False, fontsize='x-small')
f.savefig("non_sklearn_accuracy_plot.pdf", bbox_inches='tight')
