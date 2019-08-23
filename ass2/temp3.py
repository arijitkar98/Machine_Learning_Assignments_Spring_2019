import pandas
from csv import reader
import math

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

def node_split(attribute, dataset_group):
    left_child, right_child = [], []
    for sample in dataset_group:
        if attribute in train_doc_word_list[sample]:
            left_child.append(sample)
        else:
            right_child.append(sample)
    return left_child, right_child

def information_gain(parent, dataset_groups, classes):
    total_instances = float(sum([len(group) for group in dataset_groups]))
    entropy = 0.0
    for group in dataset_groups:
        score = 0.0
        size = float(len(group))
        if size == 0:
            continue
        for class_val in classes:
            p = [trainlabel[sample-1] for sample in group].count(class_val)/size
            if p != 0:
                score += -(p*math.log(p,2))
        entropy += score*(size/total_instances)
    parent_entropy = 0.0
    size = float(len(parent))
    for class_val in classes:
        p = [trainlabel[sample-1] for sample in parent].count(class_val)/size
        parent_entropy += -(p*math.log(p,2))
    return parent_entropy - entropy

def gini_index(dataset_groups, classes):
    total_instances = float(sum([len(doc_group) for dataset_group in dataset_groups]))
    gini = 0.0
    for dataset_group in dataset_groups:
        score = 0.0
        size = float(len(dataset_group))
        if size == 0:
            continue
        for class_val in classes:
            p = [trainlabel[sample-1] for sample in dataset_group].count(class_val)/size
            score += p*p
        gini += (1.0 - score)*(size/total_instances)
    return gini

# def get_best_split(doc_group, classes, words_used):
#     max_info_gain = -9999
#     best_word = -1
#     best_split = []
#     for i in range(totwords):
#         if i+1 not in words_used:
#             groups = node_split(i+1, doc_group)
#             info_gain = information_gain(doc_group, groups, classes)
#             # print("gini_val =",gini_val,' ',i+1)
#             if info_gain > max_info_gain:
#                 max_info_gain = info_gain
#                 best_word = i+1
#                 best_split = groups
#     return best_word, best_split

def get_best_split(doc_group, classes, words_used):
    min_gini = 9999
    best_word = -1
    best_split = []
    for i in range(totwords):
        if i+1 not in words_used:
            groups = node_split(i+1, doc_group)
            gini_val = gini_index(groups, classes)
            # print("gini_val =",gini_val,' ',i+1)
            if gini_val < min_gini:
                min_gini = gini_val
                best_word = i+1
                best_split = groups
    return best_word, best_split


def build_DT(node, doc_group, words_used, max_depth):
    print("Current Depth = ", node.depth)
    count = []
    cur_count = 0
    for class_val in classes:
        cur_count = 0
        for doc in doc_group:
            if trainlabel[doc-1] == class_val:
                cur_count += 1
        count.append(cur_count)

    if node.depth >= max_depth:
        node.is_terminal = True
        if count[0] > count[1]:
            node.terminal_class = 1
        else:
            node.terminal_class = 2
    elif count[0] == len(doc_group) or count[1] == len(doc_group):
        node.is_terminal = True
        if count[0] == len(doc_group):
            node.terminal_class = 1
        else:
            node.terminal_class = 2
    else:
        best_attr, best_split_groups = get_best_split(doc_group, classes, words_used)
        node.attribute = best_attr
        words_used.append(best_attr)

        node.left = DT()
        node.left.parent_attribute = best_attr
        node.left.parent_attribute_val = 1
        node.left.depth = node.depth + 1
        build_DT(node.left, best_split_groups[0], words_used, max_depth)
        node.right = DT()
        node.right.parent_attribute = best_attr
        node.right.parent_attribute_val = 0
        node.right.depth = node.depth + 1
        build_DT(node.right, best_split_groups[1], words_used, max_depth)

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
            if node.attribute in train_doc_word_list[test_instance]:
                return test_tree(node.left, test_instance, 'train')
            else:
                return test_tree(node.right, test_instance, 'train')
        else:
            if node.attribute in test_doc_word_list[test_instance]:
                return test_tree(node.left, test_instance, 'train')
            else:
                return test_tree(node.right, test_instance, 'train')

dataframe = pandas.read_csv("traindata.txt",delimiter="\t")
dataframe.to_csv("traindata_csv.csv", encoding='utf-8', index=False)

train_filename = 'traindata_csv.csv'
file = open(train_filename, "r")
lines = reader(file)
train_dataset = list(lines)
train_dataset[0][1] = '1'

num_docs = int(train_dataset[len(train_dataset)-1][0])

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
train_doc_word_list = dataset

with open('trainlabel.txt', 'r') as f:
    trainlabel = [line.strip() for line in f]

with open('words.txt', 'r') as f:
    words = [line.strip() for line in f]

totwords = len(words)

for i in range(len(trainlabel)):
    trainlabel[i] = int(trainlabel[i])

doc_group = []

for i in range(num_docs):
    doc_group.append(i+1)

classes = [1 ,2]
words_used = []
attributes_list = words
attribute_values = ['no', 'yes']
classes_values = ['alt.atheism', 'comp.graphics']

root = DT()
root.is_root = True
root.depth = 0

max_depth = 10
build_DT(root, doc_group, words_used, max_depth)

print_tree(root, attributes_list, attribute_values, classes_values, 0)


print("****Testing on Train Data****\n")

count = 0
for i in range(len(doc_group)):
    predicted_label = test_tree(root, doc_group[i], 'train')
    if predicted_label == 1:
        predicted_label = 'alt.atheism'
    else:
        predicted_label = 'comp.graphics'
    true_label = trainlabel[i]
    if true_label == 1:
        true_label = 'alt.atheism'
    else:
        true_label = 'comp.graphics'

    if predicted_label == true_label:
        count += 1

accuracy = count*100/1061
print("\n\nAccuracy =", accuracy)

print("****Testing on Test Data****\n")

dataframe = pandas.read_csv("testdata.txt",delimiter="\t")
dataframe.to_csv("testdata_csv.csv", encoding='utf-8', index=False)

test_filename = 'testdata_csv.csv'
file = open(test_filename, "r")
lines = reader(file)
test_dataset = list(lines)

num_docs = int(test_dataset[len(test_dataset)-1][0])
test_doc_word_list = []

for i in range(num_docs+1):
    temp = []
    test_doc_word_list.append(temp)
for row in test_dataset:
    test_doc_word_list[int(row[0])].append(int(row[1]))

with open('testlabel.txt', 'r') as f:
    testlabel = [line.strip() for line in f]
for i in range(len(testlabel)):
    testlabel[i] = int(testlabel[i])

doc_group = []
for i in range(num_docs):
    doc_group.append(i+1)

count = 0
for i in range(num_docs):
    predicted_label = test_tree(root, doc_group[i], 'test')
    if predicted_label == 1:
        predicted_label = 'alt.atheism'
    else:
        predicted_label = 'comp.graphics'
    true_label = testlabel[i]
    if true_label == 1:
        true_label = 'alt.atheism'
    else:
        true_label = 'comp.graphics'

    if predicted_label == true_label:
        count += 1

accuracy = count*100/707
print("\n\nAccuracy =", accuracy)
