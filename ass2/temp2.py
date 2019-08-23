import pandas
from csv import reader

def most_frequent(List):
    return max(set(List), key = List.count)

class Decision_Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.isRoot = False
        self.word = None
        self.word_bool = None
        self.isTerminal = False
        self.terminal_class = None
        self.parent = None

def node_split(word, doc_group):
    left_child, right_child = list(), list()
    for doc in doc_group:
        if word in train_doc_word_list[doc]:
            left_child.append(doc)
        else:
            right_child.append(doc)
    return left_child, right_child


def gini_index(doc_groups, classes):
    total_instances = float(sum([len(doc_group) for doc_group in doc_groups]))
    gini = 0.0
    for doc_group in doc_groups:
        score = 0.0
        size = float(len(doc_group))
        if size == 0:
            continue
        for class_val in classes:
            temp = []
            for doc in doc_group:
                temp.append(trainlabel[doc-1])
            p = temp.count(class_val)/size
            score += p*p
        gini += (1.0 - score)*(size/total_instances)
    return gini

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


def build_decision_tree(node, doc_group, words_used, depth, max_depth):
    if depth <= max_depth-1:
        print('Current Depth: ', depth)
        num_instances = len(doc_group)
        all_same = False
        for class_val in classes:
            temp = []
            for doc in doc_group:
                temp.append(trainlabel[doc-1])
            count = temp.count(class_val)
            if count == num_instances:
                all_same = True
                node_class = class_val
        if all_same == True:
            node.isTerminal = True
            node.terminal_class = node_class
        else:
            node_word, split_groups = get_best_split(doc_group, classes, words_used)
            node.left = Decision_Tree()
            node.left.parent = node
            node.left.word_bool = 1
            node.right = Decision_Tree()
            node.right.word_bool = 0
            node.right.parent = node
            node.word = node_word
            words_used.append(node_word)
            build_decision_tree(node.left, split_groups[0], words_used, depth+1, max_depth)
            build_decision_tree(node.right, split_groups[1], words_used, depth+1, max_depth)
    else:
        print('Current Depth: ', depth)
        class_count = [0,0]
        temp = []
        for doc in doc_group:
            temp.append(trainlabel[doc-1])
        node.terminal_class = most_frequent(temp)
        node.isTerminal = True


def print_tree(node, level):
	if node.isRoot == True:
		level = 1
		print_tree(node.left, level)
		print_tree(node.right, level)
	else:
		if node.isTerminal	== True:
			for i in range(level-1):
				print('\t', end=" ")
			if(level > 1):
				print("|", words[node.parent.word-1], "=", node.word_bool, ":", node.terminal_class)
			else:
				print(words[node.parent.word-1], "=", node.word_bool, ":", node.terminal_class)
		else:
			for i in range(level-1):
				print('\t', end=" ")
			if(level > 1):
				print("|", words[node.parent.word-1], "=", node.word_bool)
			else:
				print(words[node.parent.word-1], "=", node.word_bool)
			level += 1
			print_tree(node.left,  level)
			print_tree(node.right, level)

def test_tree(node, test_instance):
    if node.isTerminal == True:
        # test_instance.append(node.terminal_class)
        return node.terminal_class
    else:
        if node.word in train_doc_word_list[test_instance]:
            is_present = 1
        else:
            is_present = 0
        if is_present == 1:
            return test_tree(node.left, test_instance)
        if is_present == 0:
            return test_tree(node.right, test_instance)

dataframe = pandas.read_csv("traindata.txt",delimiter="\t")

dataframe.to_csv("traindata_csv.csv", encoding='utf-8', index=False)

train_filename = 'traindata_csv.csv'
file = open(train_filename, "r")
lines = reader(file)
train_dataset = list(lines)

train_dataset[0][1] = '1'
num_docs = int(train_dataset[len(train_dataset)-1][0])


train_doc_word_list = []

for i in range(num_docs+1):
    temp = []
    train_doc_word_list.append(temp)

for row in train_dataset:
    train_doc_word_list[int(row[0])].append(int(row[1]))


with open('trainlabel.txt', 'r') as f:
    trainlabel = [line.strip() for line in f]

with open('words.txt', 'r') as f:
    words = [line.strip() for line in f]

totwords = len(words)

print(totwords)

print(len(trainlabel))

for i in range(len(trainlabel)):
    trainlabel[i] = int(trainlabel[i])

doc_group = []

for i in range(num_docs):
    doc_group.append(i+1)

classes = [1 ,2]

print(classes[0])


words_used = []

root = Decision_Tree()
root.isRoot = True

max_depth = 20

dataframe = pandas.read_csv("testdata.txt",delimiter="\t")

dataframe.to_csv("testdata_csv.csv", encoding='utf-8', index=False)

test_filename = 'testdata_csv.csv'
file = open(test_filename, "r")
lines = reader(file)
test_dataset = list(lines)

build_decision_tree(root, doc_group, words_used, 0, max_depth)

print_tree(root, 0)

outcome = []
for doc in doc_group:
    outcome.append(test_tree(root, doc))

num_test_instances = len(doc_group)
num_true_indentification = 0
i = 0
for row in trainlabel:
    if row == outcome[i]:
        num_true_indentification = num_true_indentification + 1
    i = i+1

accuracy = num_true_indentification/num_test_instances*100

print('Accuracy: ',accuracy,'%')