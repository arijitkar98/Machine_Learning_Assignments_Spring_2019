import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import random
import numpy as np
import csv

print("*************Pre Processing Started***************")
stopwords_vector = ["a","Ok","ok", "about", "above","wen","wat","u","ur","me","once","saturday","sunday","wednesday","yo","above", "across" ,"after","wat", "Go","afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

file = open("Assignment_4_data.txt", "r");
lines = [line.rstrip('\n') for line in file]
dataset = []
for line in lines:
    temp = line.split("\t", 1);
    dataset.append(temp)
random.shuffle(dataset)
print(len(dataset))
tokens = []
delims = [',','.','...','?','!', '', ' ', '+', ')', '(', '\\', ':', '>']
stemmer = PorterStemmer()
labels = []
for row in dataset:
    if(row[0] == 'spam'):
        labels.append(1.0)
    else:
        labels.append(0.0)
    temp1 = []
    temp2 = nltk.word_tokenize(row[1])
    for token in temp2:
        if token not in delims:
            token = token.split('.')
            for word in token:
                if word not in delims:
                    word = stemmer.stem(word)
                    if word not in stopwords.words('english'):
                        temp1.append(word)
    tokens.append(temp1)
tokens_all = []
for row in tokens:
    for token in row:
        if token not in tokens_all:
            tokens_all.append(token)
token_vector = []
for row in tokens:
    temp = []
    for token in tokens_all:
        if token in row:
            temp.append(1.0)
        else:
            temp.append(0.0)
    token_vector.append(temp)

X = np.array(token_vector)
Y = np.array(labels)
total_words = len(tokens_all)
train_length = (int)(0.8*len(dataset))
X_train = X[0:train_length+1]
X_test = X[train_length+1:]
Y_train = Y[0:train_length+1]
Y_test = Y[train_length+1:]

with open('X_train.csv', 'w') as f:
	print(len(X_train))
	print(len(X_train[0]))
	writer = csv.writer(f)
	writer.writerows(X_train)
with open('Y_train.txt', 'w') as f:
	for val in Y_train:
		f.write("%s\n" % val)
with open('X_test.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerows(X_test)
with open('Y_test.txt', 'w') as f:
	for val in Y_test:
		f.write("%s\n" % val)
