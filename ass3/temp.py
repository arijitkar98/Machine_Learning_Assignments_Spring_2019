from csv import reader
import  csv
import math

dataset_filename = 'AAAI.csv'
file = open(dataset_filename, "r")
lines = reader(file)
dataset = list(lines)

del dataset[0]

paper_topics = []

for i in range(len(dataset)):
    temp = []
    temp = dataset[i][2].split('\n')
    paper_topics.append(temp)

paper_high_level_keywords = []

for i in range(len(dataset)):
    temp = []
    temp = dataset[i][3]
    paper_high_level_keywords.append(temp)

dataset_filename = 'grivan_newman_clusters.csv'
file = open(dataset_filename, "r")
lines = reader(file)
clusters = list(lines)

for row in clusters:
    for i in range(len(row)):
        row[i] = int(row[i])

keywords = []
for x in paper_high_level_keywords:
    if x not in keywords:
        keywords.append(x)
print(keywords)

keyword_count = []
for i in range(len(keywords)):
    keyword_count.append(0)

for row in paper_high_level_keywords:
    i = 0
    for i in range(len(keywords)):
        if keywords[i] == row:
            keyword_count[i] += 1
print(keyword_count)

total_papers = float(len(paper_high_level_keywords))

HY = 0

for i in keyword_count:
    p = float(i/total_papers)
    HY -= p*math.log2(p)

HC = 0

for cluster in clusters:
    p = float(len(cluster))/float(total_papers)
    HC -= p*math.log2(p)

print("HC= ",HC, "\nHY= ", HY)

HYC = 0

for cluster in clusters:
    HY_C = 0
    total = float(len(cluster))
    cluster_keyword_count = []
    for i in range(len(keywords)):
        cluster_keyword_count.append(0)

    for paper in cluster:
        i = 0
        for i in range(len(keywords)):
            if keywords[i] == paper_high_level_keywords[paper]:
                cluster_keyword_count[i] += 1
    # print(cluster_keyword_count)
    # print(total)

    for i in cluster_keyword_count:
        p = float(i)/total
        if p != 0:
            HY_C -= p*math.log2(p)
    HY_C = HY_C * (total/total_papers)
    HYC = HYC + HY_C

print('HYC= ',HYC)

IYC = HY - HYC

print('IYC= ',IYC)

NMI = 2*IYC/(HY+HC)

print('NMI= ',NMI)