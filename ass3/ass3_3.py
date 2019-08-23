import csv
import numpy as np
import math

###Loading Dataset###
print("***Loading dataset from AAAI.csv***\n\n")
dataset_filename = 'AAAI.csv'
file = open(dataset_filename, "r")
lines = csv.reader(file)
dataset = list(lines)
del dataset[0]

paper_to_keyword_mapping = []
for i in range(len(dataset)):
    paper_to_keyword_mapping.append(dataset[i][3])

print("***Keywords present in the Dataset:***\n\n")
keywords = set(paper_to_keyword_mapping)
keywords = list(keywords)
print(keywords)

print("\n\n***Count of each keyword present in the Dataset:***\n\n")
keyword_count = np.zeros(len(keywords), dtype=int)
for paper_keyword in paper_to_keyword_mapping:
    count = 0
    for i in range(len(keywords)):
        if keywords[i] == paper_keyword:
            keyword_count[i] += 1
for i in range(len(keywords)):
    print(keywords[i], "-->", keyword_count[i])

total_papers = float(len(paper_to_keyword_mapping))

###Bottom Up Hierarchical Clustering - Single Linkage###
print("\n\n***Bottom Up Hierarchical Clustering - Single Linkage***\n\n")
test_filename = 'part1_single_clusters.csv'
file = open(test_filename, "r")
lines = csv.reader(file)
clusters = list(lines)

for i in range(len(clusters)):
    clusters[i] = list(map(int, clusters[i]))

hy_value = 0
for i in keyword_count:
    hy_value -= (i/total_papers)*math.log2(i/total_papers)

hc_value = 0
for cluster in clusters:
    hc_value -= (len(cluster)/total_papers)*math.log2(len(cluster)/total_papers)

hyc_value = 0
for cluster in clusters:
    temp_hyc_value = 0
    total = float(len(cluster))
    cluster_keyword_count = []
    for i in range(len(keywords)):
        cluster_keyword_count.append(0)

    for paper in cluster:
        i = 0
        for i in range(len(keywords)):
            if keywords[i] == paper_to_keyword_mapping[paper]:
                cluster_keyword_count[i] += 1

    for i in cluster_keyword_count:
        p = float(i)/total
        if p != 0:
            temp_hyc_value -= p*math.log2(p)
    temp_hyc_value = temp_hyc_value * (total/total_papers)
    hyc_value = hyc_value + temp_hyc_value

iyc_value = hy_value - hyc_value
NMI = 2*iyc_value/(hy_value+hc_value)

print('NMI Value = ',NMI)

###Bottom Up Hierarchical Clustering - Complete Linkage###
print("\n\n***Bottom Up Hierarchical Clustering - Complete Linkage***\n\n")
test_filename = 'part1_complete_clusters.csv'
file = open(test_filename, "r")
lines = csv.reader(file)
clusters = list(lines)

for i in range(len(clusters)):
    clusters[i] = list(map(int, clusters[i]))

hy_value = 0
for i in keyword_count:
    hy_value -= (i/total_papers)*math.log2(i/total_papers)

hc_value = 0
for cluster in clusters:
    hc_value -= (len(cluster)/total_papers)*math.log2(len(cluster)/total_papers)

hyc_value = 0
for cluster in clusters:
    temp_hyc_value = 0
    total = float(len(cluster))
    cluster_keyword_count = []
    for i in range(len(keywords)):
        cluster_keyword_count.append(0)

    for paper in cluster:
        i = 0
        for i in range(len(keywords)):
            if keywords[i] == paper_to_keyword_mapping[paper]:
                cluster_keyword_count[i] += 1

    for i in cluster_keyword_count:
        p = float(i)/total
        if p != 0:
            temp_hyc_value -= p*math.log2(p)
    temp_hyc_value = temp_hyc_value * (total/total_papers)
    hyc_value = hyc_value + temp_hyc_value

iyc_value = hy_value - hyc_value
NMI = 2*iyc_value/(hy_value+hc_value)

print('NMI Value = ',NMI)

###Girvan Newman Clustering###
print("\n\n***Girvan Newman Clustering******\n\n")
test_filename = 'part2_clusters.csv'
file = open(test_filename, "r")
lines = csv.reader(file)
clusters = list(lines)

for i in range(len(clusters)):
    clusters[i] = list(map(int, clusters[i]))

hy_value = 0
for i in keyword_count:
    hy_value -= (i/total_papers)*math.log2(i/total_papers)

hc_value = 0
for cluster in clusters:
    hc_value -= (len(cluster)/total_papers)*math.log2(len(cluster)/total_papers)

hyc_value = 0
for cluster in clusters:
    temp_hyc_value = 0
    total = float(len(cluster))
    cluster_keyword_count = []
    for i in range(len(keywords)):
        cluster_keyword_count.append(0)

    for paper in cluster:
        i = 0
        for i in range(len(keywords)):
            if keywords[i] == paper_to_keyword_mapping[paper]:
                cluster_keyword_count[i] += 1

    for i in cluster_keyword_count:
        p = float(i)/total
        if p != 0:
            temp_hyc_value -= p*math.log2(p)
    temp_hyc_value = temp_hyc_value * (total/total_papers)
    hyc_value = hyc_value + temp_hyc_value

iyc_value = hy_value - hyc_value
NMI = 2*iyc_value/(hy_value+hc_value)

print('NMI Value = ',NMI)
