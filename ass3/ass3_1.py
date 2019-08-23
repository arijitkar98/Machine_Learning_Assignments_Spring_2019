import csv
import numpy as np

class cluster(object):
    def __init__(self, papers, number):
        self.papers = papers
        self.number = number

def calc_jc(paper1, paper2):
    intersection = 0
    for topic1 in paper1:
        for topic2 in paper2:
            if topic1 == topic2:
                intersection += 1
    union = len(paper1) + len(paper2) - intersection
    coefficient = intersection/union
    return coefficient

def get_similarity(cluster1, cluster2, metric):
	if metric == "single":
		similarity = -10000
	else:
		similarity = 10000

	for paper1 in cluster1.papers:
		for paper2 in cluster2.papers:
			temp = calc_jc(paper_to_topics_mapping[paper1], paper_to_topics_mapping[paper2])
			if metric == "single":
				if temp > similarity:
					similarity = temp
			else:
				if temp < similarity:
					similarity = temp	        
	return similarity

def merge_clusters(cluster1, cluster2):
	temp_papers = []
	for paper in total_clusters_list[cluster1].papers:
	    temp_papers.append(paper)

	for paper in total_clusters_list[cluster2].papers:
	    if paper not in temp_papers:
	            temp_papers.append(paper)

	temp_papers.sort()

	new_cluster = cluster(temp_papers, total_clusters)
	delete_cluster(cluster1)
	delete_cluster(cluster2)

	return new_cluster

def delete_cluster(cluster):
	for i in range(len(final_cluster_list)):
		if final_cluster_list[i] == cluster:
			del final_cluster_list[i]
			break

def add_new_cluster(cluster):
	final_cluster_list.append(new_cluster.number)
	total_clusters_list.append(new_cluster)

###Loading Dataset###
print("***Loading dataset from AAAI.csv***\n\n")
dataset_filename = 'AAAI.csv'
file = open(dataset_filename, "r")
lines = csv.reader(file)
dataset = list(lines)
del dataset[0]

paper_to_topics_mapping = []
for i in range(len(dataset)):
    topic_list = []
    topic_list = dataset[i][2].split('\n')
    paper_to_topics_mapping.append(topic_list)

###Bottom Up Hierarchical Clustering - Single Linkage###
print("***Bottom Up Hierarchical Clustering - Single Linkage******\n\n")
total_clusters_list = []
for i in range(len(dataset)):
	papers_list = []
	papers_list.append(i)
	new_cluster = cluster(papers_list, i)
	total_clusters_list.append(new_cluster)

final_cluster_list = []
for i in range(len(total_clusters_list)):
	final_cluster_list.append(i)
total_clusters = len(final_cluster_list)

print("Initial number of Clusters = 150")

print("\n\n***Merging Clusters***\n\n")

loop_count = 0
while len(final_cluster_list)>9:

	print("Current number of Clusters =", len(final_cluster_list))

	max_similarity = -1
	cluster1 = -1
	cluster2 = -1

	for i in range(len(final_cluster_list)):
		for j in range(i+1, len(final_cluster_list)):
			similarity_measure = get_similarity(total_clusters_list[final_cluster_list[i]], total_clusters_list[final_cluster_list[j]], "single")
			if similarity_measure > max_similarity:
				max_similarity = similarity_measure
				cluster1 = final_cluster_list[i]
				cluster2 = final_cluster_list[j]

	new_cluster = merge_clusters(cluster1, cluster2)
	total_clusters += 1

	add_new_cluster(new_cluster)
	loop_count += 1

print("\n\n***Merging Completed***\n\n")
print("Final number of Clusters =", len(final_cluster_list))
print("The Final Clusters are:")

cluster_list = []
n = 1
for i in final_cluster_list:
	cluster_list.append(total_clusters_list[i].papers)
	print("\n\nCluster", n, "\n")
	print("Number of Papers in Cluster =",len(total_clusters_list[i].papers))
	print(total_clusters_list[i].papers)
	n += 1

print("\n\n***Writing clusters to file part1_single_clusters.csv***")
f = open("part1_single_clusters.csv","w")
wr = csv.writer(f)
wr.writerows(cluster_list)


###Bottom Up Hierarchical Clustering - Complete Linkage###
print("\n\n***Bottom Up Hierarchical Clustering - Complete Linkage***\n\n")
total_clusters_list = []
for i in range(len(dataset)):
	papers_list = []
	papers_list.append(i)
	new_cluster = cluster(papers_list, i)
	total_clusters_list.append(new_cluster)

final_cluster_list = []
for i in range(len(total_clusters_list)):
	final_cluster_list.append(i)
total_clusters = len(final_cluster_list)

print("Initial number of Clusters = 150")

print("\n\n***Merging Clusters***\n\n")

loop_count = 0
while len(final_cluster_list)>9:

	print("Current number of Clusters =", len(final_cluster_list))

	max_similarity = -1
	cluster1 = -1
	cluster2 = -1

	for i in range(len(final_cluster_list)):
		for j in range(i+1, len(final_cluster_list)):
			similarity_measure = get_similarity(total_clusters_list[final_cluster_list[i]], total_clusters_list[final_cluster_list[j]], "complete")
			if similarity_measure > max_similarity:
				max_similarity = similarity_measure
				cluster1 = final_cluster_list[i]
				cluster2 = final_cluster_list[j]

	new_cluster = merge_clusters(cluster1, cluster2)
	total_clusters += 1

	add_new_cluster(new_cluster)
	loop_count += 1

print("\n\n***Merging Completed***\n\n")
print("Final number of Clusters =", len(final_cluster_list))
print("The Final Clusters are:")

cluster_list = []
n = 1
for i in final_cluster_list:
	cluster_list.append(total_clusters_list[i].papers)
	print("\n\nCluster", n, "\n")
	print("Number of Papers in Cluster =",len(total_clusters_list[i].papers))
	print(total_clusters_list[i].papers)
	n += 1

print("\n\n***Writing clusters to file part1_complete_clusters.csv***")
f = open("part1_complete_clusters.csv","w")
wr = csv.writer(f)
wr.writerows(cluster_list)
