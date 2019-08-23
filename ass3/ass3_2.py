import networkx
import csv

def calc_jc(paper1, paper2):
    intersection = 0
    for topic1 in paper1:
        for topic2 in paper2:
            if topic1 == topic2:
                intersection += 1
    union = len(paper1) + len(paper2) - intersection
    coefficient = intersection/union
    return coefficient

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


###Create Graph###
print("***Creating and Populating Graph***\n\n")
graph = networkx.Graph()
clusters_list = []
for i in range(150):
    clusters_list.append(i)
graph.add_nodes_from(clusters_list)

for i in range(len(clusters_list)):
    for j in range(i+1, len(clusters_list)):
        if calc_jc(paper_to_topics_mapping[clusters_list[i]], paper_to_topics_mapping[clusters_list[j]]) <= 0.15:
        	continue
        else:
            graph.add_edge(clusters_list[i], clusters_list[j])

###Removing Edges to get final clusters###
print("***Removing Edges to get final clusters***")
while networkx.number_connected_components(graph) < 9:

	temp = networkx.edge_betweenness_centrality(graph, normalized=True, weight=None)
	items = temp.items()
	max_centrality = 0
	for row in items:
		if row[1] <= max_centrality:
			continue
		else:
			edge1 = row[0][0]
			edge2 = row[0][1]
			max_centrality = row[1]

	graph.remove_edge(edge1, edge2)

print("\n\n***Clustering Completed***\n\n")
print("Final number of Clusters =", networkx.number_connected_components(graph))
print("The Final Clusters are:")

subgraph = networkx.connected_component_subgraphs(graph)
subgraph_list = []
for sg in subgraph:
    subgraph_list.append(sg.nodes)

final_cluster_list = []
n = 1
for row in subgraph_list:
	final_cluster_list.append(row)
	print("\n\nCluster", n, "\n")
	print("Number of Papers in Cluster =",len(row))
	print(row)
	n += 1

print("\n\n***Writing clusters to file part2_clusters.csv***")
f = open("part2_clusters.csv","w")
wr = csv.writer(f)
wr.writerows(final_cluster_list)
