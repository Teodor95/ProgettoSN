from collections import defaultdict

import networkx as nx
from matplotlib import pylab
from sklearn import cluster
import matplotlib
import matplotlib.pyplot as plt

class Graph:
    G = None
    RG = None
    number_of_nodes_G = None
    number_of_edges_G = None

    def __init__(self):
        print('Start Load Data')
        self.G = nx.read_adjlist('Files/com-amazon.ungraph.txt')
        self.groudTruthALLC = nx.read_adjlist('Files/com-amazon.all.dedup.cmty.txt')
        self.groudTruthTOPC = nx.read_adjlist('Files/com-amazon.top5000.cmty.txt')
        self.number_of_nodes_G = self.G.number_of_nodes()
        self.number_of_edges_G = self.G.number_of_edges()
        self.RG = nx.gnm_random_graph(self.number_of_nodes_G, self.number_of_edges_G)
        print('Finish Load Data')


def listToDict(list):
    listdict = {}
    for i in range(len(list)):
        listdict[i] = list[i]
    return listdict


def save_graph(graph, file_name):
    print('sono entrato qui')
    # initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig


def graphToEdgeMatrix(G):
    # Initialize Edge Matrix
    edgeMat = [[0 for x in range(len(G))] for y in range(len(G))]

    # For loop to set 0 or 1 ( diagonal elements are set to 1)
    for node in G:
        tempNeighList = G.neighbors(node)
        for neighbor in tempNeighList:
            edgeMat[node][neighbor] = 1
        edgeMat[node][node] = 1

    return edgeMat


def subgraph():
    nx.draw(graph.G)
    plt.savefig('mypgraph.png')

    degrees = [graph.G.degree(n) for n in graph.G.nodes()]
    plt.hist(degrees)
    plt.savefig('degree.png')
    # plt.show()


def drawCommunities(G, partition, pos):
    # G is graph in networkx form
    # Partition is a dict containing info on clusters
    # Pos is base on networkx spring layout (nx.spring_layout(G))

    # For separating communities colors
    dictList = defaultdict(list)
    nodelist = []
    for node, com in partition.items():
        dictList[com].append(node)

    # Get number of Communities
    size = len(set(partition.values()))

    # For loop to assign communities colors
    for i in range(size):

        amplifier = i % 3
        multi = (i / 3) * 0.3

        red = green = blue = 0

        if amplifier == 0:
            red = 0.1 + multi
        elif amplifier == 1:
            green = 0.1 + multi
        else:
            blue = 0.1 + multi

        # Draw Nodes
        nx.draw_networkx_nodes(G, pos,
                               nodelist=dictList[i],
                               node_color=[0.0 + red, 0.0 + green, 0.0 + blue],
                               node_size=400,
                               alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)


def cluster_detection():
    # Spectral Clustering Model
    spectral = cluster.SpectralClustering(n_clusters=kClusters, affinity="precomputed", n_init=200)
    spectral.fit(edgeMat)

    # print "spectral", list(spectral.labels_)
    # Transform our data to list form and store them in results list
    results.append(list(spectral.labels_))

    # -----------------------------------------

    # Agglomerative Clustering Model
    agglomerative = cluster.AgglomerativeClustering(n_clusters=kClusters, linkage="ward")
    agglomerative.fit(edgeMat)

    # Transform our data to list form and store them in results list
    results.append(list(agglomerative.labels_))

    # -----------------------------------------

    # K-means Clustering Model
    kmeans = cluster.KMeans(n_clusters=kClusters, n_init=200)
    kmeans.fit(edgeMat)

    # Transform our data to list form and store them in results list
    results.append(list(kmeans.labels_))
    # print kmeans.labels_


def print_initial_data():
    f = open('results.txt', 'w')
    # Quello che non deve accadere è che ci sia un cluster più alto nel random che nella rete scale free
    f.write('Average Clustering Graph G \n')
    f.write(str(nx.average_clustering(graph.G)))
    f.flush()
    f.write('Average Clustering Graph RG\n')
    f.write(str(nx.average_clustering(graph.RG)))
    f.flush()
    f.write('APL Graph G\n')
    f.write(str(nx.average_shortest_path_length(graph.G)))
    f.flush()
    f.write('APL Graph RG\n')
    f.write(str(nx.average_shortest_path_length(graph.RG)))
    f.flush()
    f.write('Density G\n')
    f.write(str(nx.density(graph.G)))
    f.flush()
    f.write('Density RG\n')
    f.write(str(nx.density(graph.RG)))
    f.close()


if __name__ == '__main__':
    graph = Graph()
    print(nx.diameter(graph.G))
   # print_initial_data()
   # nx.write_edgelist(graph.G, "test.csv", delimiter=',', data=False)

   # save_graph(graph.G, "mypgraps.pdf")
    exit(33)
    subgraph()
    exit(22)
    print_initial_data()
    exit(23)

    edgeMat = graphToEdgeMatrix(graph.G)
    affinity = cluster.affinity_propagation(S=edgeMat, max_iter=200, damping=0.6)
    print(affinity)
    exit()
    pos = nx.circular_layout(graph.G)
    # nx.draw(l, pos, node_size=60, font_size=8)
    plt.savefig('hierarchy.png')

    exit(1)
    G_eigenvector_centrality = nx.eigenvector_centrality_numpy(graph.G)
    # print(nx.density(graph.G))
    # print(['%s %0.2f'%(node, G_eigenvector_centrality[node]) for node in G_eigenvector_centrality])

    exit(1)
    # groundTruth = nx.read_adjlist('Files/com-lj.top5000.cmty.txt')
    # groundTruth_top = nx.read_adjlist('Files/com-lj.top5000.cmty.txt')
    # print(type(groundTruth))
    kClusters = 500
    results = []
    print('Affinity propagation')

    print(affinity)
