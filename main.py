import networkx as nx
from sklearn import cluster


class Graph:
    G = None
    RG = None
    number_of_nodes_G = None
    number_of_edges_G = None

    def __init__(self):
        self.G = nx.read_adjlist('Files/com-amazon.ungraph.txt')
        self.number_of_nodes_G = self.G.number_of_nodes()
        self.number_of_edges_G = self.G.number_of_edges()
        self.RG = nx.gnm_random_graph(self.number_of_nodes_G, self.number_of_edges_G)



def listToDict(list):
    listdict = {}
    for i in range(len(list)):
        listdict[i] = list[i]
    return listdict


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


if __name__ == '__main__':
    graph = Graph()
    print(nx.average_shortest_path_length(graph.G))
    print(nx.average_shortest_path_length(graph.RG))
    exit(1)
    # groundTruth = nx.read_adjlist('Files/com-lj.top5000.cmty.txt')
    # groundTruth_top = nx.read_adjlist('Files/com-lj.top5000.cmty.txt')
    # print(type(groundTruth))
    kClusters = 500
    results = []
    edgeMat = graphToEdgeMatrix(graph.G)
