import operator
from collections import defaultdict

import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import cm, colors
from sklearn import cluster
from sklearn import metrics


class MYGraph:
    G = None
    RG = None

    gtype = None
    groud_truth_array = None

    def __init__(self):
        self.graph_type_undirected = nx.Graph()
        self.graph_type_directed = nx.DiGraph()
        print('Start Load Data')
        self.G_d = nx.read_edgelist('Files/Email-eu-core.txt', create_using=self.graph_type_directed, nodetype=int)
        self.G_d_not_iso = None

        self.G_u = nx.read_edgelist('Files/Email-eu-core.txt', create_using=self.graph_type_undirected, nodetype=int)

        self.number_of_edges = self.G_d.number_of_edges()
        self.number_of_nodes = self.G_d.number_of_nodes()
        self.RG_d = nx.gnm_random_graph(self.number_of_nodes, self.number_of_edges, directed=True)
        self.RG_u = nx.gnm_random_graph(self.number_of_nodes, self.number_of_edges, directed=False)

        self.read_ground_truth()
        print('Finish Load Data')

    def read_ground_truth(self):
        f = open('Files/email-Eu-core-department-labels.txt', "r")
        lines = f.readlines()
        f.close()

        temp = []
        # Remove first column because is index of node
        for k in lines:
            line = k.split(" ")
            temp.append(int(line[1].strip()))
        self.groud_truth_array = np.array(temp)


def listToDict(list):
    listdict = {}
    for i in range(len(list)):
        listdict[i] = list[i]
    return listdict


def graphToEdgeMatrix(G):
    # Initialize edge matrix with zeros
    edge_mat = np.zeros((len(G), len(G)), dtype=int)

    # Loop to set 0 or 1 (diagonal elements are set to 1)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[node][neighbor] = 1
        edge_mat[node][node] = 1

    return edge_mat


def cluster_detection(_type=None):
    k_clusters = 42
    algorithms = {}

    # Spectral Clustering Model
    if _type == 'spectral':
        # weak_g = nx.weakly_connected_components(graph.G_d)
        # index = 0
        # for item in weak_g:
        #     index += 1
        # k_clusters = 42

        edgeMat = nx.adjacency_matrix(graph.G_d)
        algorithms['spectral'] = cluster.SpectralClustering(n_clusters=k_clusters, affinity="precomputed", n_init=200, assign_labels='discretize')
        algorithms['spectral'].fit_predict(edgeMat)

        adjusteted_rand_score = metrics.adjusted_rand_score(graph.groud_truth_array, algorithms['spectral'].labels_)
        print(adjusteted_rand_score)
    exit()

    # Agglomerative Clustering Model
    algorithms['agglomerative'] = cluster.AgglomerativeClustering(n_clusters=kClusters, linkage="ward")
    algorithms['agglomerative'].fit(edgeMat)

    # K-means Clustering Model
    algorithms['kmeans'] = cluster.KMeans(n_clusters=k_clusters, n_init=200)
    algorithms['kmeans'].fit(edgeMat)


def print_initial_data(directed):
    f = open('results.txt', 'a')
    if directed:
        f.write('Average Clustering Graph D_G \n')
        f.write(str(nx.average_clustering(graph.G_d)) + "\n")
        f.flush()
        f.write('Average Clustering Graph D_RG \n')
        f.write(str(nx.average_clustering(graph.RG_d)) + "\n")
        f.flush()
        f.write('APL Graph D_G\n')
        f.write(str(nx.average_shortest_path_length(graph.G_d_not_iso)) + "\n")
        f.flush()
        f.write('APL Graph D_RG\n')
        RD_not_iso = nx.gnm_random_graph(graph.G_d_not_iso.number_of_nodes(), graph.G_d_not_iso.number_of_edges(),
                                         directed=True)
        f.write(str(nx.average_shortest_path_length(RD_not_iso)) + "\n")
        f.flush()
        f.write('Density D_G\n')
        f.write(str(nx.density(graph.G_d)) + "\n")
        f.flush()
        f.write('Density D_RG\n')
        f.write(str(nx.density(graph.RG_d)) + "\n")

        f.write('Reciprocità D_G\n')
        f.write(str(nx.reciprocity(graph.G_d)) + "\n")
        f.flush()
        f.write('Reciprocità D_RG\n')
        f.write(str(nx.reciprocity(graph.RG_d)) + "\n")

        # f.write('Diametro D_G\n')
        # f.write(str(nx.diameter(graph.G_d)) + "\n")
        # f.flush()
        # f.write('Diametro D_RG\n')
        # f.write(str(nx.diameter(graph.RG_d)) + "\n")

    else:
        f.write('Average Clustering U_Graph G \n')
        f.write(str(nx.average_clustering(graph.G_u)) + "\n")
        f.flush()
        f.write('Average Clustering U_Graph RG\n')
        f.write(str(nx.average_clustering(graph.RG_u)) + "\n")
        f.flush()
        f.write('APL Graph U_G\n')
        f.write(str(nx.average_shortest_path_length(graph.G_u)) + "\n")
        f.flush()
        f.write('APL Graph U_RG\n')
        f.write(str(nx.average_shortest_path_length(graph.RG_u)) + "\n")
        f.flush()
        f.write('Density U_G\n')
        f.write(str(nx.density(graph.G_u)) + "\n")
        f.flush()
        f.write('Density U_RG\n')
        f.write(str(nx.density(graph.RG_u)) + "\n")
    f.close()


def reciprocity_directed_graph():
    print(nx.reciprocity(graph.G_d))


def ecentricity_directed_graph():
    # Grafico sull'Eccentricita e il Diametro
    diameter = nx.diameter(graph.G_d)
    print(diameter)
    eccs = nx.eccentricity(graph.G_d)
    eccs = eccs.values()
    eccs = sorted(eccs)
    plt.plot(range(len(graph.G_d)), eccs)
    plt.title('90th Percentile Diameter')
    plt.xlabel("Node")
    plt.ylabel("Eccentricity")
    ninethPercentile = math.floor(len(graph.G_d) * 90 / 100)
    plt.axvline(x=ninethPercentile, alpha=0.5, color='r')
    plt.text(x=ninethPercentile - 55, y=5.5, s='90th')
    # plt.show()
    plt.savefig("eccentricity.png")


# Reciprocita
# for node in G:
#    print(nx.reciprocity(G, node))
# print(nx.reciprocity(G))


def degree_distribution():
    plt.figure(figsize=(10, 10))
    degree_sequence = sorted([d for n, d in graph.G_d.degree()], reverse=True)  # degree sequence
    sns.set(color_codes=True)
    sns.set(style="white", palette="dark")
    sns.displot(degree_sequence, kde=True, linewidth=0.5, binwidth=30)
    plt.xlabel("Degree\n")
    plt.ylabel("Numero nodi")
    plt.savefig("aab.png")


def draw(G, pos, measures, measure_name, label_pos=None, ):
    plt.figure(num=None, figsize=(8, 8), dpi=1000, facecolor='w', edgecolor='k')
    nodes = nx.draw_networkx_nodes(G, pos, node_size=50, cmap=plt.cm.plasma,
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys(), linewidths=0)
    nx.draw_networkx_edges(G, pos, alpha=0.1)

    if label_pos is None:
        nodes.set_norm(mcolors.Normalize())
        # nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01,vmin=0, vmax=1, base=10))
    if label_pos != None:
        labels = nx.draw_networkx_labels(G, label_pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.savefig(measure_name + ".png")


def plot_centrality_only_best_nodes(G, number, _type, name):
    if _type == 'DEGREE':
        nodes_dict = nx.degree_centrality(G)
        G2 = nx.subgraph(G, [x for x in G.nodes() if nodes_dict[x] > float(number)])
        pos = nx.spring_layout(G2, k=4 / math.sqrt(G.order()), iterations=50, scale=100)

    if _type == 'CLOSENESS':
        nodes_dict = nx.closeness_centrality(G)
        G2 = nx.subgraph(G, [x for x in G.nodes() if nodes_dict[x] > float(number)])
        pos = nx.spring_layout(G2, k=4 / math.sqrt(G.order()), iterations=50, scale=100)

    if _type == 'BETWEENNESS':
        nodes_dict = nx.betweenness_centrality(G)
        G2 = nx.subgraph(G, [x for x in G.nodes() if nodes_dict[x] > float(number)])
        pos = nx.spring_layout(G2, k=10 / math.sqrt(G.order()), iterations=50, scale=100)

    if _type == 'Eigenvector':
        nodes_dict = nx.eigenvector_centrality(G)
        G2 = nx.subgraph(G, [x for x in G.nodes() if nodes_dict[x] > float(number)])
        pos = nx.spring_layout(G2, k=10 / math.sqrt(G.order()), iterations=50, scale=100)

    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 5)

    new_dict = {}
    for k, v in nodes_dict.items():
        if k in G2.nodes():
            new_dict[k] = v
    draw(G2, pos, new_dict, name, pos_attrs)


def plot_centrality(G):
    pos = nx.spring_layout(G, k=5 / math.sqrt(G.order()), iterations=20, scale=100)

    # Degree Centrality
    draw(G, pos, nx.degree_centrality(G), 'Degree Centrality FN')
    draw(G, pos, nx.degree_centrality(graph.G_d_not_iso), 'Degree Centrality SCC')
    plot_centrality_only_best_nodes(G, 0.3, 'DEGREE', 'Degree Centrality High Degree FN')
    plot_centrality_only_best_nodes(graph.G_d_not_iso, 0.3, 'DEGREE', 'Degree Centrality High Degree SCC')

    # Closeness Centrality
    draw(G, pos, nx.closeness_centrality(G), 'Closeness Centrality FN ')
    draw(G, pos, nx.closeness_centrality(graph.G_d_not_iso), 'Closeness Centrality SCC ')
    plot_centrality_only_best_nodes(G, 0.5, 'CLOSENESS', 'Closeness Centrality High Closeness FN')
    plot_centrality_only_best_nodes(graph.G_d_not_iso, 0.5, 'CLOSENESS', 'Closeness Centrality High Closeness SCC')

    # Betweenness Centrality
    draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality FN ')
    draw(G, pos, nx.betweenness_centrality(graph.G_d_not_iso), 'Betweenness Centrality SCC ')
    plot_centrality_only_best_nodes(G, 0.020, 'BETWEENNESS', 'Betweenness Centrality High Betweenness FN')
    plot_centrality_only_best_nodes(graph.G_d_not_iso, 0.020, 'BETWEENNESS',
                                    'Betweenness Centrality High Betweenness SCC')

    # Eigenvector Centrality
    draw(G, pos, nx.eigenvector_centrality(G), 'Eigenvector Centrality FN ')
    draw(G, pos, nx.eigenvector_centrality(graph.G_d_not_iso), 'Eigenvector Centrality SCC ')
    plot_centrality_only_best_nodes(G, 0.20, 'Eigenvector', 'Eigenvector Centrality High Eigenvector FN')
    plot_centrality_only_best_nodes(graph.G_d_not_iso, 0.20, 'Eigenvector',
                                    'Eigenvector Centrality High Eigenvector SCC')


def print_betweenness_centrality():
    bet_FN = nx.betweenness_centrality(graph.G_d)
    bet_SCC = nx.betweenness_centrality(graph.G_d_not_iso)
    nodes = [160, 107, 86, 62, 5]

    f = open("betweeness_centrality.txt", "w")
    f.write("FN: \n")
    bet_FN_med = 0
    for k, v in bet_FN.items():
        if k in nodes:
            f.write(str(k) + "\t" + str(v) + "\n")
        bet_FN_med += v
    bet_FN_med = bet_FN_med / len(bet_FN)
    f.write("Media FN: " + str(bet_FN_med) + "\n")

    bet_SCC_med = 0
    f.write("SCC: \n")
    for k, v in bet_SCC.items():
        if k in nodes:
            f.write(str(k) + "\t" + str(v) + "\n")
        bet_SCC_med += v
    bet_SCC_med = bet_SCC_med / len(bet_SCC)
    f.write("Media SCC: " + str(bet_SCC_med) + "\n")
    f.close()


def print_eigenvector_centrality():
    ei_FN = nx.eigenvector_centrality(graph.G_d)
    ei_SCC = nx.eigenvector_centrality(graph.G_d_not_iso)

    nodes = [160, 107, 432, 62, 121]

    f = open("eigenvector_centrality.txt", "w")
    f.write("FN: \n")
    for k, v in ei_FN.items():
        if k in nodes:
            f.write(str(k) + "\t" + str(v) + "\n")

    f.write("SCC: \n")
    for k, v in ei_SCC.items():
        if k in nodes:
            f.write(str(k) + "\t" + str(v) + "\n")
    f.close()


def print_degree_centrality():
    print(graph.G_d.number_of_nodes())
    print(graph.G_d_not_iso.number_of_nodes())
    nodes = [160, 86, 121, 62, 107]
    FN_degree = nx.degree_centrality(graph.G_d)
    FN_degree_IN = nx.in_degree_centrality(graph.G_d)
    FN_degree_OUT = nx.out_degree_centrality(graph.G_d)
    SCC_degree = nx.degree_centrality(graph.G_d_not_iso)
    SCC_degree_IN = nx.in_degree_centrality(graph.G_d_not_iso)
    SCC_degree_OUT = nx.out_degree_centrality(graph.G_d_not_iso)

    f = open("degree_centrality_best_noodes.txt", "w")

    for k in nodes:
        f.write(str(k) + "\n")
        f.write(str("FN degree: ") + str(FN_degree[k]) + "\n")
        f.write(str("FN degree IN: ") + str(FN_degree_IN[k]) + "\n")
        f.write(str("FN degree OUT: ") + str(FN_degree_OUT[k]) + "\n")

        f.write(str("SCC degree: ") + str(SCC_degree[k]) + "\n")
        f.write(str("SCC degree IN: ") + str(SCC_degree_IN[k]) + "\n")
        f.write(str("SCC degree OUT: ") + str(SCC_degree_OUT[k]) + "\n")

    f.close()


def print_closeness_centrality():
    closeness = nx.closeness_centrality(graph.G_d_not_iso)
    sorted_u = sorted(closeness.items(), key=operator.itemgetter(1))

    sorted_d = dict(sorted(closeness.items(), key=operator.itemgetter(1), reverse=True))

    new_dict = {}
    new_dict[160] = sorted_d[160]

    index = 0
    for k, v in sorted_d.items():
        if index < 6:
            new_dict[k] = v
            index += 1
        else:
            break
    new_dict[843] = sorted_d[843]

    f = open("closeness_centrality_best_noodes.txt", "w")
    for k, v in new_dict.items():
        f.write(str(k) + "\t" + str(v) + "\n")

    f.close()


def directed_graph():
    bigG = sorted(nx.strongly_connected_components(graph.G_d), key=len, reverse=True)

    to_remove = []
    for k in bigG:
        if len(k) == 1:
            to_remove.append(k)
    graph.G_d_not_iso = graph.G_d.copy()
    for node in to_remove:
        graph.G_d_not_iso.remove_nodes_from(node)

    # print_initial_data(directed=True)
    # plot_centrality(graph.G_d)
    # print_degree_centrality()
    # print_closeness_centrality()
    # print_betweenness_centrality()
    # print_eigenvector_centrality()

    cluster_detection('spectral')
    # degree_distribution()

    # ecentricity_directed_graph()
    # print(nx.degree_centrality(graph.G_d))
    # print_initial_data(directed=True)
    # reciprocity_directed_graph()


def undirected_graph():
    bigG = sorted(nx.connected_components(graph.G_u), key=len, reverse=True)
    print(bigG)
    to_remove = []
    for k in bigG:
        if len(k) == 1:
            to_remove.append(k)

    for node in to_remove:
        graph.G_u.remove_nodes_from(node)
    print_initial_data(directed=False)


def draw_communities(G, membership, pos):
    fig, ax = plt.subplots(figsize=(16, 9))

    # Convert membership list to a dict where key=club, value=list of students in club
    club_dict = defaultdict(list)
    for student, club in enumerate(membership):
        club_dict[club].append(student)

    # Normalize number of clubs for choosing a color
    norm = colors.Normalize(vmin=0, vmax=len(club_dict.keys()))

    for club, members in club_dict.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=members,
                               node_color=cm.jet(norm(club)),
                               node_size=200,
                               alpha=0.8,
                               ax=ax)

    # Draw edges (social connections) and show final plot
    plt.title("Zachary's Karate Club")
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)


if __name__ == '__main__':

    graph = MYGraph()
    directed_graph()
    else:
        undirected_graph()
    exit()
