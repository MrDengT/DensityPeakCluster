import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np


def read_community(community_network, sep=" ", header=False):
    """
    读取节点的社区
    """
    node2community = dict()
    community2nodes = dict()
    with open(community_network) as f:
        if header:
            f.readline()
        for line in f:
            line = line.strip().split(sep)
            node2community[line[0]] = int(line[1])
            if int(line[1]) not in community2nodes:
                community2nodes[int(line[1])] = []
            community2nodes[int(line[1])].append(line[0])
    return node2community, community2nodes


def plot_community(origin_network, community_network):
    """
    根据社区对网络节点上色
    """
    G = nx.read_weighted_edgelist(origin_network)
    node2community, community2nodes = read_community(community_network)
    #pos = nx.spring_layout(G, iterations=100)
    #pos = nx.spectral_layout(G)

    # 读取距离
    nodes = list(G.nodes())
    dp = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i!=j:
                node1 = nodes[i]
                node2 = nodes[j]
                dp[i,j] = G[node1][node2]["weight"]

    # 用多维缩放算法求每个点的坐标
    mds = MDS(max_iter=200, eps=1e-4, n_init=1, dissimilarity="precomputed")
    dp_mds = mds.fit_transform(dp)
    pos = dict()
    for i,node in enumerate(nodes):
        pos[node] = dp_mds[i]
    #print(pos)

    for i, community in enumerate(community2nodes):
        list_nodes = community2nodes[community]
        color_list = [i/len(community2nodes) for _ in range(len(list_nodes))]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=30, node_color=color_list, cmap=plt.get_cmap("Set1"), vmin=0.0, vmax=1.0)

    #nx.draw_networkx_edges(G, pos, alpha=0.01)
    plt.show()



if __name__=="__main__":

    origin_network = "./data/example_distances.dat"
    community_network = "./data/example_distances.community"

    plot_community(origin_network, community_network)
