from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt



class Graph(defaultdict):
    def __init__(self, input_file, sep=" ", header=False, undirect=True):
        """
        文件格式:
            node1, node2, dist(node1, node2)
            ...
        默认是无向图
        """
        super(Graph, self).__init__(dict)

        self.edges_num = 0

        with open(input_file) as f:
            if header:
                f.readline()
            for line in f:
                line = line.strip().split(sep)
                self[line[0]][line[1]] = float(line[2])
                self.edges_num += 1
                if undirect:
                    self[line[1]][line[0]] = float(line[2])
                    self.edges_num += 1

    def number_of_edges(self):
        """
        返回边的总数
        """
        return self.edges_num

    def edges(self):
        """
        返回边的列表
        """
        edges_list = []
        for node1 in self:
            for node2 in self[node1]:
                edges_list.append((node1, node2))
        return edges_list

    def edges_weight(self):
        """
        返回从小到大进行排序后的边权
        """
        weight_list = []
        for edge in self.edges():
            node1, node2 = edge
            weight_list.append([node1, node2, self[node1][node2]])
        weight_list = sorted(weight_list, key=lambda x:x[2])
        return weight_list

    def nodes(self):
        """
        返回所有的节点
        """
        return list(self)

    def number_of_nodes(self):
        """
        返回节点的个数
        """
        return len(self)

    def get_weight(self, node1, node2):
        """
        得到node1, node2之间的距离
        """
        return self[node1][node2]

    def get_weights(self):
        """
        返回所有的权重，顺序是self.edges的顺序
        """
        weights = []
        for edge in self.edges():
            weights.append(self.get_weight(edge[0], edge[1]))
        return weights


if __name__=="__main__":

    network_file = "./data/example_distances.dat"
    percent = 2.0
    output_file = "./data/example_distances.community"

    # 加载图数据
    G = Graph(network_file)

    # 求dc
    position = round(G.number_of_edges()*percent/100)
    dc = G.edges_weight()[position][2]
    print("average percentage of neighbours (hard coded): {}".format(percent))
    print("Computing Rho with gaussian kernel of radius: {}".format(dc))

    # 计算高斯核
    nodes = G.nodes()
    rho = [0. for _ in range(G.number_of_nodes())]
    for i in range(G.number_of_nodes()-1):
        for j in range(i+1, G.number_of_nodes()):
            node_i = nodes[i]
            node_j = nodes[j]
            dist_ij = G.get_weight(node_i, node_j)
            rho[i] = rho[i]+np.exp(-(dist_ij/dc)*(dist_ij/dc))
            rho[j] = rho[j]+np.exp(-(dist_ij/dc)*(dist_ij/dc))

    # 最大的距离
    maxd = max(G.get_weights())

    # 对rho从大到小进行排序，得到它原本的顺序的列表
    # 默认的argsort是从小到大排列的，因此需要reverse
    ordrho = list(reversed(np.argsort(rho)))

    # 计算delta
    delta = [0. for _ in range(G.number_of_nodes())]
    nneigh = [0 for _ in range(G.number_of_nodes())]
    delta[ordrho[0]] = -1
    nneigh[ordrho[0]] = 0

    for ii in range(1,G.number_of_nodes()):
        delta[ordrho[ii]] = maxd
        for jj in range(0,ii):
            dist_ii_jj = G.get_weight(nodes[ordrho[ii]], nodes[ordrho[jj]])
            if dist_ii_jj<delta[ordrho[ii]]:
                delta[ordrho[ii]] = dist_ii_jj
                # 记录rho值更大的数据点中与ordrho[ii]距离最近的点的编号
                nneigh[ordrho[ii]] = ordrho[jj]
    delta[ordrho[0]] = max(delta)

    # 画图
    plt.figure()
    plt.xlabel('rho')
    plt.ylabel("delta")
    plt.scatter(rho, delta, marker="o")
    plt.show()

    # 根据所画的图确定最小的rho值以及delta值
    #rhomin = 19.9
    #deltamin = 0.029
    rhomin = 32.0
    deltamin = 0.1
    N_cluster = 0
    # 初始化类别
    cl = [-1 for _ in range(G.number_of_nodes())]
    # 记录聚类中心
    icl = [-1 for _ in range(G.number_of_nodes())]

    for i in range(G.number_of_nodes()):
        if rho[i]>rhomin and delta[i]>deltamin:
            cl[i] = N_cluster
            icl[N_cluster] = i
            # 聚类的编号从0开始
            N_cluster += 1
    print("Number of Clusters: {}".format(N_cluster))
    print("Performing assignation")

    # 给各个节点打标签
    for i in range(G.number_of_nodes()):
        if cl[ordrho[i]] == -1:
            # 用与ordrho[i]距离最近的点给ordrho[i]打标签
            # 循环的顺序是rho从大到小开始的，所以必然打标签不会
            # 遇到nneigh[ordrho]为-1的情况
            # 因为在构造nneigh[ordrho[i]]是比ordrho[i]的rho值要更大的值
            # 而当rho是最大的时候，会被直接打标签
            cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

    # 处理光晕点
    # 如果有两个以上的cluster，则进一步将其分为光晕点以及聚类核心
    halo = cl.copy()    # 因为cl中不含有列表，因此直接copy即可，没必要deepcopy

    # 对于一个固定的cluster先确定它的边界区域
    # 这个区域由这样的数据点构成：
    # 它们本身属于该cluster，但在与其距离不超过dc的范围内，
    # 存在属于其他cluster的数据点，利用边界区域，这个cluster就可以计算出一个
    # 平均局部密度的上界
    if N_cluster>1:
        bord_rho = [0. for _ in range(N_cluster)]
        for i in range(G.number_of_nodes()-1):
            for j in range(i+1, G.number_of_nodes()):
                node1 = nodes[i]
                node2 = nodes[j]
                dist_ij = G.get_weight(node1, node2)
                if cl[i]!=cl[j] and dist_ij<=dc:
                    rho_aver = (rho[i]+rho[j])/2.
                    if rho_aver>bord_rho[cl[i]]:
                        bord_rho[cl[i]] = rho_aver
                    if rho_aver>bord_rho[cl[j]]:
                        bord_rho[cl[j]] = rho_aver
        # 标识光晕点
        for i in range(G.number_of_nodes()):
            #print("rho", rho[i])
            #print("cl", cl[i])
            #print("bord_rho", bord_rho[cl[i]])
            if rho[i]<bord_rho[cl[i]]:
                halo[i] = -1     # 将光晕点的值设置为0

    # 统计各个cluster的节点数，及其中的聚类中心的个数以及光晕点的个数
    for i in range(N_cluster):
        nc = 0
        nh = 0
        for j in range(G.number_of_nodes()):
            if cl[j] == i:
                nc = nc+1
            if halo[j] == i:
                nh=nh+1
        print("CLUSTER: {} CENTER: {} ELEMENTS: {} CORE: {} HALO: {}".format(i, nodes[icl[i]], nc, nh, nc-nh))

    # 将聚类结果输出到文件
    with open(output_file, "w") as fout:
        for i in range(G.number_of_nodes()):
            print("{} {} {}".format(nodes[i], cl[i], halo[i]), file=fout)
