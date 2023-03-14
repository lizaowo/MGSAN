import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

## 以下是自己写的各个点和中点的连接
hand = [(1, 2), (2, 21), (3, 21), (4, 21), (5, 21), (6, 21), (7, 21),
                    (8, 21), (9, 21), (10, 21), (11, 21), (12, 21), (13, 1),
                    (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1),
                    (20, 1), (22, 21), (23, 21), (24, 21), (25, 21)]
inward_hand = [(i - 1, j - 1) for (i, j) in hand]


# 这个是定义邻接矩阵A的
class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.Adim1 = tools.get_dim1(self.num_node, self.neighbor)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_dim1':
            A = tools.get_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial_way2':
            A = tools.get_graph(num_node, self_link, neighbor)
            A = tools.normalize_adjacency_matrix(A)
        else:
            raise ValueError()
        return A
