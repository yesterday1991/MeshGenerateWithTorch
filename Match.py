from OCC.Core.gp import gp_Pnt
import MyGeometry
import torch
import networkx as nx

use_networkx = True

class MyGraph:
    """
    构建图用于求几何边在网格上对应的网格点
    """
    def __init__(self):
        # 用字典来表示邻接表，键是节点，值是相邻节点及其权重
        self.graph = {}

    def add_node(self, node):
        # 如果节点不在图中，添加节点
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, node1, node2, weight):
        # 添加边，双向图（无向图）需要互相添加
        self.add_node(node1)
        self.add_node(node2)
        self.graph[node1][node2] = weight
        self.graph[node2][node1] = weight

    def update_edge_weight(self, node1, node2, new_weight):
        """
         修改边的权重，如果边存在
        :param node1:           边的顶点1
        :param node2:           边的顶点2
        :param new_weight:      权重
        """
        if node1 in self.graph and node2 in self.graph[node1]:
            self.graph[node1][node2] = new_weight
            self.graph[node2][node1] = new_weight


    def dijkstra(self, start, end):
        """
        Dijkstra算法计算最短路径
        :param start:       起点
        :param end:         终点
        :return:            路径      distances[end]为最短路径距离
        """
        # 初始化所有节点的最短距离为无穷大
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        predecessors = {node: None for node in self.graph}  # 路径前驱节点
        unvisited = list(self.graph.keys())  # 未访问节点列表

        while unvisited:
            # 从未访问节点中选择距离最小的节点
            current_node = None
            current_distance = float('inf')
            for node in unvisited:
                if distances[node] < current_distance:
                    current_node = node
                    current_distance = distances[node]

            # 如果当前节点是目标节点，构造路径并返回
            if current_node == end:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = predecessors[current_node]
                return path[::-1]

            # 如果当前节点无法再前进，结束搜索
            if current_distance == float('inf'):
                break

            # 从未访问列表中移除当前节点
            unvisited.remove(current_node)

            # 更新邻居的距离
            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node

        return None     # 如果没有路径则返回


def create_mesh_graph(mesh, geo):
    """
    创建点权重全为1，边权重为0的图
    :param mesh:        网格
    :param geo:         几何
    :return:
    """
    verts = mesh.verts_packed()
    verts_num = verts.shape[0]
    edges_packed = mesh.edges_packed().tolist()
    edge_num = len(edges_packed)
    if use_networkx:
        geo.mesh_graph = nx.Graph()
        # 添加节点和节点权重
        for i in range(verts_num):
            geo.mesh_graph.add_node(i)
        # # # 添加边
        for i in range(edge_num):
            geo.mesh_graph.add_edge(edges_packed[i][0], edges_packed[i][1])
    else:
        geo.mesh_graph = MyGraph()
        for i in range(verts_num):
            geo.mesh_graph.add_node(i)
        for i in range(edge_num):
            geo.mesh_graph.add_edge(edges_packed[i][0], edges_packed[i][1], 1)


def match_pnt_to_mesh(src_mesh, pnt, had_match):
    """
    以最短距离为标准匹配pnt在网格中对应的网格节点
    :param src_mesh:        网格
    :param pnt:             匹配点 list=[X, Y, Z]
    :param had_match:       已经匹配过的点 tensor
    :return:                pnt在src_mesh中匹配的网格节点编号 int
    """
    mesh_vert = src_mesh.verts_packed().detach()
    optional_mesh_vert = mesh_vert
    verts_num = optional_mesh_vert.shape[0]
    min_dist = float("inf")
    min_indx = -1
    geo_pnt = gp_Pnt(pnt[0], pnt[1], pnt[2])
    for i in range(verts_num):
        if i in had_match:
            continue
        current_pnt = gp_Pnt(optional_mesh_vert[i][0].item(), optional_mesh_vert[i][1].item(), optional_mesh_vert[i][2].item())
        dist = current_pnt.Distance(geo_pnt)
        if dist < min_dist:
            min_dist = dist
            min_indx = i
    return min_indx


def match_geo_vert(mesh, geo):
    """
    针对每个顶点匹配距离最近的网格点，网格点的标号储存在每个顶点类中
    :param mesh:            网格
    :param geo:             几何
    """
    total_match = []
    for key, value in geo.vertexs.items():
        pnt = value.GetPntXYZ()
        matched = match_pnt_to_mesh(mesh, pnt, total_match)
        if matched not in total_match:
            total_match.append(matched)
        else:
            print("matched error")
        geo.vertexs[key].match_mesh_vert = matched


def update_mesh_vert_to_edge_dist(mesh, geo):
    """
    更新从网格点各条边的距离
    :param mesh:    网格
    :param geo:     几何
    """
    verts = mesh.verts_packed()
    verts_num = verts.shape[0]
    for j in range(verts_num):
        projection_pnt = gp_Pnt(verts[j][0].item(), verts[j][1].item(), verts[j][2].item())  # 坐标初始化OCC_pnt
        for key, value in geo.edges.items():
            dist, _, _ = MyGeometry.project_to_curve(projection_pnt, value)
            geo.edges[key].MeshVertToEdge[j] = dist


def match_no_continuity_Edge(mesh, geo):
    """
    根据最短路径去匹配不连续边
    :param mesh:        网格
    :param geo:         几何
    """
    edges_packed = mesh.edges_packed().tolist()
    edge_num = len(edges_packed)
    for key, value in geo.edges.items():
        if key in geo.no_continuity_edge:
            weight = value.MeshVertToEdge
            # 构建关于条边的图来求两个顶点的
            graph = geo.mesh_graph
            for i in range(edge_num):
                mesh_index0 = edges_packed[i][0]
                mesh_index1 = edges_packed[i][1]
                if use_networkx:
                    graph[mesh_index0][mesh_index1]['weight'] = weight[mesh_index0] + weight[mesh_index1]
                else:
                    graph.update_edge_weight(mesh_index0, mesh_index1, weight[mesh_index0] + weight[mesh_index1])
            # 找到该边的两个顶点
            pnt1 = geo.vertexs[value.end_pnt_index[0]]
            pnt2 = geo.vertexs[value.end_pnt_index[1]]
            # 两个顶点所对应的网格点编号
            index1 = pnt1.match_mesh_vert
            index2 = pnt2.match_mesh_vert
            if use_networkx:
                shortest_path = nx.dijkstra_path(graph, source=index1, target=index2)
            else:
                shortest_path = graph.dijkstra(index1, index2)
            geo.edges[key].shortest_path = shortest_path
            value.SampleEdge(len(shortest_path) - 2)


def match_and_proj_final_mesh(mesh, geo):

    verts = mesh.verts_packed()
    verts_num = verts.shape[0]
    final_mesh_local = verts.clone()
    match_geo_vert(mesh, geo)
    final_match = []
    if len(geo.no_continuity_vert) != 0:
        for key in geo.no_continuity_vert:
            value = geo.vertexs[key]
            final_match.append(value.match_mesh_vert)
            final_mesh_local[value.match_mesh_vert] = torch.tensor(value.GetPntXYZ())
    # 边匹配
    update_mesh_vert_to_edge_dist(mesh, geo)
    match_no_continuity_Edge(mesh, geo)
    for key, value in geo.edges.items():
        if key in geo.no_continuity_edge:
            except_end_path = value.shortest_path[1: -1]
            for i in range(len(except_end_path)):
                final_mesh_local[except_end_path[i]] = torch.tensor(value.shortest_path_sample[i])
                final_match.append(except_end_path[i])
    # 面投影
    for j in range(verts_num):
        if j in final_match:
            continue
        current_pnt = gp_Pnt(verts[j][0].item(), verts[j][1].item(), verts[j][2].item())
        proj_min = float("inf")
        for key in geo.faces:
            face = geo.faces[key]
            dist, proj_point = MyGeometry.project_to_face(current_pnt, face, geo)
            if dist < proj_min:
                proj_min = dist
                new_location = torch.tensor([proj_point[0], proj_point[1], proj_point[2]])
        final_mesh_local[j] = new_location
    return final_mesh_local
