import aspose.threed as a3d
import torch
import numpy as np
import Loss
import Draw
import Myio
import MyGeometry
from collections import defaultdict, deque, Counter
from pytorch3d.structures.meshes import Meshes
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.loss import mesh_laplacian_smoothing

stitch_mesh = True
optimize_mesh = True

# 新增1 动态删除网格 2025.4.7 √
# 新增2 新增边界点与原网格点过近 2025.4.7 √
# BUG:1. 跨面网格删除的问题    2025.4.7 √
# BUG:2. 缝合过程补充边界与以后网格贴很近   2025.4.7 √
# BUG:3. 跨面网格跨非连续的面，网格缝合与网格删除均存在问题 2025.4.8 √
# BUG:4. 顶点处连接多条边（大于4）可能出现缝合不准确 2025.4.8 √
# 新增3. 脏几何之不连续边界处理    2025.4.10 √
# 新增4. 网格优化    2025.4.14 √
# BUG:5. 解决网格优化计算时出现nan的bug 2025.4.15 √

def find_neighbors(target, indx):
    """
    找到目标列表中，与indx相邻的元素
    :param target:     list
    :param indx:       [....., prev_num, indx, next_num, ....]
    """
    prev_num = None
    next_num = None
    for i, num in enumerate(target):
        if num == indx:
            prev_num = target[i - 1] if i > 0 else None  # 前一个数字
            next_num = target[i + 1] if i < len(target) - 1 else None # 后一个数字
    return prev_num, next_num


def edge_exists(faces, v1, v2):
    """
    判断网格中是否存在边v1, v2
    :param faces:  网格面
    :param v1:    点1
    :param v2:    点2
    :return:      True or False
    """
    v1, v2 = min(v1, v2), max(v1, v2)  # 统一顺序
    # 逐个检查三角形的边
    edge_mask = (
            ((faces[:, 0] == v1) & (faces[:, 1] == v2)) |
            ((faces[:, 1] == v1) & (faces[:, 2] == v2)) |
            ((faces[:, 2] == v1) & (faces[:, 0] == v2)) |
            ((faces[:, 0] == v2) & (faces[:, 1] == v1)) |
            ((faces[:, 1] == v2) & (faces[:, 2] == v1)) |
            ((faces[:, 2] == v2) & (faces[:, 0] == v1))
    )

    return edge_mask.any().item()

def delete_face(face, i, j, k):
    """
    删除i, j, k所构成的面片
    :return:
    """
    # 找到包含 i, j, k 的行
    mask_i = (face == i).any(dim=1)
    mask_j = (face == j).any(dim=1)
    mask_k = (face == k).any(dim=1)
    # 找到完全包含 i, j, k 的行
    rows_to_delete = mask_i & mask_j & mask_k
    # 保留不包含 i, j, k 的行
    filtered_tensor = face[~rows_to_delete]
    return filtered_tensor

def mesh_edge_swap(faces, fist, last, index, common_elements):
    """
    first ---index                  first ---index
    |        / |                      |   \     |
    |       /  |                      |    \    |
    |      /   |          to          |     \   |
    |     /    |                      |      \  |
    |    /     |                      |       \ |
    common ----last                  common ----last
    :param faces:                    所有网格面
    :param fist:                     点1 编号
    :param last:                     点2 编号
    :param index:                    点3 编号
    :param common_elements:          点4 编号
    :return:                         进行变交换后的网格面
    """

    faces = delete_face(faces, fist, index, common_elements)
    faces = delete_face(faces, last, index, common_elements)
    faces = torch.cat((faces, torch.tensor([[fist, last, common_elements], [fist, index, last]])), dim=0)
    return faces

def add_mesh_face_with_4_pnts(My_mesh, f1, f2, e2, e1):
    """
    将四个点构成的面片添加到网格面中
    :param My_mesh:   网格
    :param f1:         e1------e2
    :param f2:        /       /
    :param e1:       /       /
    :param e2:      f1------f2
    :return:
    """
    mesh_verts = torch.tensor(My_mesh.verts)
    dist1 = torch.norm(mesh_verts[f1] - mesh_verts[e2])
    dist2 = torch.norm(mesh_verts[f2] - mesh_verts[e1])
    if dist1 > dist2:
        My_mesh.add_face(f1, f2, e1)
        My_mesh.add_face(e1, e2, f2)
    else:
        My_mesh.add_face(f1, e1, e2)
        My_mesh.add_face(f1, f2, e2)

def is_triangle_in_faces(faces, i, j, k):
    """
    判断点i, j, k是否在faces中构成面片
    :return:  True or False
    """
    # 将查询的三个点按升序排序，
    query = torch.tensor([i, j, k]).sort().values  # [i, j, k] → 排序后
    # 对 faces 的每一行进行排序
    sorted_faces = torch.sort(faces, dim=1).values  # 对每个面按行排序
    # 判断查询的面是否存在
    return (sorted_faces == query).all(dim=1).any()


def compute_face_adjacency(faces):
    """
    计算网格面片的邻接关系
    :param faces:   (F, 3) 的张量，每行是一个三角形的顶点索引
    :return:        邻接表 {face_id: [adjacent_face_ids]}
    """
    adjacency = defaultdict(set)
    edges_to_faces = defaultdict(set)  # 每条边对应的面
    # 遍历所有三角形，记录边到面片的映射
    for i, face in enumerate(faces):
        # 对每一条边进行排序，确保无向边是唯一的
        edges = [
            tuple(sorted((face[0].item(), face[1].item()))),
            tuple(sorted((face[1].item(), face[2].item()))),
            tuple(sorted((face[2].item(), face[0].item()))),
        ]
        # 记录每条边和其对应的面片索引
        for edge in edges:
            edges_to_faces[edge].add(i)

    # 通过共享边更新邻接关系
    for edge, face_list in edges_to_faces.items():
        if len(face_list) == 2:  # 如果某条边只被两个面共享，更新它们的邻接关系
            face_1, face_2 = face_list
            adjacency[face_1].add(face_2)
            adjacency[face_2].add(face_1)

    return {k: list(v) for k, v in adjacency.items()}



def closest_point_on_segment(p, p1, p2):
    d = p2 - p1
    t = torch.dot(p - p1, d) / torch.dot(d, d)
    t = torch.clamp(t, 0, 1)
    return p1 + t * d

def project_point_to_triangle(p, tri):
    """
    计算点 p 在三角形 tri 上的最近投影点
    p: (3,)  - 目标点
    tri: (3, 3) - 三角形三个顶点
    返回: 最近投影点 (3,)
    """
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    # 计算边向量的外积（法向量）
    d1, d2 = torch.dot(ap, ab), torch.dot(ap, ac)
    d3, d4 = torch.dot(ab, ab), torch.dot(ab, ac)
    d5, d6 = torch.dot(ac, ab), torch.dot(ac, ac)

    # 计算重心坐标（u, v）
    denom = d4 * d6 - d5 * d5
    v = (d6 * d2 - d5 * d1) / denom
    w = (d4 * d1 - d5 * d2) / denom
    u = 1 - v - w

    # 判断点是否在三角形内
    if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
        return u * a + v * b + w * c
    # 点在三角形外部，返回三角形的最近边上的点
    edge_pts = torch.stack([
        closest_point_on_segment(p, a, b),
        closest_point_on_segment(p, b, c),
        closest_point_on_segment(p, c, a),
    ])

    vertex_pts = torch.stack([a, b, c])
    candidates = torch.cat([edge_pts, vertex_pts], dim=0)
    dists = torch.norm(candidates - p, dim=1)
    return candidates[dists.argmin()]


def project_point_to_tris(p, tris):
    """
    计算点 p 在所有三角形 tris 上的最近投影点
    :return:   最近投影点所在三角面片在tris中的索引
    """
    # 计算 p 到所有三角形的最近投影点
    projections = torch.stack([project_point_to_triangle(p, tri) for tri in tris])
    # 找到最近的投影点
    dists = torch.norm(projections - p, dim=1)
    return dists.argmin()

def find_tri_in_face(geo, faces, face_index, tri_index):

    mesh_pnt = []
    p1, p2, p3 = faces[tri_index]
    for p in (p1, p2, p3):
        if geo.mesh_belong_face[p.item()] == face_index:
            mesh_pnt.append(p.item())
    return mesh_pnt


def sort_faces_by_adjacency(faces, adj_faces):
    """
    按相邻关系对面进行排序
    :param faces:            无序面的index
    :param adj_faces:        面的邻接关系
    :return:                 排序后的面
    """
    if not faces:
        return []

    start_face = faces[0]  # 选取无序面列表中的第一个作为起始面
    visited = set()
    sorted_faces = []
    queue = deque([start_face])

    while queue:
        face = queue.popleft()
        if face in visited:
            continue
        visited.add(face)
        sorted_faces.append(face)

        # 将相邻的面加入队列（但确保在原始 faces 列表中）
        for neighbor in adj_faces.get(face, []):
            if neighbor in faces and neighbor not in visited:
                queue.append(neighbor)

    return sorted_faces


def group_faces_subset_by_adjacency(faces, adjacency):
    """
    将一组面按照邻接关系分组
    :param faces:
    :param adjacency:
    :return:
    """
    faces_set = set(faces)
    visited = set()
    groups = []

    for face in faces:
        if face not in visited:
            group = []
            stack = [face]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                group.append(current)
                # 只考虑在给定面组内的邻居
                neighbors = adjacency.get(current, [])
                for nbr in neighbors:
                    if nbr in faces_set and nbr not in visited:
                        stack.append(nbr)
            groups.append(group)

    return groups



def find_common_edges(sorted_faces, face_vertices):
    """
     找到排序面列表中相邻面的公共边
    :param sorted_faces:         排序的面列表
    :param face_vertices:       面的顶点列表
    :return:
    """
    common_edges = []

    for i in range(len(sorted_faces) - 1):
        face1 = sorted_faces[i]
        face2 = sorted_faces[i + 1]

        # 获取两个面的顶点集
        v1, v2, v3 = face_vertices[face1]
        v4, v5, v6 = face_vertices[face2]

        vertices1 = {v1.item(), v2.item(), v3.item()}
        vertices2 = {v4.item(), v5.item(), v6.item()}
        # 找到两个面的公共边（两个共同的顶点）
        common_vertices = vertices1 & vertices2
        if len(common_vertices) == 2:  # 共享 2 个顶点即为公共边
            edge = list(common_vertices)
            common_edges.append(edge)
        else:
            print("Error: Faces do not share an edge.")

    return common_edges

def find_cross_curve_mesh_face(curve_mesh_face, all_mesh_face, geo_faces):
    """
    从curve_mesh_face中找到横跨几何线的网格面
    :param curve_mesh_face:         待选择的网格面id
    :param all_mesh_face:           网格的所有网格面数组
    :param geo_faces:               当前线所横跨的两个几何面
    :return:
    """
    find_face = -1
    for mesh_face in curve_mesh_face:
        a, b, c = all_mesh_face[mesh_face]
        geo_faces_list = []
        for p in (a, b, c):
            geo_faces_list.append(geo.mesh_belong_face[p.item()])
        if set(geo_faces).issubset(set(geo_faces_list)):
            find_face = mesh_face
    return find_face

def sort_quad_points(faces, points):
    """
    使用四个点之间的距离来找到最长的对角线，从而排序四个点。
    :param faces:   网格面
    :param points:  排序点
    :return:
    """
    faces = torch.tensor(faces)
    # 没有相连的边为对角线
    for i in range(4):
        for j in range(i + 1, 4):
            if not edge_exists(faces, points[i], points[j]):
                p1, p2 = points[i],  points[j]

    remaining_points = [point for point in points if point != p1 and point != p2]

    return [p1, remaining_points[0], p2, remaining_points[1]]


class DynamicMesh:
    def __init__(self, verts_list, faces_list):
        self.verts = verts_list  # List of [x, y, z]
        self.faces = faces_list  # List of [i, j, k]
        self.deleted_vert_indices = set()
        self.deleted_face_indices = set()

    def modify_vert_xyz(self, idx, new_xyz):
        """修改顶点坐标"""
        self.verts[idx] = new_xyz

    def add_vertex(self, v):
        """增加网格顶点，返回id"""
        self.verts.append(v)
        return len(self.verts) - 1

    def add_face(self, i, j, k):
        """增加网格面，返回id"""
        if i != j and i != k and j != k:
            self.faces.append([i, j, k])

    def delete_vertex(self, idx):
        """增加网格点标记，后续删除"""
        self.deleted_vert_indices.add(idx)

    def delete_face(self, idx):
        """增加网格面标记，后续删除"""
        self.deleted_face_indices.add(idx)

    def delete_list_face(self, index_list):
        self.deleted_face_indices.update(index_list)

    def compact(self):
        """
        重建网格:
        - 移除标记中的网格点与网格面.
        - 重建网格面中的id以匹配新的网格点数组.
        """
        index_map = {}  # old_index -> new_index
        new_verts = []

        # Rebuild vertex list
        for old_idx, v in enumerate(self.verts):
            if old_idx not in self.deleted_vert_indices:
                new_idx = len(new_verts)
                index_map[old_idx] = new_idx
                new_verts.append(v)

        # Rebuild face list with updated indices
        new_faces = []
        for face_idx, face in enumerate(self.faces):
            if face_idx in self.deleted_face_indices:
                continue
            # Check if all vertices of the face are valid
            if all(v_idx in index_map for v_idx in face):
                new_faces.append([index_map[v] for v in face])

        # Replace old data
        self.verts = new_verts
        self.faces = new_faces
        self.deleted_vert_indices.clear()
        self.deleted_face_indices.clear()

    def to_meshes(self):
        """Export to PyTorch3D Meshes object (single mesh)."""
        #
        for face in self.faces:
            test_face = set(face)
            if len(test_face) != 3:
                print("存在面不构成三角形", face)
        verts_tensor = torch.tensor(self.verts, dtype=torch.float32)
        faces_tensor = torch.tensor(self.faces, dtype=torch.int64)
        return verts_tensor, faces_tensor




verts, faces = Myio.read_obj("final_model_test.obj")
verts = torch.tensor(verts, dtype=torch.float32)
faces = torch.tensor(faces, dtype=torch.int64)

file_name = "diamond.step"
# file_name = "visor.step"
# file_name = "new_part1_new.step"
# file_name = "new_part15.step"
mesh_vert_num =verts.shape[0]
geo = MyGeometry.OccGeo(file_name, mesh_vert_num)



global_scale = 1
final_scale = geo.scale / global_scale
final_translation = torch.tensor(geo.center) * global_scale

verts = (verts - final_translation) * final_scale
new_src_mesh = Myio.Meshes(verts=[verts], faces=[faces])
MyGeometry.create_mesh_graph(new_src_mesh, geo)

tris = verts[faces]

# Draw.plot_two_point(geo.plot_sample, new_src_mesh.verts_packed().numpy())
# abc 中需要缩放，但计算时候忘记处理了，这里进行缩放
# tmp_scale = 0.001
# tmp_scale = geo.scale / tmp_scale
# tmp_translation = torch.tensor(geo.center) * global_scale
#
# tmp_verts = (verts - tmp_translation) * final_scale
# final_obj = 'final_model.obj'
# Myio.save_obj(final_obj, tmp_verts, faces)
# scene = a3d.Scene.from_file("final_model.obj")
# scene.save("final_model.stl")

# 计算mapping后的点
# mapping_pnts = torch.Tensor(MyGeometry.mapping_pnt_to_geo(new_src_mesh, geo, 2))
# 进行边交换

# if len(geo.mesh_duplicates) == 0:
#     # 确定存在问题的边的位置
#     for index, keys in geo.mesh_duplicates.items():
#         # 先处理一个点在2条线上的情况
#         if len(keys) == 2:
#             # 匹配点按照key先后顺序，key数字越大则重复的点会在该曲线上，另一条线则会缺失该点
#             path_problem = geo.edges[keys[0]].shortest_path
#             path_normal = geo.edges[keys[1]].shortest_path
#             # 缺失点在问题曲线上的前后点编号
#             fix_first, fix_last = find_neighbors(path_problem, index)
#             # 缺失点在正常曲线上的前后点编号，用于定位存在问题的面片
#             normal_first, normal_last = find_neighbors(path_normal, index)
#             # 找到的连接情况，start到fix_end之间的形状无法保持，需要进行边交换
#             #                  fix_end
#             #                    |
#             #       start------index--------normal_end
#             start = ({fix_first, fix_last} & {normal_first, normal_last}).pop()
#             fix_end = fix_first if fix_first != start else fix_last
#             normal_end = normal_first if normal_first != start else normal_last
#             # 确认是fix_end与start之间缺失没有边相连
#             if not edge_exists(new_src_mesh, start, fix_end):
#                 print("尝试进行边交换", start, fix_end, index)
#                 # 找到包含fist和index, index和last的总共的四个面
#                 mask_v1 = (faces == start)
#                 mask_v2 = (faces == fix_end)
#                 mask_v3 = (faces == index)
#                 row_mask1 = torch.logical_and(mask_v1.any(dim=1), mask_v3.any(dim=1))
#                 row_mask2 = torch.logical_and(mask_v2.any(dim=1), mask_v3.any(dim=1))
#                 find_face1 = faces[torch.where(row_mask1)]
#                 find_face2 = faces[torch.where(row_mask2)]
#                 # 从四个面中找到除去start, fix_end, index的顶点
#                 pnts1 = find_face1[(find_face1 != start) & (find_face1 != index)]
#                 pnts2 = find_face2[(find_face2 != fix_end) & (find_face2 != index)]
#                 mask = torch.isin(pnts1, pnts2)
#                 common_elements = pnts1[mask]
#                 # 目前看到需要进行边交换有两种情况
#                 # 如果能找到common_elements，则在四边形内部边交换即可
#                 #       common_elements----fix_end
#                 #         |             \    |
#                 #       start-------------index------normal_end
#                 if len(common_elements) != 0:
#                     # 两个三角形顶点分别为fist、index、common_elements 和 last、index、common_elements
#                     faces = mesh_edge_swap(faces, start, fix_end, index, common_elements)
#                 # 第二种情况需要进行两次边交换
#                 #         v2------v1----fix_end
#                 #         |   \    |    /
#                 #       start----index------normal_end
#                 else:
#                     if normal_end in pnts2:
#                         v1 = pnts2[pnts2 != normal_end].item()
#                         if is_triangle_in_faces(faces, pnts1[0].item(), index, v1):
#                             v2 = pnts1[0].item()
#                         elif is_triangle_in_faces(faces, pnts1[1].item(), index, v1):
#                             v2 = pnts1[1].item()
#                         else:
#                             print("其他情况，无法进行交换")
#                             continue
#                         # 进行两次交换即可使得 start fix_end 之间有边相连
#                         faces = mesh_edge_swap(faces, start, v1, index, v2)
#                         faces = mesh_edge_swap(faces, start, fix_end, index, v1)
#                     else:
#                         print("无法判断方向，normal_end顶点不在 index--fix_end构建的面片中")

# path = geo.faces[4].mapping_inside_mesh_index
proj_pnts = torch.Tensor(MyGeometry.project_mesh_to_geo(new_src_mesh, geo))

new_src_mesh = Myio.Meshes(verts=[proj_pnts], faces=[faces])
new_mesh_pnt = new_src_mesh.verts_packed()
new_mesh_face = new_src_mesh.faces_packed()
# geo.update_mesh_vert_to_face(new_src_mesh)


# 记录网格中固定的边界点与边界网格边
fixed_boundary_mesh_pnt = set()
fixed_boundary_mesh_edge = set()

# 缝合几何特征
if stitch_mesh:
    # 新建动态网格数组
    My_mesh = DynamicMesh(new_mesh_pnt.tolist(), new_mesh_face.tolist())
    # 计算邻接关系
    adjacency= compute_face_adjacency(new_mesh_face)
    # 只关联一个面的边为边界边
    boundary_geo_curve = set()
    # 边界边的顶点
    boundary_geo_pnt = set()
    # 边界边所在的几何面
    geo_face_with_boundary_geo_curve = set()
    # 顶点处的网格点
    geo_vert_to_mesh_pnt = defaultdict(set)
    # 找到边界边
    for key, value in geo.edges.items():
        # 边界边不认为是几何特征
        if len(geo.edge_to_surface[key]) < 2:
            boundary_geo_curve.add(key)
            geo_face_with_boundary_geo_curve.update(value.belong_face_index)
            continue
    # 找到边界边的顶点
    for i in boundary_geo_curve:
        boundary_geo_pnt.update(geo.edges[i].end_pnt_index)
    # 遍历所有几何边，找到不连续的边，增加顶点，但不管边界边顶点
    for key, value in geo.edges.items():
        # 不连续需要对该边进行缝合，先增加端点处增加对应的网格点：
        if not value.is_continuity:
            # 添加端点
            for end in value.end_pnt_index:
                if end in boundary_geo_pnt:
                    continue
                # 保证不重复添加
                if end not in geo_vert_to_mesh_pnt:
                    new_pnt_xyz = geo.vertexs[end].GetPntXYZ()
                    new_mesh_pnt_id = My_mesh.add_vertex(new_pnt_xyz)
                    geo_vert_to_mesh_pnt[end] = new_mesh_pnt_id

    # 跨几何边的网格面
    geo_edge_to_mesh_faces = defaultdict(set)
    # 顶点处的网格面
    geo_vert_to_mesh_faces = defaultdict(set)
    # 横跨了不邻接几何面的网格面
    mesh_face_cross_unconnected_geo_face = []
    # 受到边界点牵连的不处理的几何线
    geo_edge_involve_boundary_pnt = set()
    # 遍历网格面，找出跨多个几何面的网格面索引
    for i, current_mesh_face in enumerate(new_mesh_face):
            p1, p2, p3 = current_mesh_face  # 获取该面上的三个点
            # 逐个检查每个点的几何面归属
            mesh_face_list = []
            # 逐步添加该点所属的几何面
            for p in (p1, p2, p3):
                mesh_face_list.append(geo.mesh_belong_face[p.item()])
            # 字典去重
            mesh_face_set = set(mesh_face_list)
            # 如果该网格面跨多个几何面，则记录其索引
            if len(mesh_face_set) == 2:
                # 先删除该网格面
                face_pair = tuple(sorted(mesh_face_set))
                # 利用网格面横跨的几何面与 几何边相接的几何面进行对比，确定网格面所属几何面
                found_edge = False
                for edge, surfaces in geo.edge_to_surface.items():
                    if set(face_pair) == set(surfaces):
                        geo_edge_to_mesh_faces[edge].add(i)
                        found_edge = True
                # 如果该网格面没有找到对应的几何边，则说明该网格面的几何面不连续
                if not found_edge:
                    # 检查不连续的面是否为边界面
                    if not set(face_pair).issubset(geo_face_with_boundary_geo_curve):
                        mesh_face_cross_unconnected_geo_face.append(i)
            elif len(mesh_face_set) == 3:
                # 如果该网格面跨三个几何面，为顶点位置
                edges_end = []
                # 将三个面中所有顶点全集中起来， 次数最多的顶点为该跨三个几何面的网格对应的顶点
                for face_key in mesh_face_list:
                    edges_key = geo.faces[face_key].contain_edge_index
                    for edge_key in edges_key:
                        edges_end = edges_end + geo.edges[edge_key].end_pnt_index
                # 找几何边的公共顶点
                count = Counter(edges_end)
                max_freq = max(count.values())
                most_common_elements = [k for k, v in count.items() if v == max_freq]
                # 如果该顶点为边界点，则不处理
                found = []
                for common in most_common_elements:
                    if common not in boundary_geo_pnt:
                        found.append(common)
                if len(found) == 1:
                    geo_vert_to_mesh_faces[found[0]].add(i)
                elif len(found) > 1:
                    geo_vert_to_mesh_faces[found[0]].add(i)
                    print("网格面所计算出对应的几何顶点超过一个，检查问题")

    # 检查
    if len(geo_vert_to_mesh_faces) != len(geo_vert_to_mesh_pnt):
        print("顶点处网格点与顶点处网格面数量不匹配")
    # 处理网格面跨三个面并非位于顶点处的情况
    alternative = set(mesh_face_cross_unconnected_geo_face)
    for key, value in geo_vert_to_mesh_faces.items():
        # 将初步找到的网格面按邻接关系进行分组
        group = group_faces_subset_by_adjacency(list(value), adjacency)
        if len(group) == 1:
            continue
        # 认为长度最长的为该顶点的网格面
        longest_sublist = max(group, key=len)
        other_sublist = [i for i in group if i is not longest_sublist]
        # other_sublist 中的多个网格面放入longest_sublist中，
        # 但是要保证整体为联通区域，目前的方式为将跨不连接面的网格面补充进去，即可组成联通区域
        changed = True
        extended = set(longest_sublist)

        while changed:
            changed = False
            to_add = set()

            for a in extended:
                neighbors = adjacency.get(a, [])
                for neighbor in neighbors:
                    # 检查邻居是否是跨不连结几何面的网格面
                    if neighbor in alternative:
                        to_add.add(neighbor)
                        # 如果是，看是否该面连接other_sublist
                        other_neighbors = adjacency.get(neighbor, [])
                        for other_neighbor in other_neighbors:
                            for other in other_sublist:
                                if other_neighbor in other:
                                    other_sublist = [i for i in other_sublist if i is not other]
                                    other_neighbor = set(other)
                                    to_add.update(other_neighbor)
            if to_add:
                extended.update(to_add)
                alternative.difference_update(to_add)
                changed = True
                if len(other_sublist) == 0:
                    changed = False
        if len(other_sublist) != 0:
            print("仍然跨三个几何面的网格顶点为组成联通区域", other_sublist)
            print(extended)
        geo_vert_to_mesh_faces[key] = extended
    if len(alternative) != 0:
            print("仍然存在跨不连续几何面的网格面未被匹配, 可能存在于不相接区域", alternative)
    # 删除几何顶点处对应的网格面
    for key, value in geo_vert_to_mesh_faces.items():
        if key not in boundary_geo_pnt:
            My_mesh.delete_list_face(value)

    # 统计几何顶点处的网格顶点在不同几何面中所连接的网格点
    geo_vert_to_mesh_pnt_in_geo_face = defaultdict(lambda: defaultdict(set))
    # 在不连续的边处增加网格顶点，并进行缝合
    for key, value in geo.edges.items():
        # 验证是否连续
        if value.is_continuity:
            continue
        # 边的端点
        curve_ends = value.end_pnt_index
        # 如果该边连结边界顶点，不进行缝合
        if len(set(curve_ends) & boundary_geo_pnt) != 0:
            geo_edge_involve_boundary_pnt.add(key)
            continue
        # 验证该边的属性
        geo_faces = geo.edge_to_surface[key]
        if len(geo_faces) < 2:
            print(key, "为边界边或者孤边，不处理")
            continue
        if len(geo_faces) > 2:
            print(key, "非流形边，暂不处理")
            continue
        # 找到该边处所有跨面的网格
        curve_first_mesh_face = list(geo_vert_to_mesh_faces[curve_ends[0]])
        # 顶点处可能存在多个备选的网格面,选择横跨这条边的面
        if len(curve_first_mesh_face) != 1:
            curve_first_mesh_face = [find_cross_curve_mesh_face(curve_first_mesh_face, new_mesh_face, geo_faces)]
        curve_path = list(geo_edge_to_mesh_faces[key])
        curve_last_mesh_face = list(geo_vert_to_mesh_faces[curve_ends[1]])
        if len(curve_last_mesh_face) != 1:
            curve_last_mesh_face = [find_cross_curve_mesh_face(curve_last_mesh_face, new_mesh_face, geo_faces)]
        # 将所有网格面进行排序
        curve_all_mesh_faces = curve_first_mesh_face + curve_path + curve_last_mesh_face
        sorted_faces = sort_faces_by_adjacency(curve_all_mesh_faces, adjacency)
        # 根据排序后的三角形找到公共边
        common_edge = find_common_edges(sorted_faces, new_mesh_face)
        # 删除跨不连续的边的网格面
        My_mesh.delete_list_face(sorted_faces)
        # 得到分布在两个面中的网格点
        edge_related_mesh_in_each_face = []
        for face_indx in geo_faces:
            current_pnt = []
            for current_mesh_face in sorted_faces:
                tmp = find_tri_in_face(geo, new_mesh_face, face_indx, current_mesh_face)
                for i in tmp:
                    if i not in current_pnt:
                        current_pnt.append(i)
            edge_related_mesh_in_each_face.append(current_pnt)

        # 在公共边上增加网格点
        current_edge_new_pnt = []
        current_edge_new_pnt = current_edge_new_pnt + [geo_vert_to_mesh_pnt[curve_ends[0]]]

        for common in common_edge[1::2]:
            p1 = My_mesh.verts[common[0]]
            p2 = My_mesh.verts[common[1]]
            new_pnt_xyz, t = value.closest_point_on_curve_from_mesh_edge(p1, p2)
            # 判断点是否与已有网格点过近
            if t < 0.25:
                My_mesh.modify_vert_xyz(common[0], new_pnt_xyz)
                current_edge_new_pnt.append(common[0])
            elif t > 0.75:
                My_mesh.modify_vert_xyz(common[1], new_pnt_xyz)
                current_edge_new_pnt.append(common[1])
            else:
                new_mesh_pnt_id = My_mesh.add_vertex(new_pnt_xyz)
                current_edge_new_pnt.append(new_mesh_pnt_id)
        current_edge_new_pnt = current_edge_new_pnt + [geo_vert_to_mesh_pnt[curve_ends[1]]]
        # 记录边界点
        fixed_boundary_mesh_pnt.update(current_edge_new_pnt)
        # 将两个面中的点进行缝合
        for i in range(2):
            current_face_related_mesh = edge_related_mesh_in_each_face[i]
            # 起点连接的面网格点
            geo_vert_to_mesh_pnt_in_geo_face[curve_ends[0]][geo_faces[i]].add(current_face_related_mesh[0])
            geo_vert_to_mesh_pnt_in_geo_face[curve_ends[1]][geo_faces[i]].add(current_face_related_mesh[-1])
            # 情况1   面上的点与边上的点一样
            if len(current_face_related_mesh) == len(current_edge_new_pnt):
                for j in range(len(current_face_related_mesh) - 1):
                    face_pnt1 = current_face_related_mesh[j]
                    face_pnt2 = current_face_related_mesh[j + 1]
                    edge_pnt1 = current_edge_new_pnt[j]
                    edge_pnt2 = current_edge_new_pnt[j + 1]
                    # 增加网格面
                    add_mesh_face_with_4_pnts(My_mesh, face_pnt1, face_pnt2, edge_pnt2, edge_pnt1)
                    # 记录边界
                    fixed_boundary_mesh_edge.add((edge_pnt1, edge_pnt2))
            # 情况2  面上的点比边上的少1，或者2
            elif len(current_face_related_mesh) < len(current_edge_new_pnt):
                # # 第一个三角面面片
                My_mesh.add_face(current_face_related_mesh[0], current_edge_new_pnt[0], current_edge_new_pnt[1])
                # 中间的按四个点处理，找出短边作为网格边
                for j in range(len(current_face_related_mesh) - 1):
                    face_pnt1 = current_face_related_mesh[j]
                    face_pnt2 = current_face_related_mesh[j + 1]
                    edge_pnt1 = current_edge_new_pnt[j + 1]
                    edge_pnt2 = current_edge_new_pnt[j + 2]
                    # 增加网格面
                    add_mesh_face_with_4_pnts(My_mesh, face_pnt1, face_pnt2, edge_pnt2, edge_pnt1)

                # 面上的点比边上的点少2个，最后还有一个三角片需要构造
                if len(current_edge_new_pnt) - len(current_face_related_mesh) == 2:
                    My_mesh.add_face(current_face_related_mesh[-1], current_edge_new_pnt[-1], current_edge_new_pnt[-2])
    # 处理顶点在面上存在多个对应点的情况
    for geo_pnt_key, value1 in geo_vert_to_mesh_pnt_in_geo_face.items():
        if geo_pnt_key in boundary_geo_pnt:
            continue
        for geo_face_key, value2 in value1.items():
            # 该处存在网格缺失，补足
            value = list(value2)
            if len(value2) == 2:
                My_mesh.add_face(value[0], value[1], geo_vert_to_mesh_pnt[geo_pnt_key])
            elif len(value2) > 2:
                print("顶点在面上存在多个对应点的情况，且顶点数量大于2，需要检查问题")

    # 处理连续的边，其在顶点位置连接其他不连续边，存在的网格缺失情况
    for key, value in geo.edges.items():
        # 验证是否连续
        if value.is_continuity or key in geo_edge_involve_boundary_pnt:
            # 边的端点
            curve_ends = value.end_pnt_index
            # 相关面
            geo_faces = geo.edge_to_surface[key]
            # 处理端点
            for end in curve_ends:
                if end in boundary_geo_pnt:
                    continue
                # 验证该边端点是否对应了网格顶点
                if end in geo_vert_to_mesh_pnt:
                    # 第一个点为网格点
                    new_tris = {geo_vert_to_mesh_pnt[end]}
                    # 找到该点处所有跨面的网格
                    current_mesh_faces = list(geo_vert_to_mesh_faces[end])
                    for face_indx in geo_faces:
                        for mesh_face_indx in current_mesh_faces:
                            tmp = find_tri_in_face(geo, new_mesh_face, face_indx, mesh_face_indx)
                            if len(tmp) != 0 and tmp[0]not in new_tris:
                                for v in tmp:
                                    new_tris.add(v)
                    new_tris = list(new_tris)
                    if len(new_tris) == 3:
                        My_mesh.add_face(new_tris[0], new_tris[1], new_tris[2])
                    elif len(new_tris) == 4:
                        sort_tris = sort_quad_points(My_mesh.faces, new_tris)
                        add_mesh_face_with_4_pnts(My_mesh, sort_tris[0], sort_tris[1], sort_tris[2], sort_tris[3])
                    else:
                        print("连续的边端点补充顶点大于4，需要检查问题")

    My_mesh.compact()
    new_mesh_pnt, new_mesh_face = My_mesh.to_meshes()


# 网格优化
if optimize_mesh:
    optimize_device = torch.device("cpu")
    new_mesh = Meshes(verts=[new_mesh_pnt], faces=[new_mesh_face])
    # 优化权重
    w_match_edge = 1
    w_angle = 1
    w_edge_length = 1  # 0.01
    w_normal = 0.1
    w_laplacian = 0.1
    steps = 200
    # 损失变化
    angle_losses = []
    edge_length_losses = []
    normal_losses = []
    laplacian_losses = []
    # 生成 fixed_mask
    fixed_mask = torch.zeros(len(new_mesh_pnt), dtype=torch.bool)
    fixed_mask[list(fixed_boundary_mesh_pnt)] = True
    optimize_mask = ~fixed_mask
    #
    optimize_mesh_pnt_num = len(new_mesh_pnt) - len(fixed_boundary_mesh_pnt)
    # 优化的偏移量
    deform_verts = torch.full((optimize_mesh_pnt_num, 3), 0.0, device=optimize_device, requires_grad=True)

    # 生成 optimizer
    optimizer = torch.optim.Adam([deform_verts], lr=0.01)
    # 开始优化
    for step in range(steps):
        optimizer.zero_grad()
        all_mesh_deform = torch.full(new_mesh_pnt.shape, 0.0, device=deform_verts.device)
        #
        all_mesh_deform[optimize_mask] = deform_verts
        #
        optimize_mesh = new_mesh.offset_verts(all_mesh_deform)
        #
        # 计算损失函数
        loss_edge_length = mesh_edge_loss(optimize_mesh)
        loss_laplacian = mesh_laplacian_smoothing(optimize_mesh, "cot")
        loss_normal = mesh_normal_consistency(optimize_mesh)
        loss_angle = Loss.mesh_angle_loss(optimize_mesh, torch.pi / 3)

        total_loss = loss_angle * w_angle + loss_normal * w_normal + loss_laplacian * w_laplacian

        angle_losses.append(float(loss_edge_length.detach().cpu()))
        angle_losses.append(float(loss_angle.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        print(step, loss_edge_length, loss_angle, loss_normal, loss_laplacian)
        total_loss.backward()
        optimizer.step()

    Draw.plot_final_loss(angle_losses, edge_length_losses, normal_losses,laplacian_losses)

    new_mesh_pnt, new_mesh_face = optimize_mesh.get_mesh_verts_faces(0)

    # # 边界点用路径点
# path = []
# for key, value in geo.edges.items():
#     if not value.is_continuity:
#         path += [index for index in value.shortest_path if index not in path]
#

# mapping_pnts[path] = proj_pnts[path]
# Draw.plot_point(geo.plot_sample)
# Draw.plot_point(proj_pnts.numpy())
# deform = proj_pnts - new_src_mesh.verts_packed()
# new_src_mesh = new_src_mesh.offset_verts(deform)
# final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = new_mesh_pnt / final_scale + final_translation

final_obj = 'final_model_mapping.obj'
Myio.save_obj(final_obj, final_verts, new_mesh_face)
scene = a3d.Scene.from_file("final_model_mapping.obj")
scene.save("final_model_mapping.stl")