
import torch
import torch.nn as nn
from MyGeometry import project_mesh_vert_to_geo
import Match
from pytorch3d.ops import sample_points_from_meshes


def chamfer_distance(points1, points2):
    """
        计算两个点云之间的 Chamfer 距离。
    :param points1:     第一个点云Tensor，形状为 (N, 3)
    :param points2:     第二个点云Tensor，形状为 (N, 3)
    :return:        Chamfer 距离的标量值Tensor
    """
    # 计算所有点对之间的欧氏距离平方
    diff = points1.unsqueeze(1) - points2.unsqueeze(0)  # (N, M, 3)
    dist_matrix = torch.sum(diff ** 2, dim=2)  # (N, M)

    # 对于 points1 中的每个点，找到 points2 中最近的点并取均值
    min_dist1, _ = torch.min(dist_matrix, dim=1)  # (N,)
    loss1 = torch.mean(min_dist1)  # 均值

    # 对于 points2 中的每个点，找到 points1 中最近的点并取均值
    min_dist2, _ = torch.min(dist_matrix, dim=0)  # (M,)
    loss2 = torch.mean(min_dist2)  # 均值

    # 返回 Chamfer 距离（取均值确保对称性）
    chamfer_dist = (loss1 + loss2)
    return chamfer_dist

def geo_chamfer_loss(trg_mesh, src_geo):
    """
    计算点云损失
    :param trg_mesh:        网格
    :param src_geo:         几何
    :return:                损失
    """
    # 几何离散采样
    total_sample_num = src_geo.num_vertex
    sample_geo_tensor = torch.tensor(src_geo.get_all_pnt_xyz(), dtype=torch.float32)
    num1, sample_edge = src_geo.sample_uniform_in_crv(1)
    num2, sample_surface = src_geo.sample_in_surface(4)
    total_sample_num = total_sample_num + num1 + num2
    sample_geo_tensor = torch.cat((sample_geo_tensor, torch.tensor(sample_edge, dtype=torch.float32)), 0)
    sample_geo_tensor = torch.cat((sample_geo_tensor, torch.tensor(sample_surface, dtype=torch.float32)), 0)
    # 网格采样
    sample_mesh = sample_points_from_meshes(trg_mesh, total_sample_num)
    loss_chamfer = chamfer_distance(sample_mesh.squeeze(0), sample_geo_tensor)
    # 计算损失
    return loss_chamfer



def geo_match_loss(trg_mesh, src_geo):
    """
    计算几何边的匹配损失，用于处理几何边界
    :param trg_mesh:    网格
    :param src_geo:     几何
    :return:            计算损失
    """
    mesh_verts = trg_mesh.verts_packed()
    had_matched = torch.tensor([], dtype=torch.int)
    # 匹配几何点
    Match.match_geo_vert(trg_mesh, src_geo)
    matched_pnt = []
    # 提取出几何点匹配数据
    for key, value in src_geo.vertexs.items():
        had_matched = torch.cat([had_matched, torch.tensor([value.match_mesh_vert], dtype=torch.int)])
        matched_pnt.append(value.GetPntXYZ())
    # 更新网格点到各个边的距离，作为图的权重
    Match.update_mesh_vert_to_edge_dist(trg_mesh, src_geo)
    # 利用最短路径去匹配边
    Match.match_no_continuity_Edge(trg_mesh, src_geo)
    # 获取边匹配数据
    for key, value in src_geo.edges.items():
        if key in src_geo.no_continuity_edge:
            had_matched = torch.cat([had_matched, torch.tensor(value.shortest_path, dtype=torch.int)])
            for xyz in value.shortest_path_sample:
                matched_pnt.append(xyz)
    # 计算损失
    matched_pnt_tensor = torch.tensor(matched_pnt)
    match_verts_tensor = mesh_verts[had_matched]
    mseloss = nn.MSELoss()
    mean_loss = mseloss(match_verts_tensor, matched_pnt_tensor)
    return mean_loss

def geo_proj_surface_loss(trg_mesh, src_geo):
    """
    以投影为计算方法，得到网格点到面的距离损失计算
    :param trg_mesh:    网格实例
    :param src_geo:     几何
    :return:            损失（平均距离）
    """
    verts = trg_mesh.verts_packed()
    # 找到网格顶点对应投影点位置
    tensor_pnts = torch.Tensor(project_mesh_vert_to_geo(verts, src_geo))
    # 网格点与投影点计算损失
    loss = nn.MSELoss()
    mean_loss = loss(verts, tensor_pnts)
    return mean_loss

def count_mesh_angle(mesh):
    """
    计算网格内所有角度
    :param mesh:    网格
    :return:        三角形三个角度 torch torch torch
    """
    faces_packed = mesh.faces_packed()  # 每个面对应的idx
    verts_packed = mesh.verts_packed()  # 顶点坐标
    verts_faces = verts_packed[faces_packed]  # 每个面的顶点坐标
    a, b, c = verts_faces.unbind(1)  # 面三个顶点坐标
    vector_ab = b - a  # 边 向量
    vector_ac = c - a
    vector_bc = c - b
    norm_ab = vector_ab.norm(dim=1, p=2)  # 边长度
    norm_ac = vector_ac.norm(dim=1, p=2)
    norm_bc = vector_bc.norm(dim=1, p=2)
    A = torch.acos((vector_ab * vector_ac).sum(dim=1) / (norm_ab * norm_ac))  # 角度A
    B = torch.acos((-vector_ab * vector_bc).sum(dim=1) / (norm_ab * norm_bc))  # 角度B
    C = torch.acos((vector_bc * vector_ac).sum(dim=1) / (norm_ac * norm_bc))  # 角度C
    return A, B, C


def mesh_angle_loss(mesh, angle):
    """
    计算网格的角度损失
    :param mesh:    网格
    :param angle:   最优角度
    :return:        损失
    """
    angle_A, angle_B, angle_C = count_mesh_angle(mesh)
    loss = nn.MSELoss()
    target_angle = torch.full(angle_A.shape, angle)
    loss_mean = (loss(angle_A, target_angle) + loss(angle_B, target_angle) + loss(angle_C, target_angle)) / 3
    return loss_mean / torch.pi


def edge_length_loss(mesh):
    """
    计算网格的边长一致性损失
    :param mesh:    网格
    :return:        计算损失
    """
    verts = mesh.verts_packed()  # 获取顶点坐标
    edges = mesh.edges_packed()  # 获取边的顶点索引

    # 计算每条边的长度
    v_start, v_end = verts[edges[:, 0]], verts[edges[:, 1]]
    edge_lengths = torch.norm(v_start - v_end, p=2, dim=1)

    # 边长一致性损失：最小化边的长度变化（尽可能接近平均边长）
    mean_length = edge_lengths.mean()
    edge_loss = ((edge_lengths - mean_length).pow(2)).mean()

    return edge_loss


def laplacian_smoothing_loss(mesh):
    """
    基于统一矩阵的拉普拉斯平滑损失实现
    :param mesh:  网格
    :return:      计算损失
    """

    verts = mesh.verts_packed()  # 获取顶点
    edges = mesh.edges_packed()  # 获取边

    # 构建稀疏邻接矩阵
    num_verts = verts.shape[0]
    indices = torch.cat([edges, edges.flip(1)], dim=0).t()
    values = torch.ones(indices.shape[1], device=verts.device)
    adj_matrix = torch.sparse_coo_tensor(indices, values, (num_verts, num_verts))

    # 计算邻居均值位置
    degree_matrix = torch.sparse.sum(adj_matrix, dim=1).to_dense().unsqueeze(1)  # 度矩阵
    neighbor_sum = torch.sparse.mm(adj_matrix, verts)  # 邻居顶点的坐标和
    neighbor_mean = neighbor_sum / degree_matrix.clamp(min=1)  # 邻居均值位置，避免除0

    # 计算每个顶点到邻居均值的差距
    laplacian_loss = torch.norm(verts - neighbor_mean, p=2, dim=1).pow(2).mean()

    return laplacian_loss