
import torch
import torch.nn as nn

import Draw
import MyGeometry
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
    total_sample_num = src_geo.mesh_vert_num * 2
    sample_geo = src_geo.sample_geo(total_sample_num)
    sample_geo_tensor = torch.tensor(sample_geo, dtype=torch.float32)
    # 网格采样
    sample_mesh = sample_points_from_meshes(trg_mesh, total_sample_num)
    # 计算损失
    loss_chamfer = chamfer_distance(sample_mesh.squeeze(0), sample_geo_tensor)
    return loss_chamfer



def geo_match_loss(trg_mesh, src_geo):
    """
    计算几何边的匹配损失，用于处理几何边界
    :param trg_mesh:    网格
    :param src_geo:     几何
    :return:            计算损失
    """

    # 在边界上的网格点
    match_pnt = src_geo.edge_mesh_index
    # 找到网格点所对应的采样点
    matched_pnt = []
    for mesh_index in match_pnt:
        for key, value in src_geo.edges.items():
            if mesh_index in value.shortest_path:
                tmp_index = value.shortest_path.index(mesh_index)
                pnt_xyz = value.shortest_path_sample[tmp_index]
                matched_pnt.append(pnt_xyz)
                break

    matched_pnt_tensor = torch.tensor(matched_pnt)
    # 网格点
    mesh_verts = trg_mesh.verts_packed()
    match_verts_tensor = mesh_verts[match_pnt]
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

    # 获取投影点
    if src_geo.using_map:
        tensor_pnts = torch.Tensor(MyGeometry.mapping_pnt_to_geo(trg_mesh, src_geo, 2))
        print("using mapping")
    else:
        tensor_pnts = torch.Tensor(MyGeometry.project_mesh_to_geo(trg_mesh, src_geo))
        print("using projection")
    # 网格点
    proj_mesh_vert = trg_mesh.verts_packed()
    # 找到内部点
    if src_geo.using_map:
        inside_mesh_pnt_index = src_geo.face_mesh_index
        # 找到内部点投影的坐标
        tensor_pnts = tensor_pnts[inside_mesh_pnt_index]
        # 找到内部点原始的坐标
        proj_mesh_vert = proj_mesh_vert[inside_mesh_pnt_index]
    # 网格点与投影点计算损失
    loss = nn.MSELoss(reduction='none')
    total_loss = loss(proj_mesh_vert, tensor_pnts)
    weighted_loss = total_loss.sum(dim=1)
    # Draw.plot_two_point(proj_mesh_vert.detach().numpy(), tensor_pnts.detach().numpy())
    mean_loss = torch.mean(weighted_loss)
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


def compute_cotangent_weights(mesh):

    verts = mesh.verts_packed()  # 获取顶点
    faces= mesh.faces_packed() # 获取面
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    # 计算每条边的cotangent权重
    face_verts = verts[faces]
    # 三角形三个顶点与三条边长
    # a - v0,      b - v1,        c - v2
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    a = (v1 - v2).norm(dim=1)
    b = (v0 - v2).norm(dim=1)
    c = (v0 - v1).norm(dim=1)
    # 两个三角形面积面积公式可得到每个角度的cot值
    # 公式1 area = sqrt(S(S-a)(S-b)(S-c)) S = (a+b+c)/2
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)).sqrt()
    # 公式2 4area = (a*a + b*b - c*c) / cotA = (b*b + c*c - a*a) / cotB = (a*a + c*c - b*b) / cotC
    cota = (b * b + c * c - a * a) / (4 * area)
    cotb = (a * a + c * c - b * b) / (4 * area)
    cotc = (a * a + b * b - c * c) / (4 * area)
    cot = torch.stack([cota, cotb, cotc], dim=1)
    # 1. 创建邻接矩阵 A
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, num_faces * 3)
    adj_matrix = torch.sparse_coo_tensor(idx, cot.view(-1), (num_verts, num_verts))
    adj_matrix = adj_matrix + adj_matrix.t()  # 对称矩阵
    return adj_matrix



def laplacian_smoothing_loss(mesh, model):
    """
    基于统一矩阵的拉普拉斯平滑损失实现
    :param mesh:  网格
    :param model: 模式 1:Uniform拉普拉斯矩阵 2:Cotangent 权重拉普拉斯矩阵
    :return:      计算损失
    """

    verts = mesh.verts_packed()  # 获取顶点
    edges = mesh.edges_packed()  # 获取边
    faces= mesh.faces_packed() # 获取面
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    # 构建拉普拉斯矩阵
    if model == 1:
        #L[i, j] =    -1       , if i == j
        #L[i, j] = 1 / deg(i)  , if (i, j) is an edge
        #L[i, j] =    0        , otherwise
        # 1. 创建邻接矩阵 A
        indices = torch.cat([edges, edges.flip(1)], dim=0).t()
        values = torch.ones(indices.shape[1], device=verts.device)       # Uniform 初始权重为 1
        adj_matrix = torch.sparse_coo_tensor(indices, values, (num_verts, num_verts))

        # 2. 计算度数矩阵D,并归一化
        degree = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        degree_inv = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
        e0, e1 = edges.unbind(1)
        normalized_values = torch.cat([degree_inv[e0], degree_inv[e1]])
        normalized_adj_matrix = torch.sparse_coo_tensor(indices,normalized_values, (num_verts, num_verts))

        # 计算归一化的拉普拉斯矩阵 L = D - A
        laplacian_matrix = normalized_adj_matrix - torch.sparse_coo_tensor(
            indices=torch.stack([torch.arange(num_verts), torch.arange(num_verts)]),
            values=torch.ones(num_verts),
            size=(num_verts, num_verts)
        )
        loss = torch.sum(laplacian_matrix.mm(verts).norm(dim=1), dim=0) / num_verts
    elif model == 2:
        # 1. 创建邻接矩阵 A
        adj_matrix = compute_cotangent_weights(mesh)
        #  2. 计算度数矩阵D
        degree = torch.sparse.sum(adj_matrix, dim=1).to_dense().view(-1, 1)
        idx = degree > 0
        degree[idx] = 1.0 / degree[idx]
        loss = adj_matrix.mm(verts) * degree - verts
        loss = torch.norm(loss, dim=1).mean()
    return loss