
import torch
from OCC.Core.gp import gp_Pnt
import torch.nn as nn
from MyGeometry import project_mesh_vert_to_geo


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


def match_sample_geo_with_mesh(src_mesh, geo, had_matched_indx, geo_flag):
    """
    将geo中采样点与src_mesh网格中网格点按顶点，边，面顺序按最短距离进行匹配
    :param src_mesh:            网格
    :param geo:                 几何
    :param had_matched_indx:    已经匹配过的网格点编号 tensor
    :param geo_flag:            处理边界采样点还是面内采样点
    :return:                    几何采样点集合， 采样点对应的网格点编号
    """
    sample_total = []  # 将采样点统一放入一个数组中
    match_indx = torch.tensor([], dtype=torch.int)
    if geo_flag == 0:   # 面
        matched = geo.match_sample_surface
    elif geo_flag == 1: # 边
        matched = geo.match_sample_edge
    elif geo_flag == 2: # 顶点
        matched = geo.match_sample_vert
    elif geo_flag == 3:
        matched = geo.important_sample_pnt
    matched = matched.tolist()
    for j in matched:
        current_match = match_pnt_to_mesh(src_mesh, j, had_matched_indx)  # 匹配
        sample_total.append(j)
        had_matched_indx = torch.cat((had_matched_indx, torch.tensor([current_match], dtype=torch.int)))
        match_indx = torch.cat((match_indx, torch.tensor([current_match], dtype=torch.int)))

    mesh_verts = src_mesh.verts_packed()
    # plot_two_point(sample_total, match_verts_tensor)
    return torch.tensor(sample_total), match_indx





def geo_match_loss(trg_mesh, src_geo):
    """
    计算几何边的匹配损失，用于处理几何边界
    :param trg_mesh:    网格
    :param src_geo:     几何
    :return:            计算损失
    """
    mesh_verts = trg_mesh.verts_packed()
    had_matched = torch.tensor([], dtype=torch.int)
    matched_pnt_tensor, current_matched = match_sample_geo_with_mesh(trg_mesh, src_geo, had_matched, 3)
    match_verts_tensor = mesh_verts[current_matched]
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