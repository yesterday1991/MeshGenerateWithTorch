#
from pytorch3d.structures.meshes import Meshes
import torch

def save_obj(filepath, verts, faces):
    """
    将网格的顶点和面数据保存为 .obj 文件
    :param filepath:        保存 .obj 文件的路径   (str)
    :param verts:           顶点坐标，形状为 (V, 3) (torch.Tensor)
    :param faces:           面的顶点索引，形状为 (F, 3) (torch.Tensor)
    :return:
    """
    with open(filepath, 'w') as file:
        # 写入顶点
        for v in verts:
            file.write(f"v {v[0].item()} {v[1].item()} {v[2].item()}\n")

        # 写入面信息
        for face in faces:
            face_str = "f"
            for i in face:
                # 注意：.obj 文件的索引从 1 开始
                face_str += f" {i.item() + 1}"
            file.write(face_str + "\n")


def read_obj(filepath):
    """
    读取一个 .obj 文件，并返回顶点和面信息。

    参数:
        filepath (str): .obj 文件的路径。
        device (str): 顶点和面的张量放置的设备（如 'cpu' 或 'cuda'）。

    返回:
        verts (torch.Tensor): 顶点坐标，形状为 (V, 3)。
        faces (torch.Tensor): 面的顶点索引，形状为 (F, 3)。
    """
    verts = []
    faces = []

    with open(filepath, 'r') as file:
        for line in file:
            # 移除行首尾的空白符
            line = line.strip()

            # 顶点信息
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

            # 面信息
            elif line.startswith('f '):
                parts = line.split()
                # 将 .obj 的 1-based 索引转为 0-based 索引
                faces.append([int(part.split('/')[0]) - 1 for part in parts[1:4]])

    # 转换为 PyTorch 张量并放置在指定设备上
    verts = torch.tensor(verts, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)

    return verts, faces



def generate_icosphere(level=0, device=None):
    """
    生成一个单位半径的二十面体球体（icosphere）
    :param level:       细分层数
    :param device:      pytorch的device
    :return:            网格
    """
    verts = [
        [-0.5257, 0.8507, 0.0000], [0.5257, 0.8507, 0.0000], [-0.5257, -0.8507, 0.0000],
        [0.5257, -0.8507, 0.0000],[0.0000, -0.5257, 0.8507], [0.0000, 0.5257, 0.8507],
        [0.0000, -0.5257, -0.8507],[0.0000, 0.5257, -0.8507], [0.8507, 0.0000, -0.5257],
        [0.8507, 0.0000, 0.5257], [-0.8507, 0.0000, -0.5257], [-0.8507, 0.0000, 0.5257],
    ]
    # 定义初始的二十个面（三角形）
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    # 缓存中点索引的字典，避免重复创建中点
    mid_point_cache = {}

    # 逐级细分
    for _ in range(level):
        new_faces = []
        for v1, v2, v3 in faces:
            # 计算每条边的中点，直接嵌入细分过程
            edge_keys = [(v1, v2), (v2, v3), (v3, v1)]
            mid_points = []
            for v_start, v_end in edge_keys:
                # 确保边的顺序一致 (无向边)
                edge_key = tuple(sorted((v_start, v_end)))
                if edge_key not in mid_point_cache:
                    # 计算并归一化新顶点
                    midpoint = (torch.tensor(verts[v_start]) + torch.tensor(verts[v_end])).to(device) / 2.0
                    midpoint = midpoint / midpoint.norm()  # 归一化
                    verts.append(midpoint.tolist())
                    mid_point_cache[edge_key] = len(verts) - 1  # 新顶点索引
                mid_points.append(mid_point_cache[edge_key])

            # 获取三角形的4个面
            a, b, c = mid_points
            new_faces.extend([
                [v1, a, c],
                [v2, b, a],
                [v3, c, b],
                [a, b, c],
            ])
        faces = new_faces  # 更新面列表

    # 转换为 PyTorch 张量
    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)

    return Meshes(verts=[verts], faces=[faces])
