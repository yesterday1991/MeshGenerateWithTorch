import math
import aspose.threed as a3d
import torch
import Loss
import MyGeometry
import Draw
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures.meshes import Meshes


global_scale = 0.001


# 比较浮点数是否一样
def IsFloatEq(a, b, eps):
    return abs(a - b) < eps


def generate_icosphere(level=0, device=None):
    """
    生成一个单位半径的二十面体球体（icosphere）
    :param level:       细分层数
    :param device:      pytorch的device
    :return:            网格
    """
    t = (1.0 + 5 ** 0.5) / 2.0  # 黄金比例
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
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


# # Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

#
# # 4 级别 网格面细分迭代次数， 每增加一个级别，每个面将产生四个新面
src_mesh = generate_icosphere(3, device)


# 定义初始网格
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# 几何文件名
file_name = "cube.step"

# 设置网格尺寸
mesh_vert_num = len(src_mesh.verts_packed())  # 初始网格顶点数

# 读取几何
print("初始化的网格点数量为：", mesh_vert_num)
geo = MyGeometry.OccGeo(file_name, mesh_vert_num)

# The optimizer 优化器
optimizer = torch.optim.SGD([deform_verts], lr=0.2, momentum=0.9)

# Number of optimization steps
Niter = 201

# Weight for the chamfer loss
w_surface = 10
w_match_edge = 20
w_angle = 0.3
w_edge_length = 1  # 0.01
w_normal = 0.1
w_laplacian = 0.05

# Plot period for the losses
plot_period = 100
# 网格修正周期
correct_mesh_period = 50


surface_losses = []
match_edge_losses = []
angle_losses = []
edge_length_losses = []
normal_losses = []
laplacian_losses = []


torch.set_printoptions(precision=8)
final_scale = geo.scale / global_scale
final_translation = torch.tensor([geo.center.X(), geo.center.Y(), geo.center.Z()]) * global_scale
#
print("开始计算")
for i in range(Niter):
    # Initialize optimizer 初始化
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    # 计算损失函数
    loss_edge_length = Loss.edge_length_loss(new_src_mesh)
    loss_angle = Loss.MeshAngleLoss(new_src_mesh, torch.pi / 3)
    loss_normal = mesh_laplacian_smoothing(new_src_mesh)
    loss_laplacian = Loss.laplacian_smoothing_loss(new_src_mesh)

    if i < 100:
        loss_surface = Loss.GeoProjSurfaceLoss(new_src_mesh, geo)
        total_loss = (
                    loss_surface * w_surface + loss_angle * w_angle + loss_edge_length * w_edge_length +
                    loss_normal * w_normal + loss_laplacian * w_laplacian)
        print(i, loss_surface, loss_angle, loss_edge_length, loss_normal, loss_laplacian)
        surface_losses.append(float(loss_surface.detach().cpu()))
    else:
        loss_edge_match = Loss.GeoMatchloss(new_src_mesh, geo)
        total_loss = (loss_angle * w_angle + loss_edge_length * w_edge_length +
                      loss_normal * w_normal + loss_laplacian * w_laplacian + loss_edge_match * w_match_edge)
        print(i, loss_angle, loss_edge_length, loss_normal, loss_laplacian, loss_edge_match)
        match_edge_losses.append(float(loss_edge_match.detach().cpu()))

    angle_losses.append(float(loss_angle.detach().cpu()))
    edge_length_losses.append(float(loss_edge_length.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))


    total_loss.backward()
    optimizer.step()

    # Plot mesh
    # if i % plot_period == 0:
    #     Draw.plot_pointcloud(new_src_mesh, geo, False)
    #     tmp_file_name = "tmp%d.obj" % i
    #     tmp_out_verts, tmp_out_faces = new_src_mesh.get_mesh_verts_faces(0)
    #     tmp_out_verts = tmp_out_verts / final_scale - final_translation
    #     save_obj(tmp_file_name, tmp_out_verts, tmp_out_faces)

# 保存obj以及stl
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts / final_scale - final_translation
final_obj = 'final_model.obj'
save_obj(final_obj, final_verts, final_faces)

scene = a3d.Scene.from_file("final_model.obj")
scene.save("final_model.stl")

Draw.plot_finall_loss(surface_losses, match_edge_losses, angle_losses, edge_length_losses, normal_losses, laplacian_losses)

