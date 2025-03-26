import aspose.threed as a3d
import torch
import Loss
import MyGeometry
import Draw
import Myio
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.loss import mesh_laplacian_smoothing
import numpy as np
import time
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils import ico_sphere
abc = False
# abc_data库里的几何存在单位，在occ读取的任意坐标数据为i中的1000倍，为了画图这里进行缩小
if abc:
    global_scale = 0.001
else:
    global_scale = 1
epsilon = 1e-8

ellipse_mesh = False
# chamfer进行包面收缩
wrapping = True

# # Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")



#
# # 4 级别 网格面细分迭代次数， 每增加一个级别，每个面将产生四个新面
if wrapping:
    verts, faces = Myio.generate_icosphere(4)
else:
    verts, faces = Myio.read_obj("final_model_mapping.obj")



mesh_vert_num = len(verts)  # 初始网格顶点数
# 几何文件名
file_name = "diamond.step"
# file_name = "visor.step"
# file_name = "Part 3.step"
# file_name = "new_part1.step"
# file_name = "dirty_cube.step"
# file_name = "Part_1_154.step"
# 设置网格尺寸


# 读取几何
print("初始化的网格点数量为：", mesh_vert_num)
geo = MyGeometry.OccGeo(file_name, mesh_vert_num)

final_scale = geo.scale / global_scale
final_translation = torch.tensor(geo.center) * global_scale

if not wrapping:
    verts = torch.tensor(verts, dtype=torch.float32)
    verts = (verts - final_translation) * final_scale
    verts_list = verts.numpy()
    verts = verts.to(device)
    Draw.plot_two_point(geo.plot_sample, verts_list)

if ellipse_mesh:
    # 计算椭球缩放大小
    box_len_x = geo.box[3] - geo.box[0]
    box_len_y = geo.box[4] - geo.box[1]
    box_len_z = geo.box[5] - geo.box[2]
    mesh_scale = np.array([box_len_x, box_len_y, box_len_z])
    mesh_scale =  mesh_scale/ np.max(mesh_scale)
    # 处理椭圆球点位置
    verts = np.array(verts)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    radius = np.linalg.norm(verts, axis=1)
    theta = np.arccos(z / radius)  # 计算极角
    phi = np.arctan2(y, x)  # 计算方位角
    x1 = radius * np.cos(phi) * np.sin(theta)
    y1 = radius * np.sin(phi) * np.sin(theta)
    z1 = radius * np.cos(theta)
    verts = np.stack([x * mesh_scale[0],
                      y * mesh_scale[1],
                      z * mesh_scale[2]], axis=-1)
# 构建网格
verts = torch.tensor(verts, dtype=torch.float32, device=device)
faces = torch.tensor(faces, dtype=torch.int64, device=device)


src_mesh = Meshes(verts=[verts], faces=[faces])
# 网格转换为图
MyGeometry.create_mesh_graph(src_mesh, geo)


# 定义初始网格
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# 网格点投影大致方向，控制点不往几何内移动
# src_verts = src_mesh.verts_packed()
# first_proj = torch.Tensor(MyGeometry.project_mesh_to_geo(src_verts, geo))
# proj_approx_direction = src_verts - first_proj


# The optimizer 优化器
optimizer = torch.optim.Adam([deform_verts], lr=0.01) #SGD([deform_verts], lr=1, momentum=0.9)##

# Number of optimization steps
Niter = 400

# Weight for the chamfer loss
w_chamfer = 1
w_surface = 1
w_match_edge = 1
w_angle = 1
w_edge_length = 1  # 0.01
w_normal = 0.1
w_laplacian = 0.1
step = 200

# 网格修正周期
correct_mesh_period = 50
plot_time = [0, 100, 300]

chamfer_losses = []
surface_losses = []
match_edge_losses = []
angle_losses = []
edge_length_losses = []
normal_losses = []
laplacian_losses = []
#sk-55eddd89c31f4d89afde5832565cbfd0

torch.set_printoptions(precision=8)

last_loss = 0
new_src_mesh = src_mesh
print("开始计算")
for i in range(Niter):
    # Initialize optimizer 初始化
    optimizer.zero_grad()
    #

    new_src_mesh = src_mesh.offset_verts(deform_verts)
    # 计算损失函数
    loss_edge_length = mesh_edge_loss(new_src_mesh)
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, "cot")
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_angle = Loss.mesh_angle_loss(new_src_mesh, torch.pi / 3)
    #
    #
    # loss_edge_match = Loss.geo_match_loss(new_src_mesh, geo)
    # w_match_edge += (i / Niter)
    #
    if wrapping:
        loss_chamfer = Loss.geo_chamfer_loss(new_src_mesh, geo)
        total_loss = ( loss_edge_length * w_edge_length + w_chamfer * loss_chamfer + loss_angle * w_angle +#  +
                       loss_normal * w_normal  + loss_laplacian * w_laplacian) #

        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        print("wrapping, ",i, loss_chamfer, loss_angle, loss_edge_length, loss_normal, loss_laplacian)
    else:
        geo.update_mesh_vert_to_face(new_src_mesh)
        loss_surface = Loss.geo_proj_surface_loss(new_src_mesh, geo)
        total_loss = (loss_edge_length * w_edge_length + loss_angle * w_angle + loss_surface * w_surface +
                      loss_normal * w_normal + loss_laplacian * w_laplacian)
        surface_losses.append(float(loss_surface.detach().cpu()))
        print(geo.mesh_dist_to_geo)
        print(i, loss_surface, loss_angle, loss_edge_length, loss_normal, loss_laplacian)  # , loss_edge_match,

    angle_losses.append(float(loss_angle.detach().cpu()))
    edge_length_losses.append(float(loss_edge_length.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))

    total_loss.backward()
    optimizer.step()

    # Plot mesh
    if i in plot_time:
    #     Draw.plot_pointcloud(new_src_mesh, geo, False)
        tmp_file_name = "tmp%d.obj" % i
        tmp_out_verts, tmp_out_faces = new_src_mesh.get_mesh_verts_faces(0)
        tmp_out_verts = tmp_out_verts / final_scale + final_translation
        Myio.save_obj(tmp_file_name, tmp_out_verts, tmp_out_faces)

# MyGeometry.project_mesh_to_geo(new_src_mesh, geo)
# print(geo.mesh_dist_to_geo)
# 保存obj以及stl
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts / final_scale + final_translation
if wrapping:
    final_obj = 'final_model.obj'
    final_stl = 'final_model.stl'
else:
    final_obj = 'final_model_proj.obj'
    final_stl = 'final_model_proj.stl'

Myio.save_obj(final_obj, final_verts, final_faces)
scene = a3d.Scene.from_file(final_obj)
scene.save(final_stl)

# mapping_pnts = torch.Tensor(MyGeometry.mapping_pnt_to_geo(new_src_mesh, geo, 2))
# # proj_pnts = torch.Tensor(MyGeometry.project_mesh_to_geo(new_src_mesh, geo))
# # # 边界点用路径点
# # path = []
# # for key, value in geo.edges.items():
# #     if not value.is_continuity:
# #         path += [index for index in value.shortest_path if index not in path]
# #
# # proj_pnts[path] = mapping_pnts[path]
#
# deform = mapping_pnts - new_src_mesh.verts_packed()
# new_src_mesh = new_src_mesh.offset_verts(deform)
# final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
# final_verts = final_verts / final_scale + final_translation
#
# final_obj = 'final_model_mapping.obj'
# Myio.save_obj(final_obj, final_verts, final_faces)
# scene = a3d.Scene.from_file("final_model_mapping.obj")
# scene.save("final_model_mapping.stl")


# from Myio import Meshes
# verts, faces = Myio.read_obj("final_model_test.obj")
# new_src_mesh = Meshes(verts=[verts], faces=[faces])

# #几何贴体
# final_mesh_local = Match.match_and_proj_final_mesh(new_src_mesh, geo)
# deform = final_mesh_local - new_src_mesh.verts_packed()
# new_src_mesh = new_src_mesh.offset_verts(deform)
#
# # Fetch the verts and faces of the final predicted mesh
# final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
#
# # Scale normalize back to the original target size
# final_verts = final_verts / final_scale - final_translation
#
# # Store the predicted mesh using save_obj
# final_obj = 'final_model_proj.obj'
# Myio.save_obj(final_obj, final_verts, final_faces)
# scene = a3d.Scene.from_file("final_model_proj.obj")
# scene.save("final_model_proj.stl")


Draw.plot_final_loss(surface_losses, match_edge_losses, angle_losses, edge_length_losses, normal_losses, laplacian_losses)

