import aspose.threed as a3d
import torch
import Loss
import MyGeometry
import Draw
import Myio
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.loss import mesh_laplacian_smoothing

import time

from pytorch3d.utils import ico_sphere
abc = True
# abc_data库里的几何存在单位，在occ读取的任意坐标数据为i中的1000倍，为了画图这里进行缩小
if abc:
    global_scale = 0.001
else:
    global_scale = 1
epsilon = 1e-8

# # Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

#
# # 4 级别 网格面细分迭代次数， 每增加一个级别，每个面将产生四个新面
src_mesh = Myio.generate_icosphere(4, device)

# 定义初始网格
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# 几何文件名
# file_name = "cube.step"
# file_name = "visor.step"
# file_name = "Part 3.step"
file_name = "new_part3.step"
# file_name = "dirty_cube.step"
# 设置网格尺寸
mesh_vert_num = len(src_mesh.verts_packed())  # 初始网格顶点数

# 读取几何
print("初始化的网格点数量为：", mesh_vert_num)
geo = MyGeometry.OccGeo(file_name, mesh_vert_num)
MyGeometry.create_mesh_graph(src_mesh, geo)

# 网格点投影大致方向，控制点不往几何内移动
# src_verts = src_mesh.verts_packed()
# first_proj = torch.Tensor(MyGeometry.project_mesh_to_geo(src_verts, geo))
# proj_approx_direction = src_verts - first_proj

final_scale = geo.scale / global_scale
final_translation = torch.tensor([geo.center.X(), geo.center.Y(), geo.center.Z()]) * global_scale

# The optimizer 优化器
optimizer = torch.optim.Adam([deform_verts], lr=0.01)#SGD([deform_verts], lr=1, momentum=0.9)

# Number of optimization steps
Niter = 200

# Weight for the chamfer loss
w_chamfer = 1
w_surface = 2
w_match_edge = 1
w_angle = 1
w_edge_length = 1  # 0.01
w_normal = 0.1
w_laplacian = 0.1

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

print("开始计算")
for i in range(Niter):
    # Initialize optimizer 初始化
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    # 计算损失函数
    # loss_edge_length = Loss.edge_length_loss(new_src_mesh)
    loss_edge_length = mesh_edge_loss(new_src_mesh)
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, "cot")
    # loss_laplacian = Loss.laplacian_smoothing_loss(new_src_mesh, 2)
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_angle = Loss.mesh_angle_loss(new_src_mesh, torch.pi / 3)
    #
    loss_surface = Loss.geo_proj_surface_loss(new_src_mesh, geo)
    loss_edge_match = Loss.geo_match_loss(new_src_mesh, geo)
    w_match_edge += (i / Niter)

    total_loss = (    loss_surface * w_surface  + loss_edge_length * w_edge_length +  loss_angle * w_angle +
                         loss_normal * w_normal +loss_laplacian * w_laplacian + w_match_edge * loss_edge_match) #
    print(i,loss_surface, loss_edge_match, loss_angle, loss_edge_length, loss_normal, loss_laplacian)


    surface_losses.append(float(loss_surface.detach().cpu()))
    match_edge_losses.append(float(loss_edge_match.detach().cpu()))

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
        tmp_out_verts = tmp_out_verts / final_scale - final_translation
        Myio.save_obj(tmp_file_name, tmp_out_verts, tmp_out_faces)

MyGeometry.project_mesh_to_geo(new_src_mesh, geo)
print(geo.mesh_dist_to_geo)
# 保存obj以及stl
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_obj1 = 'final_model_test.obj'
Myio.save_obj(final_obj1, final_verts, final_faces)
final_verts = final_verts / final_scale - final_translation
final_obj = 'final_model.obj'
Myio.save_obj(final_obj, final_verts, final_faces)

scene = a3d.Scene.from_file("final_model.obj")
scene.save("final_model.stl")

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

