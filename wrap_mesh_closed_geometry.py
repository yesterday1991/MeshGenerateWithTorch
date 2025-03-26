import aspose.threed as a3d
import torch

import Draw
import Myio
import MyGeometry


def find_neighbors(target, indx):
    prev_num = None
    next_num = None
    for i, num in enumerate(target):
        if num == indx:
            prev_num = target[i - 1] if i > 0 else None  # 前一个数字
            next_num = target[i + 1] if i < len(target) - 1 else None # 后一个数字
    return prev_num, next_num


def edge_exists(mesh, v1, v2):

    faces = mesh.faces_packed()
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
    :param faces: 
    :param fist: 
    :param last: 
    :param index: 
    :param common_elements: 
    :return: 
    """

    faces = delete_face(faces, fist, index, common_elements)
    faces = delete_face(faces, last, index, common_elements)
    faces = torch.cat((faces, torch.tensor([[fist, last, common_elements], [fist, index, last]])), dim=0)
    return faces


def is_triangle_in_faces(faces, i, j, k):

    # 将查询的三个点按升序排序，
    query = torch.tensor([i, j, k]).sort().values  # [i, j, k] → 排序后
    # 对 faces 的每一行进行排序
    sorted_faces = torch.sort(faces, dim=1).values  # 对每个面按行排序
    # 判断查询的面是否存在
    return (sorted_faces == query).all(dim=1).any()



verts, faces = Myio.read_obj("final_model_test.obj")
verts = torch.tensor(verts, dtype=torch.float32)
faces = torch.tensor(faces, dtype=torch.int64)


file_name = "new_part15.step"
mesh_vert_num =verts.shape[0]
geo = MyGeometry.OccGeo(file_name, mesh_vert_num)


global_scale = 1
final_scale = geo.scale / global_scale
final_translation = torch.tensor(geo.center) * global_scale

verts = (verts - final_translation) * final_scale
new_src_mesh = Myio.Meshes(verts=[verts], faces=[faces])
MyGeometry.create_mesh_graph(new_src_mesh, geo)
Draw.plot_two_point(geo.plot_sample, new_src_mesh.verts_packed().numpy())

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
mapping_pnts = torch.Tensor(MyGeometry.mapping_pnt_to_geo(new_src_mesh, geo, 2))
# 进行边交换

if len(geo.mesh_duplicates) != 0:
    # 确定存在问题的边的位置
    for index, keys in geo.mesh_duplicates.items():
        # 先处理一个点在2条线上的情况
        if len(keys) == 2:
            # 匹配点按照key先后顺序，key数字越大则重复的点会在该曲线上，另一条线则会缺失该点
            path_problem = geo.edges[keys[0]].shortest_path
            path_normal = geo.edges[keys[1]].shortest_path
            # 缺失点在问题曲线上的前后点编号
            fix_first, fix_last = find_neighbors(path_problem, index)
            # 缺失点在正常曲线上的前后点编号，用于定位存在问题的面片
            normal_first, normal_last = find_neighbors(path_normal, index)
            # 找到的连接情况，start到fix_end之间的形状无法保持，需要进行边交换
            #                  fix_end
            #                    |
            #       start------index--------normal_end
            start = ({fix_first, fix_last} & {normal_first, normal_last}).pop()
            fix_end = fix_first if fix_first != start else fix_last
            normal_end = normal_first if normal_first != start else normal_last
            # 确认是fix_end与start之间缺失没有边相连
            if not edge_exists(new_src_mesh, start, fix_end):
                print("尝试进行边交换", start, fix_end, index)
                # 找到包含fist和index, index和last的总共的四个面
                mask_v1 = (faces == start)
                mask_v2 = (faces == fix_end)
                mask_v3 = (faces == index)
                row_mask1 = torch.logical_and(mask_v1.any(dim=1), mask_v3.any(dim=1))
                row_mask2 = torch.logical_and(mask_v2.any(dim=1), mask_v3.any(dim=1))
                find_face1 = faces[torch.where(row_mask1)]
                find_face2 = faces[torch.where(row_mask2)]
                # 从四个面中找到除去start, fix_end, index的顶点
                pnts1 = find_face1[(find_face1 != start) & (find_face1 != index)]
                pnts2 = find_face2[(find_face2 != fix_end) & (find_face2 != index)]
                mask = torch.isin(pnts1, pnts2)
                common_elements = pnts1[mask]
                # 目前看到需要进行边交换有两种情况
                # 如果能找到common_elements，则在四边形内部边交换即可
                #       common_elements----fix_end
                #         |             \    |
                #       start-------------index------normal_end
                if len(common_elements) != 0:
                    # 两个三角形顶点分别为fist、index、common_elements 和 last、index、common_elements
                    faces = mesh_edge_swap(faces, start, fix_end, index, common_elements)
                # 第二种情况需要进行两次边交换
                #         v2------v1----fix_end
                #         |   \    |    /
                #       start----index------normal_end
                else:
                    if normal_end in pnts2:
                        v1 = pnts2[pnts2 != normal_end].item()
                        if is_triangle_in_faces(faces, pnts1[0].item(), index, v1):
                            v2 = pnts1[0].item()
                        elif is_triangle_in_faces(faces, pnts1[1].item(), index, v1):
                            v2 = pnts1[1].item()
                        else:
                            print("其他情况，无法进行交换")
                            continue
                        # 进行两次交换即可使得 start fix_end 之间有边相连
                        faces = mesh_edge_swap(faces, start, v1, index, v2)
                        faces = mesh_edge_swap(faces, start, fix_end, index, v1)
                    else:
                        print("无法判断方向，normal_end顶点不在 index--fix_end构建的面片中")

# path = geo.faces[4].mapping_inside_mesh_index
proj_pnts = torch.Tensor(MyGeometry.project_mesh_to_geo(new_src_mesh, geo))
print(geo.mesh_dist_to_geo)
# # 边界点用路径点
# path = []
# for key, value in geo.edges.items():
#     if not value.is_continuity:
#         path += [index for index in value.shortest_path if index not in path]
#

# mapping_pnts[path] = proj_pnts[path]
# Draw.plot_point(geo.plot_sample)
# Draw.plot_point(proj_pnts.numpy())
deform = mapping_pnts - new_src_mesh.verts_packed()
new_src_mesh = new_src_mesh.offset_verts(deform)
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts / final_scale + final_translation

final_obj = 'final_model_mapping.obj'
Myio.save_obj(final_obj, final_verts, faces)
scene = a3d.Scene.from_file("final_model_mapping.obj")
scene.save("final_model_mapping.stl")