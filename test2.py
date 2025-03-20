import aspose.threed as a3d
import torch

import Draw
import Myio
import MyGeometry

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

mapping_pnts = torch.Tensor(MyGeometry.mapping_pnt_to_geo(new_src_mesh, geo, 2))

path = geo.faces[4].mapping_inside_mesh_index
proj_pnts = torch.Tensor(MyGeometry.project_mesh_to_geo(new_src_mesh, geo))
print(geo.mesh_dist_to_geo)
# # 边界点用路径点
# path = []
# for key, value in geo.edges.items():
#     if not value.is_continuity:
#         path += [index for index in value.shortest_path if index not in path]
#

mapping_pnts[path] = proj_pnts[path]
# Draw.plot_point(geo.plot_sample)
# Draw.plot_point(proj_pnts.numpy())
deform = mapping_pnts - new_src_mesh.verts_packed()
new_src_mesh = new_src_mesh.offset_verts(deform)
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts / final_scale + final_translation

final_obj = 'final_model_mapping.obj'
Myio.save_obj(final_obj, final_verts, final_faces)
scene = a3d.Scene.from_file("final_model_mapping.obj")
scene.save("final_model_mapping.stl")