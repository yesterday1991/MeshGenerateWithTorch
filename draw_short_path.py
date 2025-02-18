from geomdl import BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width


def blend_knot(crv):
    """
    混合插值
    :param crv:
    :return:
    """

    blend = np.zeros(crv.ctrlpts_size)
    for i in range(crv.ctrlpts_size):                                 # 计算控制点插值比例
        for j in range(crv.degree):
            blend[i] += crv.knotvector[1 + i + j]
        blend[i] = blend[i] / crv.degree
    return blend


def blend_surface(crv1, crv2, crv3):
    """ 创建插值曲面

    :param crv1:
    :param crv2:
    :param crv3:
    :return:
    """
    surf = BSpline.Surface()
    surf.degree_u = crv3.degree                                         # u方向次数
    surf.degree_v = crv1.degree                                         # v方向次数
    # 插值控制点坐标
    nctrlpnt_u = crv3.ctrlpts_size                                 # u 方向控制点数量
    nctrlpnt_v = crv1.ctrlpts_size                                 # v 方向控制点数量

    surfcontrol = blend_knot(crv3)                                       # 储存控制点插值比例
    surfctrlpnt = np.zeros((nctrlpnt_u, nctrlpnt_v, 3))  # 插值后的控制点 u行 v列
    ctrlpnt1 = np.array(crv1.ctrlpts)                              # 插值的控制顶点1
    ctrlpnt2 = np.array(crv2.ctrlpts)                              # 插值的控制顶点2

    for i in range(crv3.ctrlpts_size):         # 根据比例计算控制点坐标
        controlpoint = (1 - surfcontrol[i]) * ctrlpnt1 + surfcontrol[i] * ctrlpnt2
        surfctrlpnt[i] = controlpoint
    surf.ctrlpts2d = surfctrlpnt.tolist()
    surf.knotvector_u = crv3.knotvector
    surf.knotvector_v = crv1.knotvector
    return surf



curve1 = BSpline.Curve()
curve2 = BSpline.Curve()
curve3 = BSpline.Curve()


curve4 = BSpline.Curve()
curve5 = BSpline.Curve()

curve1.degree = 3
curve2.degree = 3
curve3.degree = 3

curve4.degree = 3
curve5.degree = 3

knot1 = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0]
point1 = [[-25.0, -25.0, -10.0], [-15.0, -25.0, -7.0], [-5.0, -25.0, -5.0], [5.0, -25.0, -5.0], [15.0, -25.0, -7.0], [25.0, -25.0, -10.0]]

knot2 = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0]
point2 = [[-25.0, 0.0, -10.0], [-15.0, 0.0, -7.0], [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [15.0, 0.0, -7.0], [25.0, 0.0, -10.0]]

knot3 = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0]
point3 = [[-25.0, 0.0, -10.0], [-25.0, -5.0, -10.0], [-25.0, 10.0, -10.0], [-25.0, -15.0, -10.0], [-25.0, -20.0, -10.0], [-25.0, -25.0, -10.0]]


knot4 = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0]
point4 = [[-25.0, -35.0, -15.0], [-15.0, -35.0, -12.0], [-5.0, -35.0, -10], [5.0, -35.0, -10.0], [15.0, -35.0, -12.0], [25.0, -35.0, -15.0]]

knot5 = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0]
point5 = [[-25.0, -25.0, -10.0], [-25.0, -27.0, -11.0], [-25.0, -29.0, -12.0], [-25.0, -31.0, -13.0], [-25.0, -34.0, -18.0], [-25.0, -35.0, -15.0]]

curve1.ctrlpts = point1
curve2.ctrlpts = point2
curve3.ctrlpts = point3
curve4.ctrlpts = point4
curve5.ctrlpts = point5

curve1.knotvector = knot1
curve2.knotvector = knot2
curve3.knotvector = knot3
curve4.knotvector = knot4
curve5.knotvector = knot5

curve1.delta = 0.01
curve2.delta = 0.01
curve3.delta = 0.01
curve4.delta = 0.01
curve5.delta = 0.01

sample_list = [0.0, 0.15, 0.47, 0.83, 1.0]
sample_pnt = curve1.evaluate_list(sample_list)
np_sample_pnt = np.array(sample_pnt)
surface1 = blend_surface(curve1, curve2, curve3)
surface2 = blend_surface(curve1, curve4, curve5)

loop = [curve1]

mesh_pnt = np.array([[-25.0, -23.0, -5.0],
                    [-13, -24, -2],
                    [1 , -25, -2],
                    [14, -27, -3],
                    [25, -22, -5],
                    [-18.0, -10.0, -2.0],
                    [-6, -8, -1.5],
                    [7 , -9, -2.2],
                    [19, -11, -3],
                    [-23.0, 0.0, -2.0],
                    [-12, 0, -1.5],
                    [2 , 0, -1.5],
                    [14, 0, -2],
                    [25, 0, -2.5],
                     ])

mesh_edge = np.array([
[0, 1], [1, 2], [2, 3], [3, 4],
[0, 5], [1, 5],
[5, 6], [1, 6], [2, 6],
[6, 7], [2, 7], [3, 7],
[7, 8], [3, 8], [4, 8],
[0, 9], [5, 9],
[9, 10], [5, 10], [6, 10],
[10, 11], [6, 11], [7, 11],
[11, 12], [7, 12], [8, 12],
[12, 13],[8, 13], [4, 13],
])


short_mesh_pnt = mesh_pnt[0:5]
short_mesh_edge = mesh_edge[0:3]
short_verts_edges = mesh_pnt[mesh_edge]

face_mesh_pnt = mesh_pnt[5:]
face_mesh_edge = mesh_edge[4:]
face_verts_edges = mesh_pnt[face_mesh_edge]

ctrlpts = []
curvepts = []
#
fig = plt.figure(figsize=(10.67, 8), dpi=300)
ax = Axes3D(fig, auto_add_to_figure=False)
ax.set_xlim3d(-30, 30)
ax.set_ylim3d(-80, 30)
ax.set_zlim(-30, 30)
fig.add_axes(ax)
plt.axis('off')


ax.scatter3D(np_sample_pnt[:, 0], np_sample_pnt[:, 1], np_sample_pnt[:, 2], color="blue", s=5, alpha=0.5)
ax.scatter3D(short_mesh_pnt[:, 0], short_mesh_pnt[:, 1], short_mesh_pnt[:, 2], color="blue", s=5)
ax.scatter3D(face_mesh_pnt[:, 0], face_mesh_pnt[:, 1], face_mesh_pnt[:, 2], color="black", s=5)

for i in short_verts_edges:
    ax.plot(i[:, 0], i[:, 1], i[:, 2], color="blue", linewidth=0.5)

for i in face_verts_edges:
    ax.plot(i[:, 0], i[:, 1], i[:, 2], color="black", linewidth=0.5)


uvw = np_sample_pnt - short_mesh_pnt
ax.quiver(mesh_pnt[0][0], mesh_pnt[0][1], mesh_pnt[0][2], uvw[0, 0], uvw[0, 1], uvw[0, 2], linewidth=1)
ax.quiver(mesh_pnt[1][0], mesh_pnt[1][1], mesh_pnt[1][2], uvw[1, 0], uvw[1, 1], uvw[1, 2], linewidth=1)
ax.quiver(mesh_pnt[2][0], mesh_pnt[2][1], mesh_pnt[2][2], uvw[2, 0], uvw[2, 1], uvw[2, 2], linewidth=1)
ax.quiver(mesh_pnt[3][0], mesh_pnt[3][1], mesh_pnt[3][2], uvw[3, 0], uvw[3, 1], uvw[3, 2], linewidth=1)
ax.quiver(mesh_pnt[4][0], mesh_pnt[4][1], mesh_pnt[4][2], uvw[4, 0], uvw[4, 1], uvw[4, 2], linewidth=1)

if loop:
    for i in range(len(loop)):
        loop[i].evaluate()
        ctrlpts.append(np.array(loop[i].ctrlpts))
        curvepts.append(np.array(loop[i].evalpts))

    for i in range(len(loop)):
        ax.plot(curvepts[i][:, 0], curvepts[i][:, 1], curvepts[i][:, 2], color='red', linestyle='-')

surface1.evaluate()
surfpts1 = np.array(surface1.evalpts)
ax.plot_trisurf(surfpts1[:, 0], surfpts1[:, 1], surfpts1[:, 2], color='grey', alpha=0.5)
surface2.evaluate()
surfpts2 = np.array(surface2.evalpts)
ax.plot_trisurf(surfpts2[:, 0], surfpts2[:, 1], surfpts2[:, 2], color='grey', alpha=0.5)
plt.show()

