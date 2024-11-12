
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLib import breplib
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRepTools import breptools
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeAnalysis import shapeanalysis
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import TopTools_DataMapOfShapeInteger
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods, TopoDS_Vertex
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
import numpy as np
import math


geo_eps = 1e-4
# 判断面之间是否连续的角度
continuity_face_angel = 0.94

def pnt_in_polygon(u, v, polygon):
    """
    射线法判断
    :param u:   二维坐标u
    :param v:   二维坐标v
    :param polygon:     离散多边形
    :return:    ture 内  false 外
    """
    crossing = 0
    n = len(polygon)
    for k in range(n):
        a = np.array(polygon[k])
        b = np.array(polygon[(k + 1) % n])
        # 跳过水平边
        if a[0] == b[0]:
            continue
        # 计算斜率
        slope = (b[1] - a[1]) / (b[0] - a[0])
        # 条件判断
        cond1 = (a[0] <= u) and (u < b[0])
        cond2 = (b[0] <= u) and (u < a[0])
        above = v < slope * (u - a[0]) + a[1]
        # 如果满足条件则交点计数
        if (cond1 or cond2) and above:
            crossing += 1
    return (crossing % 2 != 0)


def get_edge_length(topo_edge):
    """
    计算边长度
    :param topo_edge:  OCC拓扑边
    :return:            长度
    """
    cgprop = GProp_GProps()
    brepgprop.LinearProperties(topo_edge, cgprop)
    crv_length = cgprop.Mass()
    return crv_length


def get_sufaces_area(topo_face):
    """
    计算所有几何面面积
    :return: 面积
    """
    fgprop = GProp_GProps()
    brepgprop.SurfaceProperties(topo_face, fgprop)
    surface_area = fgprop.Mass()
    return abs(surface_area)


def angle_in_curve(dim, curve, first, mid, last):
    """
    计算给定曲线上 first, mid last 三个参数坐标构成折线段角度的cos值
    :param dim:     curve的维度(二维或三维)
    :param curve:   OCC中的曲线
    :param first:   第一个点参数坐标
    :param mid:     第二个点参数坐标
    :param last:    第三个点参数坐标
    :return:        折线所构成角度的cos值
    """
    # 三个点坐标
    a = curve.Value(first)
    b = curve.Value(mid)
    c = curve.Value(last)
    # 计算起点终点长度
    distance = a.Distance(c)
    if distance < 0.01:  #长度过小，不进行插值
        return 1
    if dim == 2:
        a_xy = np.array([a.X(), a.Y()])
        b_xy = np.array([b.X(), b.Y()])
        c_xy = np.array([c.X(), c.Y()])
    elif dim == 3:
        a_xy = np.array([a.X(), a.Y(), a.Z()])
        b_xy = np.array([b.X(), b.Y(), a.Z()])
        c_xy = np.array([c.X(), c.Y(), a.Z()])
    else:
        return 0
    # 三个点构成的向量
    vector_ba = b_xy - a_xy
    vector_cb = c_xy - b_xy
    # 向量长度
    len_ba = np.linalg.norm(vector_ba)
    len_cb = np.linalg.norm(vector_cb)
    # 构成的角度
    return np.dot(vector_ba, vector_cb) / (len_ba * len_cb)


def discrete_curve(curve, first, last, orientation):
    """
    离散曲线curve
    :param curve:   OCC中的曲线
    :param first:   曲线的起始参数坐标
    :param last:    曲线的终止参数坐标
    :param orientation      curve对应的edge的方向
    :return:        np离散点xyz,   [[x1,y1,z1], [x2,y2,z2],.....]
                    np离散点参数坐标[t1,t2,..]
    """
    # 计算离散点
    isclosed = False
    pnt1 = curve.Value(first)
    pnt2 = curve.Value(last)
    if pnt1.IsEqual(pnt2, geo_eps):
        isclosed = True
    if isclosed:
        # 闭合曲线取5个点为初始点
        tmp_total = first + last
        tmp_1d5 = first + tmp_total / 5
        tmp_2d5 = first + tmp_total / 5 * 2
        tmp_3d5 = first + tmp_total / 5 * 3
        tmp_4d5 = first + tmp_total / 5 * 4
        subdivide_array = np.array([first, tmp_1d5, tmp_2d5, tmp_3d5, tmp_4d5, last])
    else:
        subdivide_array = np.array([first, last])
    # 开始离散
    subdivide_flag = True
    while subdivide_flag:
        subdivide_array_len = len(subdivide_array)
        subdivide_flag = False  # 默认不会发生插值
        for i in range(subdivide_array_len - 1):
            tmp_first = subdivide_array[i]
            tmp_last = subdivide_array[i + 1]
            tmp_mid = (tmp_first + tmp_last) / 2
            angle = angle_in_curve(3, curve, tmp_first, tmp_mid, tmp_last)
            if angle < 0.95:  # 不是直线，中点加入数组
                subdivide_array = np.append(subdivide_array, tmp_mid)  # 插入的值放在数组最后，不影响前面计算
                subdivide_flag = True
            subdivide_array.sort()  # 对插入值后的数组进行排序
    # 边方向与几何方向相反，进行翻转
    if orientation:
        subdivide_array = np.flip(subdivide_array)
    # 将参数坐标转换成gp_pnt
    pnt_list = []
    for t in subdivide_array:
        gp_point = curve.Value(t)
        pnt_list.append(gp_point)
    # 去除距离过小的点
    for i in range(len(pnt_list) - 1, 1, -1):
        pnt1 = pnt_list[i]
        pnt2 = pnt_list[i - 1]
        distance = pnt1.Distance(pnt2)
        if distance < 0.005:
            pnt_list.pop(i)
            subdivide_array = np.delete(subdivide_array, i)
    # 将gp_pnt转换成xyzlist
    pnts_xyz = []
    for i in pnt_list:
        point_xyz = [i.X(), i.Y(), i.Z()]
        pnts_xyz.append(point_xyz)
    return np.array(pnts_xyz), subdivide_array


def discrete_Wire(wire, face):
    """
    离散surface上的环wire
    :param wire:        occ中的环
    :param face:        occ中的面
    :return:            离散坐标[u1,v1], [u2,v2],.....]
    """
    # 计算离散点
    wire_pnt = []
    exp_edge = BRepTools_WireExplorer(wire, face)
    surface = BRep_Tool.Surface(face)
    while exp_edge.More():
        is_closed = False
        edge = exp_edge.Current()  # 当前拓扑参数曲线
        curve2d_inf = BRep_Tool.CurveOnSurface(edge, face)  # 参数曲线信息
        curve2d = curve2d_inf[0]
        pnt_2d1 = curve2d.Value(curve2d_inf[1])
        pnt_2d2 = curve2d.Value(curve2d_inf[2])
        pnt1 = surface.Value(pnt_2d1.X(), pnt_2d1.Y())
        pnt2 = surface.Value(pnt_2d2.X(), pnt_2d2.Y())
        if pnt1.IsEqual(pnt2, geo_eps):
            is_closed = True
        if is_closed:
            # 闭合曲线首位在同一个点，两个点无法离散，曲线为圆时，三个点共线，所以取5个点
            tmp_total = curve2d_inf[1] + curve2d_inf[2]
            tmp_1d5 = curve2d_inf[1] + tmp_total / 5
            tmp_2d5 = curve2d_inf[1] + tmp_total / 5 * 2
            tmp_3d5 = curve2d_inf[1] + tmp_total / 5 * 3
            tmp_4d5 = curve2d_inf[1] + tmp_total / 5 * 4
            subdivide_array = np.array([curve2d_inf[1], tmp_1d5, tmp_2d5, tmp_3d5, tmp_4d5, curve2d_inf[2]])
        else:
            subdivide_array = np.array([curve2d_inf[1], curve2d_inf[2]])  # 离散点二维坐标
        subdivide_flag = True
        while subdivide_flag:
            subdivide_array_len = len(subdivide_array)
            subdivide_flag = False  # 默认不会发生插值
            for i in range(subdivide_array_len - 1):
                first = subdivide_array[i]
                last = subdivide_array[i + 1]
                mid = (first + last) / 2
                angle = angle_in_curve(2, curve2d, first, mid, last)
                if angle < 0.99:  # 不是直线，中点加入数组
                    subdivide_array = np.append(subdivide_array, mid)  # 插入的值放在数组最后，不影响前面计算
                    subdivide_flag = True  # 发生了插值，此时还需要进行下一轮检测
            subdivide_array.sort()  # 对插入值后的数组进行排序
        if edge.Orientation():  # 边方向与几何方向相反，进行翻转
            subdivide_array = np.flip(subdivide_array)
        for t in subdivide_array:
            pnt_uv = curve2d.Value(t)  # 计算参数坐标值
            if wire_pnt:
                if not pnt_uv.IsEqual(wire_pnt[-1], geo_eps):
                    wire_pnt.append(pnt_uv)
            else:
                wire_pnt.append(pnt_uv)
        exp_edge.Next()
    # 去除距离过小的点，会导致计算可能出现平行情况影响环内外判断
    for i in range(len(wire_pnt) - 1, 1, -1):
        pnt1 = wire_pnt[i]
        pnt2 = wire_pnt[i - 1]
        dist = pnt1.Distance(pnt2)
        if dist < 0.005:
            wire_pnt.pop(i)
    # 将离散点转化成uv坐标储存
    wire_pnt_uv = []
    for i in wire_pnt:
        point_uv = [i.X(), i.Y()]
        wire_pnt_uv.append(point_uv)
    return wire_pnt_uv


def pnt_face_box_distance(pnt, face):
    """
    计算点与面之间的box距离
    :param pnt:     点
    :param face:    面
    :return:        距离
    """
    bnd_box_pnt = Bnd_Box(pnt, pnt)  # 构建点的bnd_box
    bbox = Bnd_Box()
    bbox.SetGap(geo_eps)
    brepbndlib.Add(face, bbox)
    box_dist = bbox.Distance(bnd_box_pnt)
    return box_dist


def project_to_line(p1, p2, base):
    """
    直线投影
    :param p1: 起点坐标xyz
    :param p2: 终点坐标xyz
    :param base: 投影点xyz
    :return: 投影结果xyz, 在直线上的参数坐标dt
    """
    np_p1 = np.array(p1)
    np_p2 = np.array(p2)
    np_base = np.array(base)
    v1 = np_p2 - np_p1
    v2 = np_base - np_p1
    l1 = np.dot(v1, v1)
    l2 = np.dot(v2, v1)
    if l2 > l1:
        return p2, 1
    elif l2 < 0 or l1 == 0:
        return p1, 0
    else:
        dt = l2 / l1
        proj = dt * l1 + p1
        return proj.tolist(), dt


def coordinate_descent_projection(pnt, initial_t, value, max_iter=100, tol=1e-4):
    """
    坐标下降法求解点到曲线曲线投影
    @param pnt:
    @param initial_t:
    @param value:
    @param max_iter:
    @param tol:
    @return:
    """
    t = initial_t           # 迭代初值
    step_size = 0.005       # 步长大小
    tmp_pnt1 = pnt
    # 曲线范围
    first = value.first
    last = value.last
    # 坐标下降法优化投影点
    for i in range(max_iter):
        tmp_pnt1 = value.curve.Value(t)
        current_dist1 = pnt.Distance(tmp_pnt1)
        tmp_pnt2 = value.curve.Value(t + step_size)
        current_dist2 = pnt.Distance(tmp_pnt2)
        # 逐步调整 t 来最小化目标函数
        t_new = t - step_size * (current_dist2 - current_dist1)
        # 检查收敛性
        if np.abs(t_new - t) < tol or t_new < first or t_new > last:
            return tmp_pnt1, t

        t = t_new
    return tmp_pnt1, t


def project_to_curve(pnt, value):
    """
    将点投影到线上
    :param pnt:     gp_pnt 点
    :param edge:    TopoDS_edge 边
    :return:        dist距离, proj_point投影点, curve_t投影点在线上的参数坐标
    """
    proj2edge = GeomAPI_ProjectPointOnCurve()  # 点到线投影函数
    try:
        curve3d = value.curve
        first = value.first
        last = value.last
        proj2edge.Init(pnt, curve3d)  # 初始化OCC投影函数
        dist = proj2edge.LowerDistance()
        curve_t = proj2edge.LowerDistanceParameter()  # 投影点在曲线上参数坐标 t
        proj_point = proj2edge.NearestPoint()
        if curve_t > last - geo_eps or curve_t < first + geo_eps:  # 投影点不在定义域内
            pnt1 = curve3d.Value(first)
            pnt2 = curve3d.Value(last)
            dist1 = pnt.Distance(pnt1)
            dist2 = pnt.Distance(pnt2)
            if dist1 < dist2:  # 比较网格点到两个端点的距离
                dist = dist1
                proj_point = pnt1
                curve_t = first
            else:
                dist = dist2
                proj_point = pnt2
                curve_t = last
    except RuntimeError:  # OCC投影失败
        loc_dis_curve = value.discrete_curve  # 曲线简易离散形式
        loc_dis_curve_dt = value.discrete_curve_dt
        base = [pnt.X(), pnt.Y(), pnt.Z()]
        min_dist = float("inf")
        for k in range(len(loc_dis_curve) - 1):
            p1 = loc_dis_curve[k]
            p2 = loc_dis_curve[k + 1]
            # 投影到离散曲线上计算估值
            proj_tmp_xyz, dt_loc = project_to_line(p1, p2, base)
            proj_point_tmp = gp_Pnt(proj_tmp_xyz[0], proj_tmp_xyz[1], proj_tmp_xyz[2])
            dist_tmp = pnt.Distance(proj_point_tmp)
            if dist_tmp < min_dist:
                # 迭代法计算精确值
                curve_t = dt_loc * (loc_dis_curve_dt[k + 1] - loc_dis_curve_dt[k]) + loc_dis_curve_dt[k]
                proj_tmp, new_curve_t = coordinate_descent_projection(pnt, curve_t, value)
                dist = dist_tmp
                proj_point = proj_tmp
                curve_t = new_curve_t
    return dist, proj_point, curve_t


def project_to_face(pnt, value, geo):
    """
    点投影到裁剪曲面上
    :param pnt:     点
    :param value:   面
    :param geo:     几何
    :return:        距离， 投影点坐标
    """
    proj2face = GeomAPI_ProjectPointOnSurf()  # 点到面投影函数
    srf = value.surface
    proj2face.Init(pnt, srf, geo_eps)  # 初始化投影函数
    dist = proj2face.LowerDistance()  # 点与投影点距离
    # 检查uv合法性
    u, v = proj2face.LowerDistanceParameters()  # 投影点在曲面上的uv坐标
    is_legal = True  # 默认认为uv在裁剪曲面内部
    wire_num = value.wire_num  # 面wire数量
    wire_point = value.wire_discrete_point  # 二维wire离散点
    # 在读入几何时将将外环放在wire_point[0]中，只要保证uv点在外环内，在内环外即可，两者方向相反
    # 先检查pnt与外环的关系
    outwire = wire_point[0]
    if not pnt_in_polygon(u, v, outwire):
        is_legal = False
    if wire_num > 2:  # 继续检查内环
        for m in range(1, wire_num):
            current_wire = wire_point[m]
            if pnt_in_polygon(u, v, current_wire):
                is_legal = False
    if is_legal:  # uv点在裁剪区域内，为合法投影
        proj_point = proj2face.NearestPoint()  # 投影点
        current_proj_xyz = [proj_point.X(), proj_point.Y(), proj_point.Z()]
        return dist, current_proj_xyz
    else:  # uv点在裁剪区域外，非法区域，重新投影
        edge_in_face = value.contain_edge_index
        proj_edge_min = float("inf")
        for key in edge_in_face:
            edge = geo.edges[key]
            dist, proj_point, curve_t = project_to_curve(pnt, edge)
            if dist != -1 and dist < proj_edge_min:
                proj_edge_min = dist
                current_proj_xyz = [proj_point.X(), proj_point.Y(), proj_point.Z()]
        return proj_edge_min, current_proj_xyz


def project_mesh_vert_to_geo(verts, geo):
    """
    将网格点投影到距离最近的几何面
    :param verts:           网格顶点坐标 tensor (sum(V_n), 3)
    :param geo:             几何实例
    :param plot_flag:       是否更改投影所属
    :return:                返回投影点坐标     tensor (sum(V_n), 3)
    """
    verts_num = verts.shape[0]  # 网格顶点数量
    proj_point_xyz = []  # 投影后的坐标集合
    for j in range(verts_num):
        if math.isnan(verts[j][0].item()) or math.isnan(verts[j][1].item()) or math.isnan(verts[j][2].item()):
            print("出现nan")
        projection_pnt = gp_Pnt(verts[j][0].item(), verts[j][1].item(), verts[j][2].item())  # 坐标初始化OCC_pnt
        projection_min_dist = float("inf")  # 初始化该点到几何的最短距离
        current_proj_xyz = []  # 初始化投影点坐标

        # 计算点到每个面的投影距离
        for _, value in geo.faces.items():
            face = value.face  # 拓扑面
            # 利用拓扑面构建bnd_box,利用box之间的距离排除该面
            box_dist = pnt_face_box_distance(projection_pnt, face)
            if box_dist > projection_min_dist:  # box之间的最小值都大于投影距离，投影点肯定不在该面上
                continue
            dist, proj_point = project_to_face(projection_pnt, value, geo)
            if dist < projection_min_dist:
                projection_min_dist = dist
                current_proj_xyz = proj_point.copy()
        if projection_min_dist < geo_eps:  # 投影点与网格点距离接近，为同一个点
            current_proj_xyz = [verts[j][0].item(), verts[j][1].item(), verts[j][2].item()]
        proj_point_xyz.append(current_proj_xyz)  # 记录全局最近坐标
    return proj_point_xyz


class MyVertex:
    """
    点类
    """

    def __init__(self, index, vert):
        self.index = index
        self.vert = vert
        self.pnt = BRep_Tool.Pnt(vert)
        self.X = self.pnt.X()
        self.Y = self.pnt.Y()
        self.Z = self.pnt.Z()
        self.belong_curve_index = []
        self.match_mesh_vert = -1

    def GetPntXYZ(self):
        return [self.X, self.Y, self.Z]


class MyEdge:
    """
    边类
    """

    def __init__(self, index, edge, map_vert, num):
        self.index = index
        self.edge = edge
        self.curve, self.first, self.last = BRep_Tool.Curve(edge)
        self.orientation = edge.Orientation()
        self.curve_length = get_edge_length(edge)
        self.belong_face_index = []
        # 利用图的最短路径匹配网格点
        self.MeshVertToEdge = np.zeros(num)  # 构建计算图
        self.sample_num = 0
        self.shortest_path = []
        self.shortest_path_sample = []
        # 找到两个端点
        v1 = TopoDS_Vertex()
        v2 = TopoDS_Vertex()
        topexp.Vertices(edge, v1, v2)
        num1 = map_vert.Find(v1)  # 点1编号
        num2 = map_vert.Find(v2)  # 点2编号
        v1_pnt = BRep_Tool.Pnt(v1)  # 拓扑起点位置
        first_pnt = self.curve.Value(self.first)  # 实际起点位置
        self.end_pnt_index = []
        if first_pnt.IsEqual(v1_pnt, 0.001):
            self.end_pnt_index.append(num1)
            self.end_pnt_index.append(num2)
        else:
            self.end_pnt_index.append(num2)
            self.end_pnt_index.append(num1)
        self.discrete_curve, self.discrete_curve_dt = discrete_curve(self.curve, self.first, self.last, self.orientation)

    def SampleEdge(self, sample_num):
        """
        根据匹配的最短路径上网格点的数量进行采样
        :param sample_num:      最短路径网格点数量
        """
        self.shortest_path_sample = []
        sample_interval = 1 / (sample_num + 1)                      # 单位长度的采样间隔
        current_curve_interval = sample_interval * (self.last - self.first)  # 当前线段上的采样间隔
        sample_loc = self.first + np.fromiter(iter(range(1, sample_num+1)), dtype=int) * current_curve_interval
        for j in range(sample_num):
            pnt = self.curve.Value(sample_loc[j])
            sample_curve_xyz = [pnt.X(), pnt.Y(), pnt.Z()]
            self.shortest_path_sample.append(sample_curve_xyz)

class MyFace:
    """
    面类
    """

    def __init__(self, index, face, map_edge):
        self.index = index
        self.face = face
        self.surface = BRep_Tool.Surface(face)
        self.surface_area = get_sufaces_area(face)
        # 几何边界用于采样
        umin, umax, vmin, vmax = breptools.UVBounds(face)
        self.surfaces_bound = [umin, umax, vmin, vmax]
        self.sample_num = 0
        # 找到面中包含的曲线编号
        self.contain_edge_index = []
        exp_map_edge = TopExp_Explorer(face, TopAbs_EDGE)
        while exp_map_edge.More():
            tmp_edge = topods.Edge(exp_map_edge.Current())
            if map_edge.IsBound(tmp_edge):
                num = map_edge.Find(tmp_edge)
                self.contain_edge_index.append(num)
            exp_map_edge.Next()
        # 获取环的离散表示
        self.wire_num = 1
        self.wire_discrete_point = []
        exp_wire = TopExp_Explorer(face, TopAbs_WIRE)
        out_wire = shapeanalysis.OuterWire(face)  # 外环
        out_wire_uv = discrete_Wire(out_wire, face)  # 离散外环
        self.wire_discrete_point.append(out_wire_uv)  # 储存离散外环
        while exp_wire.More():
            tmp_wire = topods.Wire(exp_wire.Current())
            if not tmp_wire.IsSame(out_wire):
                inside_wire_uv = discrete_Wire(tmp_wire, face)  # 内环外环
                self.wire_num += 1
                self.wire_discrete_point.append(inside_wire_uv)
            exp_wire.Next()



class OccGeo:
    def __init__(self, name, vert_num):
        self.original_geo = read_step_file(name)
        self.mesh_vert_num = vert_num

        self.mesh_edge_len = 0
        self.max_indx_vertex = 0
        self.max_indx_edge = 0
        self.max_indx_face = 0

        self.int_map_vert = TopTools_DataMapOfShapeInteger()  # 点编号
        self.vertexs = {}  # 顶点
        self.num_vertex = 0  # 点数量

        self.int_map_edge = TopTools_DataMapOfShapeInteger()  # 边编号
        self.edges = {}  # 拓扑边
        self.num_edge = 0  # 边数量

        self.faces = {}  # 拓扑面
        self.num_face = 0  # 面数量

        # 获取偏移量与缩放尺寸
        self.get_offset()
        # 几何偏移与缩放
        self.transform_geo()
        # 读取几何
        self.read_geo()
        # 估计工作时的网格尺寸大小
        self.update_surfaces_area()
        self.get_mesh_length()
        print("读取几何完成, 开始几何采样")
        self.delete_small_edge()

        # 计算每条线相接曲面的连续性
        self.no_continuity_edge = np.array([], dtype=np.int64)
        self.no_continuity_vert = np.array([], dtype=np.int64)
        self.get_continuity_face()
        self.mesh_graph = 0
        # 按照网格数量计算采样点
        # 按照网格数量计算采样点画图
        self.mesh_num_in_edge()
        self.mesh_num_in_face()
        self.plot_sample = np.array(self.get_all_pnt_xyz())  # 画图
        _, sample_edge = self.sample_uniform_in_crv(1)
        if len(sample_edge) != 0:
            self.plot_sample = np.append(self.plot_sample, np.array(sample_edge), 0)
        _, sample_surface = self.sample_in_surface(1)
        if len(sample_surface) != 0:
            self.plot_sample = np.append(self.plot_sample, np.array(sample_surface), 0)
        print("几何采样完成")

    def get_all_pnt_xyz(self):
        """
        返回所有几何点的坐标np_list
        :return:
        """
        all_xyz = []
        for key, value in self.vertexs.items():
            current_xyz = value.GetPntXYZ()
            all_xyz.append(current_xyz)
        return np.array(all_xyz)

    def get_offset(self):
        """
        获取原始几何到单位几何的偏移量与比例
        """
        bbox = Bnd_Box()
        bbox.SetGap(geo_eps)
        brepbndlib.Add(self.original_geo, bbox)
        shape_xmin, shape_ymin, shape_zmin, shape_xmax, shape_ymax, shape_zmax = bbox.Get()
        mean_x = (shape_xmax + shape_xmin) / 2
        mean_y = (shape_ymax + shape_ymin) / 2
        mean_z = (shape_zmax + shape_zmin) / 2
        self.center = gp_Vec(-mean_x, -mean_y, -mean_z)
        max_box_len = max([shape_xmax - mean_x, shape_ymax - mean_y, shape_zmax - mean_z])
        self.scale = 1 / (max_box_len * 2)

    def transform_geo(self):
        """
        将原始几何进行移动缩放至单位球内部
        """
        # 设置几何变换矩阵
        transiation_trsf1 = gp_Trsf()
        transiation_trsf1.SetTranslation(self.center)
        scal_base_point = gp_Pnt(0, 0, 0)
        scale_trsf = gp_Trsf()
        scale_trsf.SetScale(scal_base_point, self.scale)
        # 对shape整体进行矩阵变换
        transform_build = BRepBuilderAPI_Transform(self.original_geo, transiation_trsf1)
        tmp_geo = transform_build.Shape()
        transform_build1 = BRepBuilderAPI_Transform(tmp_geo, scale_trsf)
        self.geo = transform_build1.Shape()

    # 读取几何
    def read_geo(self):
        """
        读取几何
        """
        # 点遍历
        geo_map_vertex = TopTools_IndexedMapOfShape()
        topexp.MapShapes(self.geo, TopAbs_VERTEX, geo_map_vertex)
        for i in range(1, geo_map_vertex.Size() + 1):
            tmp_vertex = topods.Vertex(geo_map_vertex.FindKey(i))
            if not self.int_map_vert.IsBound(tmp_vertex):
                self.int_map_vert.Bind(tmp_vertex, self.num_vertex)
                my_pnt = MyVertex(self.num_vertex, tmp_vertex)
                self.vertexs[self.num_vertex] = my_pnt
                self.num_vertex += 1  # 点数量
                self.max_indx_vertex += 1

        # 遍历边
        geo_map_edge = TopTools_IndexedMapOfShape()
        topexp.MapShapes(self.geo, TopAbs_EDGE, geo_map_edge)
        map_edge2vert = []
        for i in range(1, geo_map_edge.Size() + 1):
            tmp_edge = topods.Edge(geo_map_edge.FindKey(i))
            if BRep_Tool.Degenerated(tmp_edge):  # 退化
                continue
            if not self.int_map_edge.IsBound(tmp_edge):
                self.int_map_edge.Bind(tmp_edge, self.num_edge)
                my_edge = MyEdge(self.num_edge, tmp_edge, self.int_map_vert, self.mesh_vert_num)
                map_edge2vert.append([my_edge.end_pnt_index[0], my_edge.end_pnt_index[1]])
                self.edges[self.num_edge] = my_edge  # 边
                self.num_edge += 1  # 边数量
                self.max_indx_edge += 1

        # 遍历面
        face_explorer = TopExp_Explorer(self.geo, TopAbs_FACE)
        map_face2edge = []
        while face_explorer.More():
            tmp_face = topods.Face(face_explorer.Current())
            my_face = MyFace(self.num_face, tmp_face, self.int_map_edge)
            map_face2edge.append(my_face.contain_edge_index)
            self.faces[self.num_face] = my_face
            self.num_face += 1
            face_explorer.Next()

        # 得到点与边的关系
        map_vert2edge = [[] for _ in range(self.num_vertex)]
        for i in range(self.num_edge):
            map_vert2edge[map_edge2vert[i][0]].append(i)
            map_vert2edge[map_edge2vert[i][1]].append(i)
        for i in range(self.num_vertex):
            self.vertexs[i].belong_curve_index = np.array(map_vert2edge[i])

        # 得出点线面的关联
        map_edge2face = [[] for _ in range(self.num_edge)]  # 边所关联的面
        for i in range(self.num_face):
            current_face_edge = map_face2edge[i]  # 当前面中所包含的边的编号
            for j in current_face_edge:
                map_edge2face[j].append(i)  # 得到边与面的关系
        for i in range(self.num_edge):
            self.edges[i].belong_face_index = np.array(map_edge2face[i])

    def update_surfaces_area(self):
        """
        计算所有几何面面积
        """
        self.surfaces_area = []
        for key in self.faces:
            value = self.faces[key]
            surface_area = value.surface_area
            self.surfaces_area.append(surface_area)


    def update_curve_length(self):
        """
        计算所有边的长度
        """
        self.curves_length = []
        for key, value in self.edges.items():
            crv_length = value.curve_length
            self.curves_length.append(crv_length)

    def get_continuity_face(self):
        """
        计算每条边处所属两个面的连接情况
        """

        for key, value in self.edges.items():
            topo_edge = value.edge
            edge_belong_face_indx  = value.belong_face_index
            belong_face_num = len(edge_belong_face_indx)
            if belong_face_num > 1:
                for i in range(belong_face_num):
                    topo_face1 = self.faces[edge_belong_face_indx[i]].face
                    topo_face2 = self.faces[edge_belong_face_indx[(i+1) % belong_face_num]].face
                    continuity = breplib.ContinuityOfFaces(topo_edge, topo_face1, topo_face2, continuity_face_angel)
                    if not continuity:
                        self.no_continuity_edge = np.append(self.no_continuity_edge, key)
                        self.no_continuity_vert = np.append(self.no_continuity_vert, np.array(value.end_pnt_index))
                        self.no_continuity_vert = np.unique(self.no_continuity_vert)
                        continue

    def get_mesh_length(self):
        """
        估算网格尺寸
        """
        vertex_per_len = np.sqrt(self.mesh_vert_num / np.sum(self.surfaces_area))  # 单位长度点数量
        self.mesh_edge_len = 1 / vertex_per_len
        print("根据面积估计工作网格尺寸大小为：", self.mesh_edge_len)

    #
    def mesh_num_in_edge(self):
        """
        根据面积与长度分配网格点数量,返回网格边长度,每一条线分配的点数量,每个面分配的点数量
        """
        surfaces_areas = np.array(self.surfaces_area)
        vertex_per_len = np.sqrt(self.mesh_vert_num / np.sum(surfaces_areas))  # 单位长度点数量
        for key, value in self.edges.items():
            curves_length = get_edge_length(value.edge)
            self.edges[key].sample_num = np.int64(curves_length * vertex_per_len)

    def mesh_num_in_face(self):
        """
        计算面采样点数量
        """
        surfaces_areas = np.array(self.surfaces_area)
        for key, value in self.faces.items():
            area = get_sufaces_area(value.face)
            self.faces[key].sample_num = np.int64((area / np.sum(surfaces_areas)) * self.mesh_vert_num)  # 每个面采样点数量

    def delete_geo(self, index, dim):
        """
        删除编号为index的几何
        :param index:   删除几何的编号
        :param dim:     几何维度
        """
        if dim == 0:
            self.vertexs.pop(index)
            self.num_vertex -= 1
        if dim == 1:
            self.edges.pop(index)
            self.num_edge -= 1
        if dim == 2:
            self.faces.pop(index)
            self.num_face -= 1

    def delete_small_edge(self):
        """
        删除几何中打的短边
        """
        delete_edge = []
        for key, value in self.edges.items():
            topo_edge = value.edge
            edge_length = get_edge_length(topo_edge)
            if edge_length < self.mesh_edge_len * 0.5:
                # 处理线两端点的合并
                v1_index = value.end_pnt_index[0]
                v2_index = value.end_pnt_index[1]
                pnt1 = self.vertexs[v1_index]
                pnt2 = self.vertexs[v2_index]
                self.merge_closed_pnt(pnt1, pnt2)
                # 删除面中包含的该短边的编号
                for i in value.belong_face_index:
                    self.faces[i].contain_edge_index = [x for x in self.faces[i].contain_edge_index if x != value.index]
                # 删除短边
                delete_edge.append(value.index)
        for small in delete_edge:
            self.delete_geo(small, 1)

    def merge_closed_pnt(self, point1, point2):
        """
        合并两个很近的点
        :param point1:      点1
        :param point2:      点2
        """
        pnt1_gp = point1.pnt
        pnt2_gp = point2.pnt
        dist = pnt1_gp.Distance(pnt2_gp)
        if dist < 0.5 * self.mesh_edge_len:
            reserve_pnt_index = point1.index
            delete_pnt_index = point2.index
            delete_pnt_curve = point2.belong_curve_index
            # 修改pnt2相关的边关系
            for i in delete_pnt_curve:
                modify_edge = self.edges[i]
                # 修改pnt2相关的线的端点至pnt1
                if modify_edge.end_pnt_index[0] == delete_pnt_index:
                    new_end_pnt = [reserve_pnt_index, modify_edge.end_pnt_index[1]]
                elif modify_edge.end_pnt_index[1] == delete_pnt_index:
                    new_end_pnt = [modify_edge.end_pnt_index[0], reserve_pnt_index]
                self.edges[i].end_pnt_index = new_end_pnt
            # 增加pnt2的关系至pnt1
            belong_curve = np.unique(np.append(point2.belong_curve_index, point1.belong_curve_index))
            self.vertexs[reserve_pnt_index].belong_curve_index = belong_curve
            self.delete_geo(delete_pnt_index, 0)

    # 根据网格长度对模型中边界进行采样
    def sample_uniform_in_crv(self, multiple):
        """
        geo中所有几何线均匀采样(不包含起点终点)
        :return: 采样点坐标集合 [[[x1,y1,z1], [x2,y2,z2],...], [线2]，...]
        """
        sample_curve = []
        sample_num = 0
        # 找到不连续需要采样的边
        sample_edges_index = []
        sample_curve_num = 0
        for key, value in self.edges.items():
            # 连续条件与采样点数量不为零
            if key in self.no_continuity_edge and self.edges[key].sample_num != 0:
                sample_edges_index.append(key)
                sample_curve_num += 1

        # 开始采样
        for key in sample_edges_index:
            value = self.edges[key]
            curve3D = value.curve
            first = value.first
            last = value.last
            current_sample_num = value.sample_num * multiple  # 采样总数
            sample_num += current_sample_num
            sample_interval = 1 / (current_sample_num + 1)  # 单位长度的采样间隔
            current_curve_interval = sample_interval * (last - first)  # 当前线段上的采样间隔
            sample_loc = first + np.fromiter(iter(range(1, current_sample_num + 1)),
                                             dtype=int) * current_curve_interval
            for j in range(current_sample_num):
                pnt = curve3D.Value(sample_loc[j])
                sample_curve_xyz = [pnt.X(), pnt.Y(), pnt.Z()]
                sample_curve.append(sample_curve_xyz)
        return sample_num, sample_curve

    def sample_in_surface(self, multiple):
        """
        geo中所有几何面随机采样
        :param vertex_num_in_each_surface: 采每一个面样点数量[m, m, m,...]
        :return: 采样点坐标集合 [[x1,y1,z1], [x2,y2,z2],...]
        """
        sample_surface = []
        sample_num = 0
        for key, value in self.faces.items():
            surface = value.surface
            current_surface_wire = value.wire_discrete_point
            wire_num = value.wire_num
            current_surface_sample_num = value.sample_num * multiple
            sample_num += current_surface_sample_num
            bound = value.surfaces_bound
            had_sample_num = 0
            while had_sample_num < current_surface_sample_num:
                random_u = np.random.random() * (bound[1] - bound[0]) + bound[0]
                random_v = np.random.random() * (bound[3] - bound[2]) + bound[2]
                outwire = current_surface_wire[0]
                if not pnt_in_polygon(random_u, random_v, outwire):
                    continue
                if wire_num > 2:  # 继续检查内环
                    for m in range(1, wire_num):
                        current_wire = current_surface_wire[m]
                        if pnt_in_polygon(random_u, random_v, current_wire):
                            continue
                # 检查的uv合法
                sample_pnt_3d = surface.Value(random_u, random_v)
                sample_pnt_3d_xyz = [sample_pnt_3d.X(), sample_pnt_3d.Y(), sample_pnt_3d.Z()]
                sample_surface.append(sample_pnt_3d_xyz)
                had_sample_num += 1
        return sample_num, sample_surface
