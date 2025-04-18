from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Shell
import math


def make_face(p1, p2, p3):
    """
    创建一个由三个点定义的平面
    :return:   face
    """
    polygon = BRepBuilderAPI_MakePolygon()
    polygon.Add(p1)
    polygon.Add(p2)
    polygon.Add(p3)
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()

# 定义标准钻石切割形状
crown_height = 3  # 冠部高度
neck_height = 2.5 # 颈部高度
pavilion_depth = 4.5  # 亭部深度
girdle_radius = 5  # 腰部半径

# 六边形桌面点
num_table = 6
table_radius = girdle_radius * 0.5
table_points = [gp_Pnt(table_radius * math.cos(i * 2 * math.pi / num_table),
                        table_radius * math.sin(i * 2 * math.pi / num_table),
                        crown_height) for i in range(num_table)]

neck_radius = girdle_radius * 0.6
neck_points = [gp_Pnt(neck_radius * math.cos(i * 2 * math.pi / num_table + math.pi / num_table),
                      neck_radius * math.sin(i * 2 * math.pi / num_table + math.pi / num_table),
                      neck_height) for i in range(num_table)]

# 亭尖
culet = gp_Pnt(0, 0, -pavilion_depth)

# 腰部点
num_girdle = 12
girdle_points = [gp_Pnt(girdle_radius * math.cos(i * 2 * math.pi / num_girdle),
                 girdle_radius * math.sin(i * 2 * math.pi / num_girdle), 0)
                 for i in range(num_girdle)]

# 亭部主面顶点（下方）
pavilion_points = [gp_Pnt(girdle_radius * 0.5 * math.cos(i * 2 * math.pi / num_girdle),
                   girdle_radius * 0.5 * math.sin(i * 2 * math.pi / num_girdle),
                   -pavilion_depth * 0.5) for i in range(num_girdle)]

# 构造面
faces = set()

# 冠部面（六边形桌面）
table_face = BRepBuilderAPI_MakePolygon()
for i in range(num_table):
    table_face.Add(table_points[i])
table_face.Close()
faces.add(BRepBuilderAPI_MakeFace(table_face.Wire()).Face())

# 冠部到颈部的连接面
for i in range(num_table):
    faces.add(make_face(table_points[i], neck_points[i], table_points[(i + 1) % num_table]))
    # faces.add(make_face(neck_points[i], table_points[(i + 1) % num_table], neck_points[(i + 1) % num_table]))

# 腰部面1
for i in range(num_table):
    faces.add(make_face(neck_points[i], girdle_points[2 * i], girdle_points[2 * i + 1]))
    faces.add(make_face(neck_points[i], girdle_points[2 * i + 1], girdle_points[(2 * i + 2) % num_girdle]))
# 腰部面2
for i in range(num_table):
    faces.add(make_face(table_points[i], girdle_points[2 * i], neck_points[i]))
    faces.add(make_face(table_points[i], girdle_points[2 * i], neck_points[(i + num_table - 1) % num_table]))
# 腰部到亭部
for i in range(num_girdle):
    pavilion_face = BRepBuilderAPI_MakePolygon()
    pavilion_face.Add(girdle_points[i])
    pavilion_face.Add(pavilion_points[i])
    pavilion_face.Add(pavilion_points[(i + 1) % num_girdle])
    pavilion_face.Add(girdle_points[(i + 1) % num_girdle])
    pavilion_face.Close()
    faces.add(BRepBuilderAPI_MakeFace(pavilion_face.Wire()).Face())

# 亭部收敛到亭尖
for i in range(num_girdle):
    faces.add(make_face(pavilion_points[i], pavilion_points[(i + 1) % num_girdle], culet))


# 组合面为封闭 shell
sewing = BRepOffsetAPI_Sewing()
for face in faces:
    if face:
        sewing.Add(face)
sewing.Perform()

# 显式转换为 Shell
shell = sewing.SewedShape()

# 导出为 STEP 文件
def export_step(shape, filename="diamond.step"):
    step_writer = STEPControl_Writer()
    step_writer.Transfer(shape, STEPControl_AsIs)
    status = step_writer.Write(filename)
    if status == IFSelect_RetDone:
        print(f"STEP file '{filename}' successfully written.")
    else:
        print("Failed to write STEP file.")

export_step(shell)
