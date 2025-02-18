import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_pointcloud(mesh, geo, plot_line_flag):
    """
    绘制图形，包括网格与几何采样点
    :param mesh:    网格
    :param geo:     几何
    :param plot_line_flag:      是否绘制网格线
    :return:
    """
    verts = mesh.verts_packed()
    x, y, z = verts.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    np_point = np.array(geo.plot_sample)
    x1 = np_point[:, 0]
    y1 = np_point[:, 1]
    z1 = np_point[:, 2]
    ax.scatter3D(x1, y1, z1, color="red", zorder=1, s=5)
    ax.scatter3D(x, y, z, color="blue")
    if plot_line_flag:
        edges_packed = mesh.edges_packed()
        verts_edges = verts[edges_packed]
        for i in verts_edges:
            xx, yy, zz = i.clone().detach().unbind(1)
            ax.plot(xx, yy, zz,  color="orange")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("title")
    ax.view_init(190, 30)
    plt.show()


def plot_point(point):
    np_point = np.array(point)
    x = np_point[:, 0]
    y = np_point[:, 1]
    z = np_point[:, 2]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(190, 30)
    plt.show()


def plot_two_point(points1, points2):
    np_point1 = np.array(points1)
    x1 = np_point1[:, 0]
    y1 = np_point1[:, 1]
    z1 = np_point1[:, 2]
    np_point2 = np.array(points2)
    x2 = np_point2[:, 0]
    y2 = np_point2[:, 1]
    z2 = np_point2[:, 2]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x1, y1, z1)
    ax.scatter3D(x2, y2, z2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(190, 30)
    plt.show()

def plot_final_loss(surface_losses, match_edge_losses, angle_losses, edge_length_losses, normal_losses, laplacian_losses):
    fig1 = plt.figure(figsize=(13, 5))
    ax1 = fig1.gca()
    ax1.plot(surface_losses, label="surface_losses")
    ax1.plot(match_edge_losses, label="match_edge_losses")
    ax1.plot(angle_losses, label="angle losse")
    ax1.plot(edge_length_losses, label="edge_length_losses")
    ax1.plot(normal_losses, label="normal loss")
    ax1.plot(laplacian_losses, label="laplacian loss")
    ax1.legend(fontsize="16")
    ax1.set_xlabel("Iteration", fontsize="16")
    ax1.set_ylabel("Loss", fontsize="16")
    ax1.set_title("Loss vs iterations", fontsize="16")
    plt.show()

def plot_mesh(mesh,  faces):
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = faces.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        # 获取面片的顶点坐标
        print(1)
        triangle = verts[face]

        # 创建面片的三角形对象并加入图形中
        poly3d = Poly3DCollection([triangle], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
        ax.add_collection3d(poly3d)

    # 设置坐标轴的标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()