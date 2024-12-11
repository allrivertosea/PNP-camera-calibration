import cv2
import numpy as np
import math
from lxml import etree
import sys


# 从XML文件中读取相机内参矩阵
def read_camIntrinsicMat(xml_file):
    """
    从XML文件中读取相机内参矩阵
    参数:
        xml_file: XML文件路径
    返回:
        cam_intrinsic_mat: 相机内参矩阵
    """
    try:
        doc = etree.parse(xml_file)
    except Exception as e:
        print(f"Error: could not parse file {xml_file}, {e}", file=sys.stderr)
        sys.exit(-1)

    root = doc.getroot()
    cam_intrinsic_mat = np.zeros((3, 3), dtype=np.float64)

    for element in root.iter('camIntrinsicMat'):
        data = element.find('data').text
        values = list(map(float, data.split()))
        cam_intrinsic_mat = np.array(values).reshape((3, 3))

    return cam_intrinsic_mat


# 从XML文件中读取畸变系数
def read_distCoeffs(xml_file):
    """
    从XML文件中读取畸变系数
    参数:
        xml_file: XML文件路径
    返回:
        distCoeffs: 畸变系数
    """
    try:
        doc = etree.parse(xml_file)
    except Exception as e:
        print(f"Error: could not parse file {xml_file}, {e}", file=sys.stderr)
        sys.exit(-1)

    root = doc.getroot()
    distCoeffs = np.zeros((5, 1), dtype=np.float64)

    for element in root.iter('distortionCoefficients'):
        data = element.find('data').text
        values = list(map(float, data.split()))
        distCoeffs = np.array(values).reshape((5, 1))

    return distCoeffs


# 从文件中读取2D图像点和3D世界坐标点
def readPoints(file):
    """
    从文件中读取图像点和对应的世界坐标点
    参数:
        file: 点文件路径
    返回:
        boxPoints: 2D图像点
        worldBoxPoints: 3D世界坐标点
    """
    boxPoints = []
    worldBoxPoints = []

    try:
        with open(file, 'r') as infile:
            lines = infile.readlines()
    except Exception as e:
        print(f"Error: could not open or read file {file}, {e}", file=sys.stderr)
        sys.exit(-1)

    if len(lines) != 8:
        print(f"Error: Expected 8 lines in the file, but got {len(lines)}", file=sys.stderr)
        sys.exit(-1)

    try:
        for i in range(4):
            x, y = map(float, lines[i].strip().split())
            boxPoints.append((x, y))
        for i in range(4, 8):
            x, y, z = map(float, lines[i].strip().split())
            worldBoxPoints.append((x, y, z))
    except ValueError as e:
        print(f"Error: Could not convert data to float. Check file format. {e}", file=sys.stderr)
        sys.exit(-1)

    return np.array(boxPoints, dtype=np.float32), np.array(worldBoxPoints, dtype=np.float32)


# 将矩阵写入INI文件
def writeMatToIni(file, name, value, mat):
    """
    将矩阵写入INI文件
    参数:
        file: INI文件对象
        name: 矩阵名称
        value: 矩阵值的描述
        mat: 矩阵数据
    """
    file.write(f"\n[{name}]\n")
    file.write(f"{value}")
    for i in range(mat.shape[0]):
        line = ", ".join(map(str, mat[i]))
        if i < mat.shape[0] - 1:
            line += ","
        file.write(line)


# 主函数
def main():
    """
    主函数，读取相机标定数据，执行PNP算法，计算旋转矩阵和平移向量，最终保存到INI文件。
    """
    xml_file = "cameraIntrinsic.xml"
    points_file = "PNP.txt"
    ini_file = "cfg.ini"

    # 读取相机内参和畸变系数
    cameraMatrix1 = read_camIntrinsicMat(xml_file)
    distCoeffs1 = read_distCoeffs(xml_file)

    # 读取2D图像点和3D世界坐标点
    boxPoints, worldBoxPoints = readPoints(points_file)

    # 使用OpenCV的solvePnP算法计算旋转向量和平移向量
    retval, rvec1, tvec1 = cv2.solvePnP(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1)
    rvecM1, _ = cv2.Rodrigues(rvec1)
    print(f"recM1: {rvecM1}\ntvec1: {tvec1}")

    # 计算欧拉角
    thetaZ = math.atan2(rvecM1[1, 0], rvecM1[0, 0]) * 180 / math.pi
    thetaY = math.atan2(-rvecM1[2, 0], math.sqrt(rvecM1[2, 1] ** 2 + rvecM1[2, 2] ** 2)) * 180 / math.pi
    thetaX = math.atan2(rvecM1[2, 1], rvecM1[2, 2]) * 180 / math.pi
    print(f"theta x: {thetaX}\ntheta Y: {thetaY}\ntheta Z: {thetaZ}")

    # 拼接RT矩阵
    RT_ = np.hstack((rvecM1, tvec1))
    print(RT_)

    # 将结果写入INI文件
    with open(ini_file, 'a') as iniFile:
        writeMatToIni(iniFile, "cameraMatrix1", "value=", cameraMatrix1)
        writeMatToIni(iniFile, "distCoeffs1", "value=", distCoeffs1)
        writeMatToIni(iniFile, "RT_", "value=", RT_)


# 启动程序
if __name__ == "__main__":
    main()
