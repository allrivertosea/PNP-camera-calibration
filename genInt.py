# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import xml.dom.minidom

# 保存相机内参和畸变系数到XML文件
def save_xml(cam_mtx, dist):
    """
    保存相机内参和畸变系数到XML文件
    参数:
        cam_mtx: 相机内参矩阵
        dist: 畸变系数
    """
    root = ET.Element("opencv_storage")
    camIntrinsicMat = ET.SubElement(root, "camIntrinsicMat")
    camIntrinsicMat_data = ET.SubElement(camIntrinsicMat, "data")
    camIntrinsicMat_data.text = " ".join(map(str, cam_mtx.flatten()))

    distortionCoefficients = ET.SubElement(root, "distortionCoefficients")
    distortionCoefficients_data = ET.SubElement(distortionCoefficients, "data")
    distortionCoefficients_data.text = " ".join(map(str, dist.flatten()))

    xml_string = ET.tostring(root, encoding='utf-8').decode('utf-8')

    dom = xml.dom.minidom.parseString(xml_string)
    formatted_xml = dom.toprettyxml(indent="\t")

    with open("cameraIntrinsic.xml", "w") as f:
        f.write(formatted_xml)
    print("保存完成！")

class MonocularCalibration(object):
    """
    单目相机标定类
    参数:
        data_root: 数据文件夹路径
        board_size: 棋盘格尺寸，默认为 (7, 11)
        img_shape: 图片的分辨率，默认为 (720, 1280)
        suffix: 图片后缀，默认为 "png"
    """
    def __init__(self, data_root,  board_size=(7,11), img_shape=(720, 1280), suffix="png"):
        self.H, self.W = img_shape
        print("\n===> Start Calibration...")
        self.board_size = board_size
        self.Mono_Calibration(corner_h=self.board_size[0], corner_w=self.board_size[1], source_path=data_root, suffix=suffix)

    def Mono_Calibration(self, corner_h, corner_w, source_path, suffix="png"):
        """
        执行单目相机标定
        参数:
            corner_h: 棋盘格的行数
            corner_w: 棋盘格的列数
            source_path: 数据路径
            suffix: 图片文件的后缀
        """
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        objp = np.zeros((corner_h * corner_w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:corner_w, 0:corner_h].T.reshape(-1, 2)
        square_size = 50.0  # 50 mm
        objp = objp * square_size

        obj_points = []
        img_points = []

        images = glob.glob(os.path.join(source_path, "*." + suffix))
        for index, fname in enumerate(images):
            print("=== ", index)
            img = cv2.imread(fname)
            h_, w_, _ = img.shape
            if h_ != self.H or w_ != self.W:
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = self.gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(self.gray, (corner_w, corner_h), None)

            if (corners[0, 0, 0] < corners[-1, 0, 0]):
                print("*" * 5 + " order of {} is inverse! ".format(index) + "*" * 5)
                corners = np.flip(corners, axis=0).copy()

            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                img_points.append(corners2)

                cv2.drawChessboardCorners(img, (corner_w, corner_h), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(50)

        cv2.destroyAllWindows()

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(obj_points, img_points, self.gray.shape[::-1], None, None)

        print("mtx:\n", self.mtx)
        print("dist:\n", self.dist)

        total_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print("total error: ", total_error / len(obj_points))

        return self.mtx, self.dist, self.rvecs, self.tvecs

    def Mono_Undisort_list(self, source_path, suffix="png"):
        """
        去畸变处理多个图像文件
        参数:
            source_path: 图片文件夹路径
            suffix: 图片后缀
        """
        images = glob.glob(os.path.join(source_path, "*." + suffix))
        for fname in images:
            fname = '/'.join(fname.split('\\'))
            print(fname)
            prefix = fname.split('/')[-1].replace(".", "_undistort.")
            img = cv2.imread(fname)
            h_, w_, _ = img.shape
            if h_ != self.H or w_ != self.W:
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            dst = self.Mono_Rectify(img, flag=True)
            print("img: ", img.shape)
            print("dst: ", dst.shape)

            if not os.path.isdir(os.path.join(source_path, "mono_rec")):
                os.mkdir(os.path.join(source_path, "mono_rec"))
            cv2.imwrite(os.path.join(source_path, "mono_rec", prefix), dst)

    def Mono_Rectify(self, image, flag=True):
        """
        对单目图像进行去畸变
        参数:
            image: 输入的畸变图像
            flag: 是否使用内参矩阵进行校正
        返回:
            image_rec: 去畸变后的图像
        """
        image = cv2.resize(image, (self.W, self.H))

        if flag:
            image_rec = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        else:
            h, w = self.H, self.W
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 0, (w, h))
            image_rec = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)

        return image_rec

def main():
    """
    主函数，执行单目相机标定并保存标定结果
    """
    source_path = "calibration_images"
    mono_calibration = MonocularCalibration(data_root=source_path,
                                            board_size=(8, 9),
                                            img_shape=(720, 1280),
                                            suffix="png")
    save_xml(mono_calibration.mtx,mono_calibration.dist)


if __name__ == '__main__':
    main()
