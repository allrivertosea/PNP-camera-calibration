import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import configparser

img = None

# 读取校准参数文件
def read_calibration_parameters():
    """
    从配置文件 'cfg.ini' 中读取相机标定参数，包括相机内参、畸变系数和RT矩阵。
    返回:
        cameraMatrix1: 相机内参矩阵
        distCoeffs1: 畸变系数
        RT_: RT矩阵
    """
    config = configparser.ConfigParser()
    with open('cfg.ini', 'r', encoding='utf-8') as file:
        config.read_file(file)

    cameraMatrix1_values = list(map(float, config['cameraMatrix1']['value'].split(',')))
    distCoeffs1_values = list(map(float, config['distCoeffs1']['value'].split(',')))
    RT_values = list(map(float, config['RT_']['value'].split(',')))

    cameraMatrix1 = np.array(cameraMatrix1_values).reshape((3, 3))
    distCoeffs1 = np.array(distCoeffs1_values)
    RT_ = np.array(RT_values).reshape((3, 4))

    return cameraMatrix1, distCoeffs1, RT_

# 根据图像点和校准参数计算世界坐标
def getWorldPoint(imagePoint, cameraMatrix1, RT_):
    """
    将图像坐标转换为世界坐标。
    参数:
        imagePoint: 输入的图像坐标
        cameraMatrix1: 相机内参矩阵
        RT_: RT矩阵
    返回:
        worldPoint: 计算得到的世界坐标
    """
    rvecM1 = RT_[:, :3]
    tvec1 = RT_[:, 3:]

    if imagePoint.shape != (3, 1):
        raise ValueError("imagePoint must be a 3x1 vector.")

    try:
        inv_cameraMatrix = np.linalg.inv(cameraMatrix1)
        inv_rvecM1 = np.linalg.inv(rvecM1)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed. Check if cameraMatrix1 and rvecM1 are invertible.")

    tempMat = inv_rvecM1 @ (inv_cameraMatrix @ imagePoint)
    tempMat2 = inv_rvecM1 @ tvec1
    zConst = 0
    s = zConst + tempMat2[2, 0]
    s /= tempMat[2, 0]
    worldPoint = inv_rvecM1 @ (inv_cameraMatrix @ (s * imagePoint) - tvec1)
    print(worldPoint)

    return worldPoint

# 鼠标点击事件处理函数，计算并显示距离信息
def on_click(event, panel, cameraMatrix1, RT_):
    """
    鼠标点击事件，获取点击位置的像素坐标并计算对应的世界坐标，然后显示距离信息。
    参数:
        event: 鼠标点击事件
        panel: 显示图像的Tkinter面板
        cameraMatrix1: 相机内参矩阵
        RT_: RT矩阵
    """
    global img
    x, y = event.x, event.y
    imagePoint = np.array([[x], [y], [1.0]])
    worldPoint = getWorldPoint(imagePoint, cameraMatrix1, RT_)
    img_h_dist = worldPoint[0, 0] / 1000.0  # 水平距离，单位米
    img_dist = worldPoint[1, 0] / 1000.0   # 垂直距离，单位米

    # 图像处理部分
    if isinstance(img, Image.Image):
        try:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            messagebox.showerror("错误", f"图像转换错误: {str(e)}")
            return
    else:
        messagebox.showerror("错误", "无效的图像格式")
        return

    # 在图像上标注点击坐标
    cv2.putText(img_cv, f"({x},{y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # 显示距离信息
    messagebox.showinfo("距离信息",
                        f"像素坐标: ({x}, {y})\n水平距离: {img_h_dist:.2f} 米\n垂直距离: {img_dist:.2f} 米")

# 加载图像
def load_image(panel):
    """
    弹出文件选择框，选择并加载图像。
    参数:
        panel: 用于显示图像的Tkinter面板
    """
    global img
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_cv)
        img_tk = ImageTk.PhotoImage(img)

        panel.config(image=img_tk)
        panel.image = img_tk

    return img

# 主函数
def main():
    """
    主函数，初始化GUI界面并执行相应操作。
    """
    # 读取校准参数
    cameraMatrix1, distCoeffs1, RT_ = read_calibration_parameters()

    # GUI设置
    root = tk.Tk()
    root.title("距离测量工具")

    panel = tk.Label(root)
    panel.pack()

    # 绑定点击事件
    panel.bind("<Button-1>", lambda event: on_click(event, panel, cameraMatrix1, RT_))

    # 加载图像按钮
    btn = tk.Button(root, text="加载图像", command=lambda: load_image(panel))
    btn.pack()

    # 启动Tkinter事件循环
    root.mainloop()

# 启动程序
if __name__ == "__main__":
    main()
