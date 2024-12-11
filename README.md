# PNP-camera-calibration-demo
Perform intrinsic parameter calibration using a checkerboard pattern and extrinsic parameter calibration using the PnP method, enabling monocular distance measurement.

## 测距效果

测试点世界坐标实际横距1.25米，纵距5米。

![功能测试](https://github.com/allrivertosea/PNP-camera-calibration/blob/main/result.png)


## 环境配置

conda create -n pnp_cc python=3.8 -y
conda activate pnp_cc
pip install -r requirements.txt

## 使用说明

```
python genInt.py   #生成内参
python genExt.py   #生成外参
python PNP_dis.py  #测距验证
```


