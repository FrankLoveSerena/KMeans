#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
# 导入包
import numpy as np
from skimage import color
from sklearn.cluster import KMeans
from PIL import Image as image
from sklearn.preprocessing import MinMaxScaler

# 加载图片
img = image.open('./weixin.png')
# 将4通道的png图像转化成jpg格式的3通道
img = img.convert('RGB')
# 获取图片尺寸
width, height = img.size
# 获取图片像素
data = []
for x in range(width):
    for y in range(height):
        c1, c2, c3 = img.getpixel((x, y))
        data.append([c1, c2, c3])
# 将特征集规范化
mm = MinMaxScaler()
data = mm.fit_transform(data)
# 创建k-means模型
kmeans = KMeans(n_clusters = 16)
kmeans.fit(data)
label = kmeans.predict(data).reshape((width, height))
# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label) * 255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2)
# 生成图片
images = image.fromarray(label_color)
images.save('test16.jpg')
