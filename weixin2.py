#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
# 导入包
from sklearn.cluster import KMeans
from PIL import Image as image
from sklearn.preprocessing import MinMaxScaler

# 加载图像
img = image.open('./weixin.png')
# 将4通道的png图像转化成jpg格式的3通道
img = img.convert('RGB')
# 得到图像尺寸
width, height = img.size
# 获取像素数据
data = []
for x in range(width):
    for y in range(height):
        # 得到点(x, y)的三个通道值
        c1, c2, c3 = img.getpixel((x, y))
        data.append([c1, c2, c3])
# 将像素数据规范化
mm = MinMaxScaler()
data = mm.fit_transform(data)
# 利用K-Means将图像聚类为2部分
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data)
label = kmeans.predict(data)
# 将图像聚类结果转化为图像尺寸的矩阵
label = label.reshape((width, height))
# 创建新图像并写入聚类结果
new_pic = image.new(mode = 'L', size = (width, height))
for x in range(width):
    for y in range(height):
        new_pic.putpixel((x, y), int(256 / (label[x, y] + 1)) - 1)
new_pic.save('test2.jpg', format = 'jpeg')
