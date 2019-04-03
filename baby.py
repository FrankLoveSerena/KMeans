#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
# 使用 K-means 对图像进行聚类，并显示聚类压缩后的图像
# 导入包
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image as image
from skimage import color


# 定义加载图像函数
def load_image(filepath):
    with open(filepath, 'rb') as f:
        img = image.open(f)
        # 获取图像大小
        width, height = img.size
        # 获取像素
        data = []
        for x in range(width):
            for y in range(height):
                c1, c2, c3 = img.getpixel((x, y))
                data.append([c1 / 255, c2 / 255, c3 / 255])
    return data, width, height


# 加载图像
data, width, height = load_image('./baby.jpg')
# 利用k-means对图像进行16聚类
kmeans = KMeans(n_clusters = 16)
kmeans.fit(data)
labels = kmeans.predict(data)
labels = labels.reshape((width, height))
# 创建新图像，用来保存聚类压缩后的结果
new_pic = image.new('RGB', size = (width, height))
for x in range(width):
    for y in range(height):
        c1 = kmeans.cluster_centers_[labels[x, y], 0]
        c2 = kmeans.cluster_centers_[labels[x, y], 1]
        c3 = kmeans.cluster_centers_[labels[x, y], 2]
        new_pic.putpixel((x, y), (int(c1 * 255), int(c2 * 255), int(c3 * 255)))
new_pic.save('compressed_baby.jpg')
# 或者将聚类标识矩阵转化为不同颜色的矩阵，然后直接生成图片
labels_color = (color.label2rgb(labels) * 255).astype(np.uint8)
labels_color = labels_color.transpose(1, 0, 2)
pic_new = image.fromarray(labels_color)
pic_new.save('colored_baby.jpg')
