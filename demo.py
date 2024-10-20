# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:09:50 2019
这段代码主要用于文本处理和机器学习，包括以下几个方面：

分词与词频统计：使用 jieba 对中文文本进行分词，并统计词频，输出出现频率最高的词。

列表扁平化：将嵌套列表扁平化，便于后续处理。

词到索引映射：创建词与其索引之间的映射，方便后续使用。

TensorFlow嵌入：演示如何使用 TensorFlow 创建和操作嵌入矩阵，并通过特征批次获取嵌入向量。

数据集创建与处理：展示如何使用 tf.data.Dataset 创建数据集，并对其进行遍历、打乱、批处理和重复操作。

多项式采样：使用 tf.multinomial 从 logits 中进行多项式采样。


@author: lizhenping
"""

import jieba
from nltk.probability import FreqDist

# 分词示例
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 使用全模式进行分词并打印结果
text = "我来到北京清华大学"
words = jieba.lcut(text)  # 使用精确模式进行分词

# 统计词频
fdist = FreqDist(words)  # 计算分词后的词频
tops = fdist.most_common(50)  # 获取出现频率最高的50个词
print(tops)  # 打印词频统计结果

# 扁平化列表示例
from itertools import chain
b = [["this", "is", "test"], ["to", "make sure"], ["the use", "of this", "tool"]]
c = list(chain(*b))  # 将嵌套列表扁平化
print(c)  # 打印扁平化后的列表

# 词到索引的映射
index_to_word = ['高大', '香樟', '...', '』', '爱下', 'txt', '版', '阅读', '下载', '和', '分享', '更多', '请', '访问']
a = dict([(w, i) for i, w in enumerate(index_to_word)])  # 创建词到索引的映射字典

# TensorFlow示例
import tensorflow as tf

# 创建嵌入矩阵
embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32
)

feature_batch = tf.constant([2, 3, 1, 0])  # 定义特征批次
get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)  # 根据索引获取嵌入向量

# 另一种嵌入方法
feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)  # 创建独热编码
get_embedding2 = tf.matmul(feature_batch_one_hot, embedding)  # 计算嵌入向量

# 使用TensorFlow会话执行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化变量
    embedding1, embedding2 = sess.run([get_embedding1, get_embedding2])  # 执行并获取结果
    print(embedding1)  # 打印第一种获取的嵌入向量
    print(embedding2)  # 打印第二种获取的嵌入向量

# 使用logits进行采样
with tf.Graph().as_default(), tf.Session() as sess:
    tf.random.set_random_seed(123)  # 设置随机种子
    logits = tf.log([[1., 1., 1., 1.],  # 定义logits
                     [10, 1., 2., 3.]])
    num_samples = 30  # 设置采样数量
    cat = tf.random.categorical(logits, num_samples)  # 从logits中进行分类采样
    print(sess.run(cat))  # 打印采样结果

# 创建数据集示例
import numpy as np

# 模拟特征和标签数据
features, labels = (
    np.random.sample((6, 3)),  # 模拟6组数据，每组数据3个特征
    np.random.sample((6, 1))    # 模拟6组数据，每组数据对应一个标签
)

print((features, labels))  # 输出数据

# 创建tf.data.Dataset
data = tf.data.Dataset.from_tensor_slices((features, labels))  # 从特征和标签创建数据集
print(data)

# 创建一个简单的tf.data.Dataset并遍历
tf.enable_eager_execution()

dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))  # 创建数据集
iterator = dataset.make_one_shot_iterator()  # 创建迭代器

# 遍历数据集
for i in range(5):
    print(iterator.get_next())  # 打印每个元素

# 使用map函数处理数据
def fun(x):
    return x + 1  # 定义函数，输入加1

ds = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))  # 创建数据集
ds = ds.map(fun)  # 应用map函数
iterator = ds.make_one_shot_iterator()

# 遍历处理后的数据集
for i in range(5):
    print(iterator.get_next())  # 打印每个元素

##################################

# 从TensorFlow 2.x导入
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split    

# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # 特征a
        "b": np.random.uniform(size=(5, 2))  # 特征b
    }
)

iterator = dataset.make_one_shot_iterator()  # 创建迭代器
one_element = iterator.get_next()  # 获取一个元素

# 遍历数据集
for i in range(4):
    print(iterator.get_next())  # 打印每个元素

# 打乱数据集
dataset = dataset.shuffle(buffer_size=5)  # 打乱数据集
iterator = dataset.make_one_shot_iterator()

# 遍历打乱后的数据集
for i in range(4):
    print(iterator.get_next())  # 打印每个元素

# 获取数据集的个数并指定重复次数
dstaset = dataset.batch(5)  # 批处理
dataset = dataset.repeat(2)  # 重复数据集两次

# 再次创建数据集
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # 特征a
        "b": np.random.uniform(size=(5, 2))  # 特征b
    }
)

iterator = dataset.make_one_shot_iterator()  # 创建迭代器
one_element = iterator.get_next()  # 获取一个元素

with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))  # 打印每个元素
    except tf.errors.OutOfRangeError:
        print("end!")  # 捕获超出范围错误并打印结束消息

# 嵌入矩阵扩展示例
embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32
)

t = tf.expand_dims(embedding, 0)  # 在第一维扩展
t = tf.expand_dims(embedding, 1)  # 在第二维扩展
t = tf.expand_dims(embedding, 2)  # 在第三维扩展
tf.shape(t)  # 获取张量形状
tf.print(embedding)  # 打印嵌入矩阵

# multinomial示例
samples = tf.multinomial(tf.log([[10., 10., 10., 10]]), 5)  # 从logits进行采样
print(samples)  # 打印采样结果

# 另一种multinomial示例
a = [[2.9548662, -3.6840122, -5.1210394, -3.522386, -3.1039748,
      0.7156184, -4.868491, -0.86917454, -5.952623, -1.777837,
      -1.8837416, -5.605637, -5.139599, 4.3572464, 3.4516597,
      3.3654604, 2.8221784, 2.2809572, 3.8045962, 3.4819393,
      4.5289392, 4.543256, 0.7061087, 1.898689, 3.3021584,
      4.078075, 3.8023853, 3.8480783, 2.5631208, 0.11563743,
      2.0673213, 4.4374638, 4.52647, 1.603873, 1.99586,
      4.904728, -1.8135399, 3.6987545, -2.0649893, -0.45059168,
      -0.2764087, -0.9308289, -0.7122377, -3.2698283, -0.08732652,
      -0.368623, 0.7182142, -0.6845197, -3.140317, -2.2951982,
      -0.39157423, -0.10450371, -0.53786445, -0.77470714, -1.0274234,
      -3.1644518, -1.348865, 0.13062388, 0.18956065, -1.5740802,
      -1.4364492, 0.6546663, -5.090876, -1.0586171, -3.6439128]]

predicted_id = tf.multinomial(a, num_samples=1)  # 从输入的logits中进行多项式采样
predicted_id = predicted_id[-1, 0]  # 获取最后一个样本的预测ID
print(predicted_id)  # 打印预测的ID
