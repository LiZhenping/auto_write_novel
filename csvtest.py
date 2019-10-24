# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:09:50 2019

@author: lizhenping
"""




import jieba
from nltk.probability import FreqDist
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式
text = "我来到北京清华大学"

words = jieba.lcut(text)  # 默认是精确模式

#cuted=' '.join(words)

fdist = FreqDist(words)
tops=fdist.most_common(50)
print(tops)


from itertools import chain
b=[["this","is","test"], ["to","make sure"], ["the use","of this","tool"]]
c=list(chain(*b))
print(c)
[1, 2, 3, 5, 8, 7, 8, 9]

index_to_word=[ '高大', '香樟', '...', '』', '爱下', 'txt', '版', '阅读', '下载', '和', '分享', '更', '多', '请', '访问']
a=dict([(w,i)for i,w in enumerate(index_to_word)])





                                      
                                      
import tensorflow as tf
 
embedding = tf.constant(
        [
        [0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]],dtype=tf.float32)

feature_batch = tf.constant([2,3,1,0])

get_embedding1 = tf.nn.embedding_lookup(embedding,feature_batch)

embedding = tf.constant(
    [
        [0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]
    ],dtype=tf.float32)

feature_batch = tf.constant([2,3,1,0])
feature_batch_one_hot = tf.one_hot(feature_batch,depth=4)
get_embedding2 = tf.matmul(feature_batch_one_hot,embedding)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embedding1,embedding2 = sess.run([get_embedding1,get_embedding2])
    print(embedding1)
    print(embedding2)
    
    
import tensorflow as tf

with tf.Graph().as_default(), tf.Session() as sess:
    tf.random.set_random_seed(123)
    logits = tf.log([[1., 1., 1., 1.],
                     [10, 1., 2., 3.]])
    num_samples = 30
    cat = tf.random.categorical(logits, num_samples)
    print(sess.run(cat))


import tensorflow as tf
import numpy as np
 
features, labels = (np.random.sample((6, 3)),  # 模拟6组数据，每组数据3个特征
                    np.random.sample((6, 1)))  # 模拟6组数据，每组数据对应一个标签，注意两者的维数必须匹配
 
print((features, labels))  #  输出下组合的数据
data = tf.data.Dataset.from_tensor_slices((features, labels))
print(data) 
 

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
 
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))
iterator = dataset.make_one_shot_iterator()

for i in range(5):
    print(iterator.get_next())




import tensorflow as tf
def fun(x):
    return x +1
 
 
ds = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))
ds = ds.map(fun)
iterator = ds.make_one_shot_iterator()
for i in range(5):
    print(iterator.get_next())



##################################
    
    
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split    

dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                      
        "b": np.random.uniform(size=(5, 2))
    })


iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

for i in range(4):
    print(iterator.get_next())

# 将数据打乱的混乱程度
dataset = dataset.shuffle(buffer_size=5)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

for i in range(4):
    print(iterator.get_next())
    
# 从数据集中取出数据集的个数
dstaset = dataset.batch(5)
# 指定数据集重复的次数
dataset = dataset.repeat(2)


dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                      
        "b": np.random.uniform(size=(5, 2))
    })
 
 
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

import tensorflow as tf
tf.enable_eager_execution() 
embedding = tf.constant(
        [
        [0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]],dtype=tf.float32)
t=tf.expand_dims(embedding, 0)  
t=tf.expand_dims(embedding, 1) 
t=tf.expand_dims(embedding, 2) 
tf.shape(t)   
tf.print(embedding)
    
    
    
import tensorflow as tf
samples = tf.multinomial(tf.log([[10., 10., 10.,10]]), 5)
print(samples)


import tensorflow as tf
tf.enable_eager_execution() 
a = [[ 2.9548662 , -3.6840122 , -5.1210394 , -3.522386  , -3.1039748 ,
         0.7156184 , -4.868491  , -0.86917454, -5.952623  , -1.777837  ,
        -1.8837416 , -5.605637  , -5.139599  ,  4.3572464 ,  3.4516597 ,
         3.3654604 ,  2.8221784 ,  2.2809572 ,  3.8045962 ,  3.4819393 ,
         4.5289392 ,  4.543256  ,  0.7061087 ,  1.898689  ,  3.3021584 ,
         4.078075  ,  3.8023853 ,  3.8480783 ,  2.5631208 ,  0.11563743,
         2.0673213 ,  4.4374638 ,  4.52647   ,  1.603873  ,  1.99586   ,
         4.904728  , -1.8135399 ,  3.6987545 , -2.0649893 , -0.45059168,
        -0.2764087 , -0.9308289 , -0.7122377 , -3.2698283 , -0.08732652,
        -0.368623  ,  0.7182142 , -0.6845197 , -3.140317  , -2.2951982 ,
        -0.39157423, -0.10450371, -0.53786445, -0.77470714, -1.0274234 ,
        -3.1644518 , -1.348865  ,  0.13062388,  0.18956065, -1.5740802 ,
        -1.4364492 ,  0.6546663 , -5.090876  , -1.0586171 , -3.6439128 ]]
predicted_id = tf.multinomial(a,num_samples = 1)
predicted_id=predicted_id[-1,0]
print(predicted_id)