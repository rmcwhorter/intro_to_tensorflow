from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
#2019-04-05 19:48:49.047569: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
#AVX2 and FMA are CPU level operations (in basic/machine code) that speed up linear algebra operations. It isn't enabled by default, but I could google it and rebuild tensorflow to utilize it
#This would improve preformance. The code below disables the warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("SOF")
print()
print()

print(tf.__version__)





a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

'''
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()'''
#tensorboard --logdir .


sess = tf.Session()
print(sess.run(total))

print(sess.run({'ab':(a, b), 'total':total}))

print()

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

print()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(z)


my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

print()
print(slices)
print(next_item)

print()

while True:
  try:
    print(next_item)
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break

print()
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

print(sess.run(iterator.initializer))
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break

print()

sess2 = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.Dense(units=10)(x)
z = tf.layers.Dense(units=1)(y)

init = tf.global_variables_initializer()
sess2.run(init)

x_var = [[1, 2, 3],[4, 5, 6],[7,8,9]]
y_var = sess2.run(y, {x: x_var})
z_var = sess2.run(z, {y: y_var})

print("X: ",x_var)
print("Y: ",y_var)
print("Z: ",z_var)





print()
print()
print("EOF")