from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

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

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()

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

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

print(x)
print(y)

init = tf.global_variables_initializer()
sess.run(init)




print()
print()
print("EOF")