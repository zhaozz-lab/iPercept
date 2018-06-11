import tensorflow as tf

saver = tf.train.import_meta_graph('outputs/DenseNetOriginal.meta')

with tf.Session() as sess:
    saver.restore(sess, 'test')

total_parameters = 0
for variable in tf.trainable_variables():
    total_parameters += 1
print(total_parameters)