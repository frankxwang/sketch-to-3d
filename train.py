# from model import *
from layers import *
import tensorflow as tf
import h5py
import numpy as np


# training parameters
batch_size = 64
learning_rate = 0.01
epochs = 1000000000000

# real voxels
voxels = tf.placeholder(tf.float32, (None, voxel_size, voxel_size, voxel_size, 1), name="voxels")
# real images
images = tf.placeholder(tf.float32, (None, 512, 512, 1), name="images")
# noise to generate voxels
noise_voxels = tf.placeholder(tf.float32, (None, noise_size))
# fake encodings of images
noise_encoding = tf.placeholder(tf.float32, (None, condition_size))

# generate real encodings of images
encoding = encode_image(images)

# create generator
gen = generator(noise_voxels, noise_encoding)
print(gen)

# predictions for real data
real_pred = discriminator(voxels, encoding)
# predictions for fake (generated) data
fake_pred = discriminator(gen, noise_encoding)

# get loss (wasserstein loss)
loss = tf.reduce_mean(fake_pred, name="fake_pred") - tf.reduce_mean(real_pred, name="real_pred")

# optimizer
train = tf.train.AdamOptimizer(learning_rate, name="adam-optimizer").minimize(loss, name="minimize")

saver = tf.train.Saver()

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    sess.run(init)

    data_voxels = h5py.File("data/voxels.h5", mode="r")["data"]
    data_images = h5py.File("data/sketches.h5", mode="r")["data"]

    print(data_voxels.shape)
    print(data_images.shape)

    data_num = data_voxels.shape[0]

    minimum = 100000000
    prev_epoch = 0

    for i in range(epochs):
        pos1 = i*batch_size % data_num
        if pos1+batch_size >= data_num:
            pos1 = 0

        feed_dict = {voxels: np.expand_dims(data_voxels[pos1:pos1+batch_size], 4),
                     images: np.expand_dims(np.transpose(data_images[pos1:pos1+batch_size], (1, 0, 2, 3))[0], 3),
                     noise_voxels: np.random.randn(batch_size, noise_size),
                     noise_encoding: np.random.randn(batch_size, condition_size),
                     training: True}
        curr_loss, _ = sess.run([loss, train], feed_dict=feed_dict)
        print(curr_loss)
        if curr_loss <= minimum and i-prev_epoch > 10:
            saver.save(sess, "model", global_step=i)

    # data_voxels = tables.open_file("data/voxels.h5", mode="r", driver="H5FD_CORE")
    # data_images = tables.open_file("data/sketches.h5", mode="r", driver="H5FD_CORE")
    # data_voxels.get_node()

    # train_writer = tf.summary.FileWriter('./logs/test ', sess.graph)
