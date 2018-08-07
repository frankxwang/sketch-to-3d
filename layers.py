import tensorflow as tf

# model parameters
condition_size = 4096
voxel_encoding_size = 4096
voxel_size = 128  # 128*128*128 voxelization
noise_size = 128
momentum_batch_norm = 0.99

# whether or not the model is training or testing (for batch-norm)
training = tf.placeholder(tf.bool, name="training-bool")


def conv3d(inputs, filters, filter_size=3, stride=1, padding="same", name="conv3d"):
    return tf.layers.conv3d(inputs, filters, filter_size, stride, padding, name=name)


def deconv3d(inputs, filters, filter_size=3, stride=1, padding="same", name="deconv3d"):
    return tf.layers.conv3d_transpose(inputs, filters, filter_size, stride, padding, name=name)


def pool3d(inputs, pool_size=2, stride=1, name="pool3d"):
    return tf.layers.max_pooling3d(inputs, pool_size, stride, name=name)


def resnet3d(inputs, filters1, filters2, stride1=1, stride2=1, name="resnet3d"):
    with tf.variable_scope(name):
        conv1 = conv3d(inputs, filters1, stride=stride1, name="conv3d-1")
        lrelu1 = lrelu(conv1, name="lrelu1")
        conv2 = conv3d(lrelu1, filters2, stride=stride2, name="conv3d-2")
        shortcut = conv2 + inputs
        lrelu2 = lrelu(shortcut, name="lrelu2")
        return lrelu2


def conv2d(inputs, filters, filter_size=3, stride=1, padding="same", name="conv2d"):
    return tf.layers.conv2d(inputs, filters, filter_size, stride, padding, name=name)


def pool2d(inputs, pool_size=2, stride=1, name="pool2d"):
    return tf.layers.max_pooling2d(inputs, pool_size, stride, name=name)


def resnet2d(inputs, filters1, filters2, stride1=1, stride2=1, name="resnet2d"):
    with tf.variable_scope(name):
        conv1 = conv2d(inputs, filters1, stride=stride1, name="conv2d-1")
        lrelu1 = lrelu(conv1, name="lrelu1")
        conv2 = conv2d(lrelu1, filters2, stride=stride2, name="conv2d-2")
        shortcut = conv2 + inputs
        lrelu2 = lrelu(shortcut, name="lrelu2")
        return lrelu2


def lrelu(inputs, slope=0.2, use_batch_norm=True, name="lrelu"):
    if use_batch_norm:
        return batch_norm(tf.nn.leaky_relu(inputs, slope, name=name), name+"-batch-norm")
    return tf.nn.leaky_relu(inputs, slope, name=name)


def batch_norm(inputs, name="batch-norm"):
    return tf.layers.batch_normalization(inputs, momentum=momentum_batch_norm, training=training, name=name)


def dense(inputs, out_size, name="dense"):
    return tf.layers.dense(inputs, out_size, name=name)


def flatten(inputs, name="flatten"):
    return tf.layers.flatten(inputs, name=name)


def normalize(inputs, name="norm"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(inputs, -1, name="calc-mean-and-variance")
        return (inputs - mean)/variance


def encode_voxels(voxels):
    with tf.variable_scope("voxel-encoder", reuse=tf.AUTO_REUSE):
        # create voxel encoding
        conv1 = conv3d(voxels, 32, name="conv3d-1")
        lrelu1 = lrelu(conv1, name="lrelu-1")

        conv2 = conv3d(lrelu1, 64, stride=2, padding="same", name="conv3d-2")
        # pool2 = pool3d(conv2, name="pool3d-2")
        lrelu2 = lrelu(conv2, name="lrelu-2")

        resnet3 = resnet3d(lrelu2, 64, 64, name="resnet3d-3")

        conv4 = conv3d(resnet3, 128, stride=2, padding="same", name="conv3d-4")
        lrelu4 = lrelu(conv4, name="lrelu-4")

        resnet5 = resnet3d(lrelu4, 128, 128, name="resnet3d-5")

        conv6 = conv3d(resnet5, 256, stride=2, padding="same", name="conv3d-6")
        lrelu6 = lrelu(conv6, name="lrelu-6")

        resnet6 = resnet3d(lrelu6, 256, 256, name="resnet3d-6")

        conv7 = conv3d(resnet6, 256, stride=2, padding="same", name="conv3d-7")
        lrelu7 = lrelu(conv7, name="lrelu-7")

        resnet6 = resnet3d(lrelu7, 256, 256, name="resnet3d-8")

        flattened = flatten(resnet6, name="flatten-1")

        dense1 = dense(flattened, 4096, name="dense-1")

        dense2 = dense(dense1, voxel_encoding_size, name="dense-2")

    return dense2


def encode_image(image):
    with tf.variable_scope("image-encoder", reuse=tf.AUTO_REUSE):
        # create image encoding
        conv1 = conv2d(image, 32, padding="same", name="conv2d-1")
        lrelu1 = lrelu(conv1, name="lrelu-1")

        conv2 = conv2d(lrelu1, 64, stride=2, padding="same", name="conv2d-2")
        # pool2 = pool2d(conv2, name="pool2d-2")
        lrelu2 = lrelu(conv2, name="lrelu-2")

        resnet3 = resnet2d(lrelu2, 64, 64, name="resnet2d-3")

        conv4 = conv2d(resnet3, 128, stride=2, padding="same", name="conv2d-4")
        lrelu4 = lrelu(conv4, name="lrelu-4")

        resnet5 = resnet2d(lrelu4, 128, 128, name="resnet2d-5")

        conv6 = conv2d(resnet5, 256, stride=2, padding="same", name="conv2d-6")
        lrelu6 = lrelu(conv6, name="lrelu-6")

        resnet6 = resnet2d(lrelu6, 256, 256, name="resnet2d-6")

        conv7 = conv2d(resnet6, 256, stride=2, padding="same", name="conv2d-7")
        lrelu7 = lrelu(conv7, name="lrelu-7")

        resnet6 = resnet2d(lrelu7, 256, 256, name="resnet2d-8")

        flattened = flatten(resnet6, name="flatten-1")

        dense1 = dense(flattened, 4096, name="dense-1")

        dense2 = dense(dense1, condition_size, name="dense-2")

        normalized = normalize(dense2)

    return normalized


def discriminator(voxels, condition, enconding_pre_comp=False):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        if not enconding_pre_comp:
            voxels_encoding = encode_voxels(voxels)
        else:
            voxels_encoding = voxels
        # print(voxels_encoding)
        # print(condition)
        encoding = tf.concat([voxels_encoding, condition], 1, name="concat-1")

        dense1 = dense(encoding, 4096, name="dense-1")
        lrelu1 = lrelu(dense1, name="lrelu-1")

        dense2 = dense(lrelu1, 1024, name="dense-2")
        lrelu2 = lrelu(dense2, name="lrelu-2")

        dense3 = dense(lrelu2, 1, name="output")

    return dense3


def generator(noise, condition):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        encoding = tf.concat([noise, condition], 1, name="concat-1")
        dense1 = dense(encoding, 4096, name="dense-1")

        # 1x1x1 voxels noise
        encoding3d = tf.reshape(dense1, [-1, 1, 1, 1, 4096], name="reshape3d")

        # 2x2x2 voxels
        deconv1 = deconv3d(encoding3d, 2048, stride=2, name="deconv3d-1")
        lrelu1 = lrelu(deconv1, name="lrelu-1")

        # 4x4x4 voxels
        deconv2 = deconv3d(lrelu1, 512, stride=2, name="deconv3d-2")
        lrelu2 = lrelu(deconv2, name="lrelu-2")

        deconv3 = deconv3d(lrelu2, 256, stride=2, name="deconv3d-3")
        lrelu3 = lrelu(deconv3, name="lrelu-3")

        deconv4 = deconv3d(lrelu3, 128, stride=2, name="deconv3d-4")
        lrelu4 = lrelu(deconv4, name="lrelu-4")

        deconv5 = deconv3d(lrelu4, 64, stride=2, name="deconv3d-5")
        lrelu5 = lrelu(deconv5, name="lrelu-5")

        deconv6 = deconv3d(lrelu5, 32, stride=2, name="deconv3d-6")
        lrelu6 = lrelu(deconv6, name="lrelu-6")

        # 128x128x128 voxels output
        deconv7 = deconv3d(lrelu6, 1, stride=2, name="deconv3d-7")
        sigmoid = tf.sigmoid(deconv7, name="output")

    return sigmoid
