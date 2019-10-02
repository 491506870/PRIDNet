import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import global_avg_pool


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def unet(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu)
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu)
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu)
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu)
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu)
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu)
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu)
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu)
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu)
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu)
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu)
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu)
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu)
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu)
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu)
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu)
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu)
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu)
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu)
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu)
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu)
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu)
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu)

    conv10 = slim.conv2d(conv9, 1, [1, 1], rate=1, activation_fn=None)
    #out = tf.depth_to_space(conv10, 2)
    return conv10


def feature_encoding(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='fe_conv1')
    conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='fe_conv2')
    conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='fe_conv3')
    conv4 = slim.conv2d(conv3, 32, [3, 3], rate=1, activation_fn=lrelu, scope='fe_conv4')
    conv4 = squeeze_excitation_layer(conv4, 32, 2)
    output = slim.conv2d(conv4, 1, [3, 3], rate=1, activation_fn=lrelu, scope='fe_conv5')

    return output


def avg_pool(feature_map):
    ksize = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 4, 4, 1], [1, 8, 8, 1], [1, 16, 16, 1]]
    pool1 = tf.nn.avg_pool(feature_map, ksize=ksize[0], strides=ksize[0], padding='VALID')
    pool2 = tf.nn.avg_pool(feature_map, ksize=ksize[1], strides=ksize[1], padding='VALID')
    pool3 = tf.nn.avg_pool(feature_map, ksize=ksize[2], strides=ksize[2], padding='VALID')
    pool4 = tf.nn.avg_pool(feature_map, ksize=ksize[3], strides=ksize[3], padding='VALID')
    pool5 = tf.nn.avg_pool(feature_map, ksize=ksize[4], strides=ksize[4], padding='VALID')

    return pool1, pool2, pool3, pool4, pool5


def all_unet(pool1, pool2, pool3, pool4, pool5):
    unet1 = unet(pool1)
    unet2 = unet(pool2)
    unet3 = unet(pool3)
    unet4 = unet(pool4)
    unet5 = unet(pool5)

    return unet1, unet2, unet3, unet4, unet5


def resize_all_image(unet1, unet2, unet3, unet4, unet5):
    resize1 = tf.image.resize_images(images=unet1, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize2 = tf.image.resize_images(images=unet2, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize3 = tf.image.resize_images(images=unet3, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize4 = tf.image.resize_images(images=unet4, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize5 = tf.image.resize_images(images=unet5, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)

    return resize1, resize2, resize3, resize4, resize5


def to_clean_image(feature_map, resize1, resize2, resize3, resize4, resize5):
    concat = tf.concat([feature_map, resize1, resize2, resize3, resize4, resize5], 3)
    sk_conv1 = slim.conv2d(concat, 7, [3, 3], rate=1, activation_fn=lrelu)
    sk_conv2 = slim.conv2d(concat, 7, [5, 5], rate=1, activation_fn=lrelu)
    sk_conv3 = slim.conv2d(concat, 7, [7, 7], rate=1, activation_fn=lrelu)
    sk_out = selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, 4, 7)
    output = slim.conv2d(sk_out, 1, [3, 3], rate=1, activation_fn=None)

    return output


def squeeze_excitation_layer(input_x, out_dim, middle):
    squeeze = global_avg_pool(input_x)
    excitation = tf.layers.dense(squeeze, use_bias=True, units=middle)
    excitation = tf.nn.relu(excitation)
    excitation = tf.layers.dense(excitation, use_bias=True, units=out_dim)
    excitation = tf.nn.sigmoid(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input_x * excitation
    return scale


def selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, middle, out_dim):
    sum_u = sk_conv1 + sk_conv2 + sk_conv3
    squeeze = global_avg_pool(sum_u)
    squeeze = tf.reshape(squeeze, [-1, 1, 1, out_dim])
    z = tf.layers.dense(squeeze, use_bias=True, units=middle)
    z = tf.nn.relu(z)
    a1 = tf.layers.dense(z, use_bias=True, units=out_dim)
    a2 = tf.layers.dense(z, use_bias=True, units=out_dim)
    a3 = tf.layers.dense(z, use_bias=True, units=out_dim)

    before_softmax = tf.concat([a1, a2, a3], 1)
    after_softmax = tf.nn.softmax(before_softmax, dim=1)
    a1 = after_softmax[:, 0, :, :]
    a1 = tf.reshape(a1, [-1, 1, 1, out_dim])
    a2 = after_softmax[:, 1, :, :]
    a2 = tf.reshape(a2, [-1, 1, 1, out_dim])
    a3 = after_softmax[:, 2, :, :]
    a3 = tf.reshape(a3, [-1, 1, 1, out_dim])

    select_1 = sk_conv1 * a1
    select_2 = sk_conv2 * a2
    select_3 = sk_conv3 * a3

    out = select_1 + select_2 + select_3

    return out


def network(in_image):
    feature_map = feature_encoding(in_image)
    feature_map_2 = tf.concat([in_image, feature_map], 3)
    pool1, pool2, pool3, pool4, pool5 = avg_pool(feature_map_2)
    unet1, unet2, unet3, unet4, unet5 = all_unet(pool1, pool2, pool3, pool4, pool5)
    resize1, resize2, resize3, resize4, resize5 = resize_all_image(unet1, unet2, unet3, unet4, unet5)
    out_image = to_clean_image(feature_map_2, resize1, resize2, resize3, resize4, resize5)

    return out_image