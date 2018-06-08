from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import mobilenet_v1
import tensorflow as tf

slim = tf.contrib.slim


def mynet_base(inputs, scope=None):
  end_points = {}
  with tf.variable_scope(scope, 'Mynet', [inputs]):
    net_s2 = slim.conv2d(inputs, 16, 3, stride=2)
    net_s2 = slim.separable_conv2d(net_s2, None, 3, 1)
    net_s2 = slim.conv2d(net_s2, 24, 1)
    end_points['last_stride2'] = net_s2

    net_s4 = slim.separable_conv2d(net_s2, None, 3, 1, stride=2)
    net_s4 = slim.conv2d(net_s4, 32, 1)
    net_s4 = slim.separable_conv2d(net_s4, None, 3, 1)
    net_s4 = slim.conv2d(net_s4, 32, 1)
    end_points['last_stride4'] = net_s4

    net_s8_0 = slim.separable_conv2d(net_s4, None, 3, 1, stride=2)
    net_s8_1 = slim.conv2d(net_s8_0, 64, 1)
    net_s8_1 = slim.separable_conv2d(net_s8_1, None, 3, 1)
    net_s8_1 = slim.conv2d(net_s8_1, 16, 1)
    net_s8_concat = tf.concat([net_s8_0, net_s8_1], 3)
    net_s8_2 = slim.conv2d(net_s8_concat, 64, 1)
    net_s8_2 = slim.separable_conv2d(net_s8_2, None, 3, 1)
    net_s8_2 = slim.conv2d(net_s8_2, 16, 1)
    net_s8_concat = tf.concat([net_s8_concat, net_s8_2], 3)
    net_s8 = slim.conv2d(net_s8_concat, 64, 1)
    end_points['last_stride8'] = net_s8

    net_s16_0 = slim.separable_conv2d(net_s8, None, 3, 1, stride=2)
    net_s16_1 = slim.conv2d(net_s16_0, 128, 1)
    net_s16_1 = slim.separable_conv2d(net_s16_1, None, 3, 1)
    net_s16_1 = slim.conv2d(net_s16_1, 32, 1)
    net_s16_concat = tf.concat([net_s16_0, net_s16_1], 3)
    net_s16_2 = slim.conv2d(net_s16_concat, 128, 1)
    net_s16_2 = slim.separable_conv2d(net_s16_2, None, 3, 1)
    net_s16_2 = slim.conv2d(net_s16_2, 32, 1)
    net_s16_concat = tf.concat([net_s16_0, net_s16_2], 3)
    net_s16_3 = slim.conv2d(net_s16_concat, 128, 1)
    net_s16_3 = slim.separable_conv2d(net_s16_3, None, 3, 1)
    net_s16_3 = slim.conv2d(net_s16_3, 32, 1)
    net_s16 = tf.concat([net_s16_concat, net_s16_3], 3)
    net_s16 = slim.conv2d(net_s16, 64, 1)
    end_points['last_stride16'] = net_s16

    net_s32_0 = slim.separable_conv2d(net_s16, None, 3, 1, stride=2)
    net_s32_1 = slim.conv2d(net_s32_0, 192, 1)
    net_s32_1 = slim.separable_conv2d(net_s32_1, None, 3, 1)
    net_s32_1 = slim.conv2d(net_s32_1, 32, 1)
    net_s32_concat = tf.concat([net_s32_0, net_s32_1], 3)
    net_s32_2 = slim.conv2d(net_s32_concat, 192, 1)
    net_s32_2 = slim.separable_conv2d(net_s32_2, None, 3, 1)
    net_s32_2 = slim.conv2d(net_s32_2, 32, 1)
    net_s32 = tf.concat([net_s32_0, net_s32_2], 3)
    end_points['concat_stride32'] = net_s32
    net_s32 = slim.conv2d(net_s32, 192, 1)
    end_points['penultimate_stride32'] = net_s32
    net_s32 = slim.separable_conv2d(net_s32, None, 3, 1)
    net_s32 = slim.conv2d(net_s32, 384, 1)
    end_points['last_stride32'] = net_s32
  return net_s32, end_points


def mynet(inputs, num_classes=1000, is_training=True,
          prediction_fn=tf.contrib.layers.softmax, spatial_squeeze=True,
          reuse=None, scope='Mynet', global_pool=False):
  with tf.variable_scope(scope, 'Mynet', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = mynet_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = mobilenet_v1._reduced_kernel_size_for_small_input(
              net, [7, 7])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a')
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 1024
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      if prediction_fn:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


mynet.default_image_size = 224


def mynet_arg_scope(is_training=True,
                    weight_decay=0.00004,
                    stddev=0.09,
                    regularize_depthwise=False,
                    batch_norm_decay=0.9,
                    batch_norm_epsilon=0.00001):
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc
