import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import context_manager
from object_detection.utils import ops
from nets import mynet

slim = tf.contrib.slim


def resize_neareast_neighbor_nhwc_using_tile(x):
  output_shape = [dim * multiple for dim, multiple in zip(x.shape,
                                                          [1, 2, 2, 1])]
  # TODO(pw): replace following operations with tf.image.resize* or tf.tile once
  # their tflite operations are implemented.
  output = tf.concat([x, x], axis=3)
  output = tf.concat([output, output], axis=2)
  output = tf.reshape(
      output, (tf.shape(x) * [1, 2, 2, 1] if not x.shape.is_fully_defined() else
               output_shape))
  output.set_shape(output_shape)
  return output


class MyFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):

  def preprocess(self, resized_inputs):
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    with tf.variable_scope('Mynet', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(mynet.mynet_arg_scope(is_training=None)):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams
              else context_manager.IdentityContextManager()):
          _, image_features = mynet.mynet_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              scope=scope)
    with slim.arg_scope(self._conv_hyperparams_fn()):
      return self._generate_feature_maps(image_features)

  def _generate_feature_maps(self, image_features):
    feature_map_16 = image_features['last_stride16']
    feature_map_32 = image_features['last_stride32']

    net = slim.conv2d(feature_map_32, 64, 1)
    net = resize_neareast_neighbor_nhwc_using_tile(net)
    net = tf.concat([net, feature_map_16], 3)
    net = slim.conv2d(net, 64, 1)
    feature_map_16 = slim.separable_conv2d(net, None, 3, 1)
    feature_map_16 = slim.conv2d(feature_map_16, 128, 1)

    net = slim.conv2d(net, 128, 1)
    net = slim.separable_conv2d(net, None, 3, 1, stride=2)
    feature_map_32 = slim.conv2d(net, 128, 1)
    feature_map_32 = slim.separable_conv2d(feature_map_32, None, 3, 1)
    feature_map_32 = slim.conv2d(feature_map_32, 128, 1)

    net = slim.conv2d(net, 128, 1)
    net = slim.separable_conv2d(net, None, 3, 1, stride=2)
    feature_map_64 = slim.conv2d(net, 128, 1)
    feature_map_64 = slim.separable_conv2d(feature_map_64, None, 3, 1)
    feature_map_64 = slim.conv2d(feature_map_64, 128, 1)

    net = slim.conv2d(net, 128, 1)
    net = slim.separable_conv2d(net, None, 3, 1, stride=2)
    feature_map_128 = slim.conv2d(net, 64, 1)
    feature_map_128 = slim.separable_conv2d(feature_map_128, None, 3, 1)
    feature_map_128 = slim.conv2d(feature_map_128, 64, 1)

    net = slim.conv2d(net, 64, 1)
    net = slim.separable_conv2d(net, None, 3, 1, stride=2)
    feature_map_256 = slim.conv2d(net, 64, 1)

    return [feature_map_16, feature_map_32, feature_map_64, feature_map_128,
            feature_map_256]
