import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import context_manager
from object_detection.utils import ops
from nets import mynet

slim = tf.contrib.slim


def resize_neareast_neighbor_nhwc_using_tile(x):
  return tf.reshape(
      tf.tile(tf.expand_dims(x, axis=-1),
              multiples=[1, 1, 2, 2, 1]),
      (tf.shape(x) * [1, 2, 2, 1]
       if not x.shape.is_fully_defined() else
       [dim * multiple for dim, multiple in zip(x.shape,
                                                [1, 2, 2, 1])]))


class FE(ssd_meta_arch.SSDFeatureExtractor):

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
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['last_stride32'], 64, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['last_stride32'], 64, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 32, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 32, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 32, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = feature_map_32
    layer32_c = slim.conv2d(layer32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 32, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.conv2d(layer32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 32, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64_c = slim.conv2d(layer64, 128, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64_c = slim.conv2d(layer64_c, 32, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)
    layer64_c = slim.conv2d(layer64, 128, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64_c = slim.conv2d(layer64_c, 32, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)

    return [layer16, layer32, layer32, layer64, layer64]


class FEWide(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['last_stride32'], 128, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['last_stride32'], 128, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 64, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = feature_map_32
    layer32_c = slim.conv2d(layer32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 64, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.conv2d(layer32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 64, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64_c = slim.conv2d(layer64, 128, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64_c = slim.conv2d(layer64_c, 64, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)
    layer64_c = slim.conv2d(layer64, 128, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64_c = slim.conv2d(layer64_c, 64, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)

    return [layer16, layer32, layer32, layer64, layer64]


class FEWideShallow(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['last_stride32'], 128, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['last_stride32'], 128, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 64, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = feature_map_32
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64_c = slim.conv2d(layer64, 64, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)
    layer64_c = slim.conv2d(layer64, 64, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)

    return [layer16, layer32, layer32, layer64, layer64]


class FEConcat32(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['concat_stride32'], 128, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['concat_stride32'], 128, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 64, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = feature_map_32
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64_c = slim.conv2d(layer64, 64, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)
    layer64_c = slim.conv2d(layer64, 64, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)

    return [layer16, layer32, layer32, layer64, layer64]


class FEPen32(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['penultimate_stride32'], 128, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['penultimate_stride32'], 128, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 64, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = feature_map_32
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64_c = slim.conv2d(layer64, 64, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)
    layer64_c = slim.conv2d(layer64, 64, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)

    return [layer16, layer32, layer32, layer64, layer64]


class FEFat16(ssd_meta_arch.SSDFeatureExtractor):

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
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['last_stride32'], 64, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['last_stride32'], 64, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 32, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = feature_map_32
    layer32_c = slim.conv2d(layer32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 32, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.conv2d(layer32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 32, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64_c = slim.conv2d(layer64, 128, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64_c = slim.conv2d(layer64_c, 32, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)
    layer64_c = slim.conv2d(layer64, 128, 1)
    layer64_c = slim.separable_conv2d(layer64_c, None, 3, 1)
    layer64_c = slim.conv2d(layer64_c, 32, 1)
    layer64 = tf.concat([layer64, layer64_c], 3)

    return [layer16, layer16, layer32, layer64, layer64]
