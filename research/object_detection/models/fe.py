import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
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


class FEFat16(FE):

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


class FESSDLite(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_layout = {
        'from_layer': [
            'last_stride16', 'last_stride32', '', '', ''
        ],
        'layer_depth': [-1, -1, 512, 256, 256],
        'conv_kernel_size': [-1, -1, 3, 3, 2],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }
    with slim.arg_scope(self._conv_hyperparams_fn()):
      feature_maps = feature_map_generators.multi_resolution_feature_maps(
          feature_map_layout=feature_map_layout,
          depth_multiplier=0.25,
          min_depth=12345,
          insert_1x1_conv=True,
          image_features=image_features)

    return feature_maps.values()


class FESSDLitePen32(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_layout = {
        'from_layer': [
            'last_stride16', 'penultimate_stride32', '', '', ''
        ],
        'layer_depth': [-1, -1, 512, 256, 256],
        'conv_kernel_size': [-1, -1, 3, 3, 2],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }
    with slim.arg_scope(self._conv_hyperparams_fn()):
      feature_maps = feature_map_generators.multi_resolution_feature_maps(
          feature_map_layout=feature_map_layout,
          depth_multiplier=0.25,
          min_depth=12345,
          insert_1x1_conv=True,
          image_features=image_features)

    return feature_maps.values()


class FESSDLiteConcat32(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_layout = {
        'from_layer': [
            'last_stride16', 'concat_stride32', '', '', ''
        ],
        'layer_depth': [-1, -1, 512, 256, 256],
        'conv_kernel_size': [-1, -1, 3, 3, 2],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }
    with slim.arg_scope(self._conv_hyperparams_fn()):
      feature_maps = feature_map_generators.multi_resolution_feature_maps(
          feature_map_layout=feature_map_layout,
          depth_multiplier=0.25,
          min_depth=12345,
          insert_1x1_conv=True,
          image_features=image_features)

    return feature_maps.values()


class FEWiderShallowerConcat32(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['concat_stride32'], 192, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['concat_stride32'], 256, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 128, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 128, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = slim.conv2d(feature_map_32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 64, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.separable_conv2d(layer32, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 64, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64 = slim.separable_conv2d(layer64, None, 3, 1)
    layer64 = slim.conv2d(layer64, 192, 1)
    layer64 = slim.separable_conv2d(layer64, None, 3, 1)

    return [layer16, layer32, layer32, layer64, layer64]


class FEWiderShallowerConcat32Add(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['concat_stride32'], 192, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['concat_stride32'], 256, 1)
    feature_map_16 = (image_features['last_stride16'] +
                      resize_neareast_neighbor_nhwc_using_tile(
                          slim.conv2d(feature_map_32, 128, 1)))

    layer16 = slim.conv2d(feature_map_16, 128, 1)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.separable_conv2d(layer16, None, 3, 1)
    layer16_c = slim.conv2d(layer16_c, 64, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = slim.conv2d(feature_map_32, 128, 1)
    layer32_c = slim.separable_conv2d(layer32, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 64, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.separable_conv2d(layer32, None, 3, 1)
    layer32_c = slim.conv2d(layer32_c, 64, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64 = slim.separable_conv2d(layer64, None, 3, 1)
    layer64 = slim.conv2d(layer64, 192, 1)
    layer64 = slim.separable_conv2d(layer64, None, 3, 1)

    return [layer16, layer32, layer32, layer64, layer64]


class FEWiderShallowerConcat32Squeeze(FE):

  def _generate_feature_maps(self, image_features):
    feature_map_64 = slim.separable_conv2d(
        slim.conv2d(image_features['concat_stride32'], 192, 1), None, 3, 1,
        stride=2)
    feature_map_32 = slim.conv2d(image_features['concat_stride32'], 256, 1)
    feature_map_16 = tf.concat([image_features['last_stride16'],
                                resize_neareast_neighbor_nhwc_using_tile(
                                    slim.conv2d(feature_map_32, 128, 1))], 3)

    layer16 = slim.conv2d(feature_map_16, 128, 1)
    layer16_c = slim.conv2d(layer16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16_c, None, 3, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)
    layer16_c = slim.conv2d(layer16, 64, 1)
    layer16_c = slim.separable_conv2d(layer16_c, None, 3, 1)
    layer16 = tf.concat([layer16, layer16_c], 3)

    layer32 = slim.conv2d(feature_map_32, 128, 1)
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)
    layer32_c = slim.conv2d(layer32, 64, 1)
    layer32_c = slim.separable_conv2d(layer32_c, None, 3, 1)
    layer32 = tf.concat([layer32, layer32_c], 3)

    layer64 = feature_map_64
    layer64 = slim.separable_conv2d(layer64, None, 3, 1)
    layer64 = slim.conv2d(layer64, 192, 1)
    layer64 = slim.separable_conv2d(layer64, None, 3, 1)

    return [layer16, layer32, layer32, layer64, layer64]
