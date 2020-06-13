import math
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
from utils import *
from tensorflow.contrib.layers.python.layers import normalization


class batch_norm(object):
	# h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
	def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
		                                    scale=True, scope=self.name)


def batch_normalization(data, is_training, name, reuse=None):  # work !!!
	return tf.contrib.layers.batch_norm(data, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


'''
def batch_normalization_slim(data,is_training,name,reuse=None):  #havn't tried!
	return slim.batch_norm(decay=0.9,
						   epsilon=1e-5,
						   scale=True,
						   scope=name,
						   is_training=is_training,
						   updates_collections=tf.GraphKeys.UPDATE_OPS)
'''

'''
def batch_normalization(data,is_training,name,reuse=None):   #follow PointCNN, does not work! need to look up the reason.
	return tf.layers.batch_normalization(data, momentum=0.99, training=is_training,
										 beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
										 gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
										 reuse=reuse, name=name)
'''


def residual_block_with_IN(incoming, nb_blocks, out_channels, downsample=False,
                           downsample_strides=2, activation='relu', batch_norm=True,
                           bias=True, weights_init='variance_scaling',
                           bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                           trainable=True, restore=True, reuse=False, scope=None,
                           name="ResidualBlock", is_training=True):
	""" Residual Block.

	A residual block as described in MSRA's Deep Residual Network paper.
	Full pre-activation architecture is used here.

	Input:
		4-D Tensor [batch, height, width, in_channels].

	Output:
		4-D Tensor [batch, new height, new width, nb_filter].

	Arguments:
		incoming: `Tensor`. Incoming 4-D Layer.
		nb_blocks: `int`. Number of layer blocks.
		out_channels: `int`. The number of convolutional filters of the
			convolution layers.
		downsample: `bool`. If True, apply downsampling using
			'downsample_strides' for strides.
		downsample_strides: `int`. The strides to use when downsampling.
		activation: `str` (name) or `function` (returning a `Tensor`).
			Activation applied to this layer (see tflearn.activations).
			Default: 'linear'.
		batch_norm: `bool`. If True, apply batch normalization.
		bias: `bool`. If True, a bias is used.
		weights_init: `str` (name) or `Tensor`. Weights initialization.
			(see tflearn.initializations) Default: 'uniform_scaling'.
		bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
			(see tflearn.initializations) Default: 'zeros'.
		regularizer: `str` (name) or `Tensor`. Add a regularizer to this
			layer weights (see tflearn.regularizers). Default: None.
		weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
		trainable: `bool`. If True, weights will be trainable.
		restore: `bool`. If True, this layer weights will be restored when
			loading a model.
		reuse: `bool`. If True and 'scope' is provided, this layer variables
			will be reused (shared).
		scope: `str`. Define this layer scope (optional). A scope can be
			used to share variables between layers. Note that scope will
			override name.
		name: A name for this layer (optional). Default: 'ShallowBottleneck'.
		is_training: True for training mode and False for val or test mode.
	References:
		- Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
			Zhang, Shaoqing Ren, Jian Sun. 2015.
		- Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
			Zhang, Shaoqing Ren, Jian Sun. 2015.

	Links:
		- [http://arxiv.org/pdf/1512.03385v1.pdf]
			(http://arxiv.org/pdf/1512.03385v1.pdf)
		- [Identity Mappings in Deep Residual Networks]
			(https://arxiv.org/pdf/1603.05027v2.pdf)

	"""
	resnet = incoming
	in_channels = incoming.get_shape().as_list()[-1]

	with tf.variable_scope(values=[incoming], name_or_scope=scope, default_name=name, reuse=reuse) as scope:
		name = scope.name  # TODO
		for i in range(nb_blocks):

			identity = resnet

			if not downsample:
				downsample_strides = 1

			if batch_norm:
				resnet = normalization.instance_norm(resnet, scope=name + '_bn_' + str(i) + '_1')
			resnet = tflearn.activation(resnet, activation)

			resnet = tflearn.conv_2d(resnet, out_channels, 3,
			                         downsample_strides, 'same', 'linear',
			                         bias, weights_init, bias_init,
			                         regularizer, weight_decay, trainable,
			                         restore)

			if batch_norm:
				resnet = normalization.instance_norm(resnet, scope=name + '_bn_' + str(i) + '_2')
			resnet = tflearn.activation(resnet, activation)

			resnet = tflearn.conv_2d(resnet, out_channels, 3, downsample_strides, 'same',
			                         'linear', bias, weights_init,
			                         bias_init, regularizer, weight_decay,
			                         trainable, restore)

			# Downsampling
			if downsample_strides > 1:
				identity = tflearn.avg_pool_2d(identity, 1,
				                               downsample_strides)

			# Projection to new dimension
			'''
			if in_channels != out_channels:
				ch = (out_channels - in_channels)//2
				identity = tf.pad(identity,
								  [[0, 0], [0, 0], [0, 0], [ch, ch]])
				in_channels = out_channels
			'''
			resnet = resnet + identity

	return resnet


def residual_block_1x1_conv(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock", is_training=True):
	""" Residual Block.

	A residual block as described in MSRA's Deep Residual Network paper.
	Full pre-activation architecture is used here.

	Input:
		4-D Tensor [batch, height, width, in_channels].

	Output:
		4-D Tensor [batch, new height, new width, nb_filter].

	Arguments:
		incoming: `Tensor`. Incoming 4-D Layer.
		nb_blocks: `int`. Number of layer blocks.
		out_channels: `int`. The number of convolutional filters of the
			convolution layers.
		downsample: `bool`. If True, apply downsampling using
			'downsample_strides' for strides.
		downsample_strides: `int`. The strides to use when downsampling.
		activation: `str` (name) or `function` (returning a `Tensor`).
			Activation applied to this layer (see tflearn.activations).
			Default: 'linear'.
		batch_norm: `bool`. If True, apply batch normalization.
		bias: `bool`. If True, a bias is used.
		weights_init: `str` (name) or `Tensor`. Weights initialization.
			(see tflearn.initializations) Default: 'uniform_scaling'.
		bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
			(see tflearn.initializations) Default: 'zeros'.
		regularizer: `str` (name) or `Tensor`. Add a regularizer to this
			layer weights (see tflearn.regularizers). Default: None.
		weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
		trainable: `bool`. If True, weights will be trainable.
		restore: `bool`. If True, this layer weights will be restored when
			loading a model.
		reuse: `bool`. If True and 'scope' is provided, this layer variables
			will be reused (shared).
		scope: `str`. Define this layer scope (optional). A scope can be
			used to share variables between layers. Note that scope will
			override name.
		name: A name for this layer (optional). Default: 'ShallowBottleneck'.
		is_training: True for training mode and False for val or test mode.
	References:
		- Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
			Zhang, Shaoqing Ren, Jian Sun. 2015.
		- Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
			Zhang, Shaoqing Ren, Jian Sun. 2015.

	Links:
		- [http://arxiv.org/pdf/1512.03385v1.pdf]
			(http://arxiv.org/pdf/1512.03385v1.pdf)
		- [Identity Mappings in Deep Residual Networks]
			(https://arxiv.org/pdf/1603.05027v2.pdf)

	"""
	resnet = incoming
	in_channels = incoming.get_shape().as_list()[-1]

	with tf.variable_scope(values=[incoming], name_or_scope=scope, default_name=name, reuse=reuse) as scope:
		name = scope.name  # TODO
		for i in range(nb_blocks):

			identity = resnet

			if not downsample:
				downsample_strides = 1

			if batch_norm:
				resnet = batch_normalization(resnet, is_training, name=name + '_bn_' + str(i) + '_1')
			resnet = tflearn.activation(resnet, activation)



			resnet = tflearn.conv_2d(resnet, out_channels, 1,
			                         downsample_strides, 'same', 'linear',
			                         bias, weights_init, bias_init,
			                         regularizer, weight_decay, trainable,
			                         restore)

			if batch_norm:
				resnet = batch_normalization(resnet, is_training, name=name + '_bn_' + str(i) + '_2')
			resnet = tflearn.activation(resnet, activation)

			resnet = tflearn.conv_2d(resnet, out_channels, 1, downsample_strides, 'same',
			                         'linear', bias, weights_init,
			                         bias_init, regularizer, weight_decay,
			                         trainable, restore)

			# Downsampling
			if downsample_strides > 1:
				identity = tflearn.avg_pool_2d(identity, 1,
				                               downsample_strides)

			# Projection to new dimension
			'''
			if in_channels != out_channels:
				ch = (out_channels - in_channels)//2
				identity = tf.pad(identity,
								  [[0, 0], [0, 0], [0, 0], [ch, ch]])
				in_channels = out_channels
			'''
			resnet = resnet + identity

	return resnet

def residual_block(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock", is_training=True):
	""" Residual Block.

	A residual block as described in MSRA's Deep Residual Network paper.
	Full pre-activation architecture is used here.

	Input:
		4-D Tensor [batch, height, width, in_channels].

	Output:
		4-D Tensor [batch, new height, new width, nb_filter].

	Arguments:
		incoming: `Tensor`. Incoming 4-D Layer.
		nb_blocks: `int`. Number of layer blocks.
		out_channels: `int`. The number of convolutional filters of the
			convolution layers.
		downsample: `bool`. If True, apply downsampling using
			'downsample_strides' for strides.
		downsample_strides: `int`. The strides to use when downsampling.
		activation: `str` (name) or `function` (returning a `Tensor`).
			Activation applied to this layer (see tflearn.activations).
			Default: 'linear'.
		batch_norm: `bool`. If True, apply batch normalization.
		bias: `bool`. If True, a bias is used.
		weights_init: `str` (name) or `Tensor`. Weights initialization.
			(see tflearn.initializations) Default: 'uniform_scaling'.
		bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
			(see tflearn.initializations) Default: 'zeros'.
		regularizer: `str` (name) or `Tensor`. Add a regularizer to this
			layer weights (see tflearn.regularizers). Default: None.
		weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
		trainable: `bool`. If True, weights will be trainable.
		restore: `bool`. If True, this layer weights will be restored when
			loading a model.
		reuse: `bool`. If True and 'scope' is provided, this layer variables
			will be reused (shared).
		scope: `str`. Define this layer scope (optional). A scope can be
			used to share variables between layers. Note that scope will
			override name.
		name: A name for this layer (optional). Default: 'ShallowBottleneck'.
		is_training: True for training mode and False for val or test mode.
	References:
		- Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
			Zhang, Shaoqing Ren, Jian Sun. 2015.
		- Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
			Zhang, Shaoqing Ren, Jian Sun. 2015.

	Links:
		- [http://arxiv.org/pdf/1512.03385v1.pdf]
			(http://arxiv.org/pdf/1512.03385v1.pdf)
		- [Identity Mappings in Deep Residual Networks]
			(https://arxiv.org/pdf/1603.05027v2.pdf)

	"""
	resnet = incoming
	in_channels = incoming.get_shape().as_list()[-1]

	with tf.variable_scope(values=[incoming], name_or_scope=scope, default_name=name, reuse=reuse) as scope:
		name = scope.name  # TODO
		for i in range(nb_blocks):

			identity = resnet

			if not downsample:
				downsample_strides = 1

			if batch_norm:
				resnet = batch_normalization(resnet, is_training, name=name + '_bn_' + str(i) + '_1')
			resnet = tflearn.activation(resnet, activation)

			resnet = tflearn.conv_2d(resnet, out_channels, 3,
			                         downsample_strides, 'same', 'linear',
			                         bias, weights_init, bias_init,
			                         regularizer, weight_decay, trainable,
			                         restore)

			if batch_norm:
				resnet = batch_normalization(resnet, is_training, name=name + '_bn_' + str(i) + '_2')
			resnet = tflearn.activation(resnet, activation)

			resnet = tflearn.conv_2d(resnet, out_channels, 3, downsample_strides, 'same',
			                         'linear', bias, weights_init,
			                         bias_init, regularizer, weight_decay,
			                         trainable, restore)

			# Downsampling
			if downsample_strides > 1:
				identity = tflearn.avg_pool_2d(identity, 1,
				                               downsample_strides)

			# Projection to new dimension
			'''
			if in_channels != out_channels:
				ch = (out_channels - in_channels)//2
				identity = tf.pad(identity,
								  [[0, 0], [0, 0], [0, 0], [ch, ch]])
				in_channels = out_channels
			'''
			resnet = resnet + identity

	return resnet


def binary_cross_entropy(preds, targets, name=None):
	"""Computes binary cross entropy given `preds`.

	For brevity, let `x = `, `z = targets`.  The logistic loss is

		loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

	Args:
		preds: A `Tensor` of type `float32` or `float64`.
		targets: A `Tensor` of the same type and shape as `preds`.
	"""
	eps = 1e-12
	with ops.op_scope([preds, targets], name, "bce_loss") as name:
		preds = ops.convert_to_tensor(preds, name="preds")
		targets = ops.convert_to_tensor(targets, name="targets")
		return tf.reduce_mean(-(targets * tf.log(preds + eps) +
		                        (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
	"""Concatenate conditioning vector on feature map axis."""
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
		                    initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases),
		                  [-1, conv.get_shape()[1], conv.get_shape()[2], conv.get_shape()[3]])
		# conv = tf.nn.bias_add(conv, biases)
		return conv


def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
		                    initializer=tf.random_normal_initializer(stddev=stddev))

		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
			                                strides=[1, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
			                        strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv


def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
		                         tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
		                       initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias


def depthwise_conv(input, name,activation_fn='relu'):
	input_c = input.get_shape().as_list()[-1]
	dw_conv_w = tf.Variable(tf.random_uniform((3, 3, input_c, 1)), dtype=tf.float32, name='dw_' + name)
	net = tf.nn.depthwise_conv2d(input, dw_conv_w, [1, 1, 1, 1],
	                                                     padding='SAME', name='dwconv_' + name)  # [bs,1,4,num_point]
	if activation_fn == 'sigmoid':
		net = tf.nn.sigmoid(net)
	else:
		net = tf.nn.sigmoid(net)
	return  net