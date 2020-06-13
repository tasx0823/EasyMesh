from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2
from ops import *
from utils import *
import scipy.io as scio
import random
import tensorflow.contrib.slim as slim
import tflearn
import linecache
from cd_dist import *
import tf_util
from resnet import *
import shutil


class GeoGan(object):
	def __init__(self, sess, image_size=64,
	             batch_size=1, sample_size=1, output_size=128,
	             gf_dim=32, df_dim=32, L1_lambda=100,
	             input_c_dim=3, output_c_dim=3, dataset_name='airplane',
	             checkpoint_dir=None, sample_dir=None):

		"""
		Args:
			sess: TensorFlow session
			batch_size: The size of batch. Should be specified before training.
			output_size: (optional) The resolution in pixels of the images. [256]
			gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
			df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
			input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
			output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
		"""

		self.sess = sess
		self.is_grayscale = (input_c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.sample_size = sample_size
		self.output_size = output_size

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.L1_lambda = L1_lambda

		# self.cates = ['airplane', 'bench', 'cabinet','car', 'chair', 'display', 'lamp', 'loudspeaker', 'rifle', 'sofa',
		#              'table', 'telephone']
		self.cates = ['car']

		self.experiment_name = 'a10_2_7'
		self.checkpoint_dir = checkpoint_dir + 'ckpt_' + self.experiment_name + '_' + self.cates[0]
		self.sample_dir = sample_dir + 'sp_' + self.experiment_name + '_' + self.cates[0]

		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)
		shutil.copy('./main_folding.py', os.path.join(self.checkpoint_dir, 'main_folding.py'))
		shutil.copy('./model_folding.py', os.path.join(self.checkpoint_dir, 'model_folding.py'))

		self.build_dict()
		for i, cate in enumerate(self.cates):
			self.cate2label[cate] = i

		if len(self.cates) > 1:
			self.cate = 'all categories'
			self.use_amgan = True
			self.sample_number = 5000
		else:
			self.cate = self.cates[0]
			self.use_amgan = False
			self.sample_number = 1000
		self.use_chamfer = True

		self.data_rootdir = r'/data1/sunx/CVPR2019/data'
		# self.data_rootdir = r'/S1/CSCL/sunx/geometry'

		for cate in self.cates:
			# --------------load data of 64x64 ---------------#
			self.matfile[cate] = self.data_rootdir + r'/{:s}/{:s}_train_geoimg_64.mat'.format(cate, cate)
			self.matfile_normal[cate] = self.data_rootdir + r'/{:s}/{:s}_train_normal_64.mat'.format(cate, cate)
			self.matfile_test[cate] = self.data_rootdir + r'/{:s}/{:s}_test_geoimg_64.mat'.format(cate, cate)

			self.mat_data[cate] = scio.loadmat(self.matfile[cate])['geoimgs']
			self.mat_data_test[cate] = scio.loadmat(self.matfile_test[cate])['geoimgs']
			self.mat_normal[cate] = scio.loadmat(self.matfile_normal[cate])['normal']
			self.mat_data_size[cate] = np.shape(self.mat_data[cate])[0]

			# --------------load data of 32x32 ---------------#
			self.matfile_32[cate] = self.data_rootdir + r'/{:s}/{:s}_train_geoimg_32.mat'.format(cate, cate)
			self.matfile_normal_32[cate] = self.data_rootdir + r'/{:s}/{:s}_train_normal_32.mat'.format(cate, cate)
			self.matfile_test_32[cate] = self.data_rootdir + r'/{:s}/{:s}_test_geoimg_32.mat'.format(cate, cate)

			self.mat_data_32[cate] = scio.loadmat(self.matfile_32[cate])['geoimgs']
			self.mat_normal_32[cate] = scio.loadmat(self.matfile_normal_32[cate])['normal']
			self.mat_data_size_32[cate] = np.shape(self.mat_data_32[cate])[0]

			# --------------load data of 32x32 ---------------#
			self.matfile_16[cate] = self.data_rootdir + r'/{:s}/{:s}_train_geoimg_16.mat'.format(cate, cate)
			self.matfile_normal_16[cate] = self.data_rootdir + r'/{:s}/{:s}_train_normal_16.mat'.format(cate, cate)
			self.matfile_test_16[cate] = self.data_rootdir + r'/{:s}/{:s}_test_geoimg_16.mat'.format(cate, cate)

			self.mat_data_16[cate] = scio.loadmat(self.matfile_16[cate])['geoimgs']
			self.mat_normal_16[cate] = scio.loadmat(self.matfile_normal_16[cate])['normal']
			self.mat_data_size_16[cate] = np.shape(self.mat_data_16[cate])[0]

			# ---------------------others---------------------#
			self.mat_data_size_test[cate] = np.shape(self.mat_data_test[cate])[0]
			# self.depth_dir = r'/data/sunx/geoimage/car/data/depth_image'
			self.depth_dir[cate] = self.data_rootdir + r'/{:s}/sil_train_new'.format(cate)
			self.depth_dir_test[cate] = self.data_rootdir + r'/{:s}/sil_test'.format(cate)
			self.files_in_depth_dir = os.listdir(self.depth_dir[cate])

			self.depth_dir_old[cate] = self.data_rootdir + r'/{:s}/sil_train'.format(cate)

		self.camera_poses = np.loadtxt('point66.txt')
		# self.depth_idx = [1,26,27,37,44,55,59,64]  # randomly select 6 images from these 8 idxs
		# self.depth_idx = [1, 10, 15, 26, 27, 33, 37, 40, 44, 50, 55, 57, 59, 62, 64]
		self.depth_idx = [0, 1, 4, 5, 24, 25, 31, 32, 51, 56]  # for my new generated image
		self.depth_num = 1

		self.input_c_dim = input_c_dim  # 3 is the depth of geometry image
		self.output_c_dim = output_c_dim

		self.dataset_name = self.cate

		# ---------------initial mesh-----------------#
		self.scale_factor = 4
		mat = scio.loadmat('init_geo')['geoimgs']
		mat = cv2.resize(mat, (int(self.image_size / self.scale_factor), int(self.image_size / self.scale_factor)))
		mats = []
		for i in range(self.batch_size):
			mats.append(mat)
		self.mat = np.stack(mats)
		# --------------------------------------------#
		self.build_model()

	def build_model(self):
		self.real_data = tf.placeholder(tf.float32,
		                                [self.batch_size, self.image_size, self.image_size,
		                                 self.input_c_dim + self.depth_num],
		                                name='real_A_and_B_images')
		self.init_geo_img = tf.placeholder(tf.float32, [self.batch_size, int(self.image_size / self.scale_factor),
		                                                int(self.image_size) / self.scale_factor, 3])
		self.camera_pose_gt = tf.placeholder(tf.float32, [self.batch_size, 3])

		self.sil_aux_gt = tf.placeholder(tf.float32, [self.batch_size * 18, self.image_size, self.image_size, 1])
		self.real_B_32 = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 3], name='real_geo_img_32')
		self.real_B_16 = tf.placeholder(tf.float32, [self.batch_size, 16, 16, 3], name='real_geo_img_16')

		self.real_B = self.real_data[:, :, :, :self.input_c_dim]  # ground truth geometry image
		self.real_A = self.real_data[:, :, :,
		              self.input_c_dim:self.input_c_dim + self.depth_num]  # Silhouette or depth image
		self.weighted_mask = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1])
		self.gt_normal = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
		self.gt_normal_32 = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 3])
		self.gt_normal_16 = tf.placeholder(tf.float32, [self.batch_size, 16, 16, 3])

		self.geoimg_labels = tf.placeholder(tf.int32, [self.batch_size],
		                                    name='geoimg_labels')  # label corresponding to point cloud
		self.camera_pose_pre = self.estimate_pose(self.real_A)  # batch_size*3
		# self.fake_B = self.generator(self.real_A,self.camera_pose_pre)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		self.fake_B, self.fake_output2, self.fake_output1 = self.generator(self.real_A,
		                                                                   self.camera_pose_pre)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# -------------------------------------------------------------#
		# pose estimate module
		# self.camera_pose_pre = self.estimate_pose(self.real_A)  #batch_size*3
		self.pose_estimate_loss = tf.reduce_mean(tf.abs(self.camera_pose_pre - self.camera_pose_gt))
		# self.reconstructed_silhouette = self.reconstruct_silhouette(self.fake_B,self.camera_pose_pre)
		self.reconstructed_silhouette = self.reconstruct_silhouette_from_geoimg(self.fake_B,
		                                                                        self.camera_pose_pre)  # !!!!!!!!!
		self.reconstructed_silhouette_gt = self.reconstruct_silhouette_from_geoimg(self.real_B, self.camera_pose_pre)

		self.reconstruct_sil_loss = tf.reduce_mean(((
					                                            self.reconstructed_silhouette_gt - self.reconstructed_silhouette) * (
					                                            self.reconstructed_silhouette_gt - self.reconstructed_silhouette)) * self.weighted_mask)
		self.pose_loss = self.pose_estimate_loss + self.reconstruct_sil_loss
		# -------------------------------------------------------------#
		# 3d shape local feature extracted module
		# insight: A good generated points cloud should render good 2d silhouette image from any perspective
		self.view_points = [8, 10, 15, 16, 26, 27, 30, 33, 39, 40, 44, 45, 48, 52, 54, 57, 62, 65]
		camera_pose = []
		for k in self.view_points:
			camera_pose.append(self.camera_poses[k])
		camera_pose_aux = tf.tile(np.stack(camera_pose, 0).astype(np.float32), [self.batch_size, 1])
		self.fake_B_tile = tf.reshape(tf.tile(self.fake_B, [1, 18, 1, 1]),
		                              [self.batch_size * 18, self.image_size, self.image_size, 3])
		self.reconstructed_silhouette_aux = self.reconstruct_silhouette_from_geoimg(self.fake_B_tile, camera_pose_aux)

		self.real_B_tile = tf.reshape(tf.tile(self.real_B, [1, 18, 1, 1]),
		                              [self.batch_size * 18, self.image_size, self.image_size, 3])
		self.reconstructed_silhouette_aux_gt = self.reconstruct_silhouette_from_geoimg(self.real_B_tile,
		                                                                               camera_pose_aux)
		self.reconstruct_sil_loss_aux = 30 * tf.reduce_mean(
			(self.reconstructed_silhouette_aux - self.reconstructed_silhouette_aux_gt) * (
						self.reconstructed_silhouette_aux - self.reconstructed_silhouette_aux_gt))
		self.pose_loss += self.reconstruct_sil_loss_aux
		# -------------------------------------------------------------#
		# normal loss
		self.normal_loss1 = self.cal_normal_loss(self.fake_B, self.gt_normal) * 1
		self.normal_loss2 = self.cal_normal_loss(self.fake_output2, self.gt_normal_32)
		self.normal_loss3 = self.cal_normal_loss(self.fake_output1, self.gt_normal_16)
		self.normal_loss = self.normal_loss1 + self.normal_loss2 + self.normal_loss3

		# -------------------------------------------------------------#
		# edge distance loss
		self.edge_distance_loss1 = 1.0 * self.cal_edge_distance_loss(self.fake_B, self.real_B)
		self.edge_distance_loss2 = 1 * self.cal_edge_distance_loss(self.fake_output2,
		                                                           self.fake_output2)  # the second parameter is not important
		self.edge_distance_loss3 = 1 * self.cal_edge_distance_loss(self.fake_output1, self.fake_output1)
		self.edge_distance_loss = self.edge_distance_loss1 + self.edge_distance_loss2 + self.edge_distance_loss3
		# -------------------------------------------------------------#
		# chamfer loss
		if self.use_chamfer:
			fake_B_pad2 = self.padding_for_geoimg(self.padding_for_geoimg(self.fake_B))
			real_B_pad2 = self.padding_for_geoimg(self.padding_for_geoimg(self.real_B))
			self.fake_B_points = tf.reshape(fake_B_pad2, (self.batch_size, (self.image_size+4) * (self.image_size+4), 3))
			self.real_B_points = tf.reshape(real_B_pad2, (self.batch_size, (self.image_size+4) * (self.image_size+4), 3))
			dist1, idx1, dist2, idx2 = nn_distance(self.real_B_points, self.fake_B_points)
			self.chamfer_loss1 = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 300

			self.fake_B_points_32 = tf.reshape(self.fake_output2, (self.batch_size, 32 * 32, 3))
			self.real_B_points_32 = tf.reshape(self.real_B_32, (self.batch_size, 32 * 32, 3))
			dist1_2, idx1_2, dist2_2, idx2_2 = nn_distance(self.real_B_points_32, self.fake_B_points_32)
			self.chamfer_loss2 = (tf.reduce_mean(dist1_2) + 0.55 * tf.reduce_mean(dist2_2)) * 300

			self.fake_B_points_16 = tf.reshape(self.fake_output1, (self.batch_size, 16 * 16, 3))
			self.real_B_points_16 = tf.reshape(self.real_B_16, (self.batch_size, 16 * 16, 3))
			dist1_3, idx1_3, dist2_3, idx2_3 = nn_distance(self.real_B_points_16, self.fake_B_points_16)
			self.chamfer_loss3 = (tf.reduce_mean(dist1_3) + 0.55 * tf.reduce_mean(dist2_3)) * 300

			self.chamfer_loss = self.chamfer_loss1 + self.chamfer_loss2 + self.chamfer_loss3

		# -------------------------------------------------------------#
		if self.use_amgan:
			pass
		else:
			self.D, self.D_logits, self.net1_real = self.discriminator_pointnet(self.real_B, reuse=False)
			self.D_, self.D_logits_, self.net1_fake = self.discriminator_pointnet(self.fake_B, reuse=True)

			self.d_sum = tf.summary.histogram("d", self.D)
			self.d__sum = tf.summary.histogram("d_", self.D_)
			self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

			self.d_loss_real = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
			self.d_loss_fake = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
			self.g_loss0 = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

			self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
			self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

			self.d_loss = (self.d_loss_real + self.d_loss_fake) * 0.01

		# self.l2_loss = tf.reduce_mean((self.fake_B - self.real_B) * (self.fake_B - self.real_B)) * 10
		# self.l2_loss = tf.reduce_mean(tf.abs(self.fake_B-self.real_B)) * 10
		if self.use_chamfer:
			self.l2_loss = self.chamfer_loss
		# self.l2_loss = self.chamfer_loss
		else:
			self.l2_loss = tf.reduce_mean((self.fake_B - self.real_B) * (self.fake_B - self.real_B)) * 10
		# self.perception_loss = 1000*tf.reduce_mean((self.net1_fake - self.net1_real) * (self.net1_fake - self.net1_real))

		self.g_loss = 0.01 * self.g_loss0 + self.l2_loss + self.pose_loss + self.edge_distance_loss + self.normal_loss

		self.g_loss_for_output1 = self.chamfer_loss3 + self.edge_distance_loss3 + self.normal_loss3
		self.g_loss_for_output2 = self.chamfer_loss2 + self.edge_distance_loss2 + self.normal_loss2

		self.l2_loss_sum = tf.summary.scalar("l2_loss", self.l2_loss)
		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]
		# print(self.g_vars)
		self.saver = tf.train.Saver()
		self.posenet_saver = tf.train.Saver(var_list=[var for var in t_vars if 'g_estimate_pose' in var.name])


	def cal_edge_distance_loss(self, a, gt_geoimg):
		assert np.shape(a) == np.shape(gt_geoimg)
		a = self.padding_for_geoimg(a)
		with tf.variable_scope('fix_calculate_edge_distance_loss'):
			# w = np.array([[1,1,1],[1,-8.0,1],[1,1,1]]).astype(np.float32)
			# w = np.array([[-1,1]]).astype(np.float32)
			w1 = np.array([[-1], [1]]).astype(np.float32)
			filter1 = []
			for i in range(3):
				filter1.append(w1)
			filter1 = np.stack(filter1, 2)
			filter1 = np.expand_dims(filter1, 3)
			e1 = tf.nn.depthwise_conv2d(a, filter1, strides=[1, 1, 1, 1], padding='VALID', name='depthwise_conv1')
			edge_loss1 = tf.reduce_sum(tf.square(e1), 3)

			w2 = np.array([[-1, 1]]).astype(np.float32)
			filter2 = []
			for i in range(3):
				filter2.append(w2)
			filter2 = np.stack(filter2, 2)
			filter2 = np.expand_dims(filter2, 3)
			e2 = tf.nn.depthwise_conv2d(a, filter2, strides=[1, 1, 1, 1], padding='VALID', name='depthwise_conv2')
			edge_loss2 = tf.reduce_sum(tf.square(e2), 3)

			w3 = np.array([[0, 1], [-1, 0]]).astype(np.float32)
			filter3 = []
			for i in range(3):
				filter3.append(w3)
			filter3 = np.stack(filter3, 2)
			filter3 = np.expand_dims(filter3, 3)
			e3 = tf.nn.depthwise_conv2d(a, filter3, strides=[1, 1, 1, 1], padding='VALID', name='depthwise_conv3')
			edge_loss3 = tf.reduce_sum(tf.square(e3), 3)

			w4 = np.array([[1, 0], [0, -1]]).astype(np.float32)
			filter4 = []
			for i in range(3):
				filter4.append(w4)
			filter4 = np.stack(filter4, 2)
			filter4 = np.expand_dims(filter4, 3)
			e4 = tf.nn.depthwise_conv2d(a, filter4, strides=[1, 1, 1, 1], padding='VALID', name='depthwise_conv4')
			edge_loss4 = tf.reduce_sum(tf.square(e4), 3)

			# edge_loss = tf.reduce_mean(edge_loss1 + edge_loss2 + edge_loss3 + edge_loss4)
			edge_loss = tf.reduce_mean(edge_loss1) \
			            + tf.reduce_mean(edge_loss2) \
			            + tf.reduce_mean(edge_loss3) \
			            + tf.reduce_mean(edge_loss4)

			return edge_loss * 50
			'''
			w = np.array([[-2, 1], [0, 1]]).astype(np.float32)
			weight = np.zeros([2, 2, 3, 1])
			for i in range(3):
				weight[:, :, i, 0] = w
			w = tf.Variable(weight.astype(np.float32))
			b = tf.abs(tf.nn.conv2d(a, w, strides=[1, 1, 1, 1], padding='SAME'))
			gt = tf.abs(tf.nn.conv2d(gt_geoimg, w, strides=[1, 1, 1, 1], padding='SAME'))
			edge_loss = tf.reduce_mean(tf.abs(b - gt))
			return edge_loss'''

	def cal_normal_loss(self, a, gt_normal):
		assert np.shape(a) == np.shape(gt_normal)

		with tf.variable_scope('fix_calculate_normal_loss'):
			# calculate tangent line of each point
			ww = np.array([[1, 1, 1], [1, -8.0, 1], [1, 1, 1]]).astype(np.float32)
			weight = np.zeros([3, 3, 1, 1]).astype(np.float32)
			weight[:, :, 0, 0] = ww
			weight = tf.Variable(weight)
			bs, w, h, c = np.shape(a)
			a_transpose = tf.transpose(a, [0, 3, 1, 2])

			a1 = tf.slice(a_transpose, [0, 0, 0, 0], [bs, 1, w, h])
			a1 = tf.transpose(a1, [0, 2, 3, 1])
			a1 = tf.nn.conv2d(a1, weight, strides=[1, 1, 1, 1], padding='SAME')

			a2 = tf.slice(a_transpose, [0, 1, 0, 0], [bs, 1, w, h])
			a2 = tf.transpose(a2, [0, 2, 3, 1])
			a2 = tf.nn.conv2d(a2, weight, strides=[1, 1, 1, 1], padding='SAME')

			a3 = tf.slice(a_transpose, [0, 2, 0, 0], [bs, 1, w, h])
			a3 = tf.transpose(a3, [0, 2, 3, 1])
			a3 = tf.nn.conv2d(a3, weight, strides=[1, 1, 1, 1], padding='SAME')

			# print('a3 shape is ',np.shape(a3))
			a_concat = tf.concat([a1, a2, a3], axis=3)
			a_concat_exp = tf.expand_dims(a_concat, axis=3)
			gt_normal_exp = tf.expand_dims(gt_normal, axis=4)
			print('a_aconcat_exp ', np.shape(a_concat_exp))
			print('gt_normal ', np.shape(gt_normal_exp))
			cosine = tf.matmul(a_concat_exp, gt_normal_exp)
			cosine = tf.squeeze(cosine, axis=4)
			cosine = cosine * cosine
			# print('cosine shape ',np.shape(cosine))

			normal_loss = tf.reduce_mean(cosine)
			return normal_loss

	def reconstruct_silhouette(self, image, cam_locs, reuse=False):
		with tf.variable_scope('g_reconstruct_silhouette') as scope:
			if reuse:
				tf.get_variable_scope().reuse_variables()

			bs, w, h, c = image.get_shape().as_list()
			cam_locs = tf.convert_to_tensor(cam_locs, tf.float32)
			bs1, _ = cam_locs.get_shape().as_list()
			assert bs == bs1
			points = tf.reshape(image, [bs, w * h, c])

			def cross(a, b):
				# calculate the cross product
				# shape a: Bs x 3
				# shape b: Bs x 3
				cross_product = tf.cross(a, b)
				nm = tf.sqrt(tf.reduce_sum(cross_product * cross_product, axis=1, keep_dims=True))
				return cross_product / nm

			def get_up_direction(cam_locs):
				bs = np.shape(cam_locs)[0]
				ups = np.ones([bs, 3], dtype=np.float32) * np.array([0, 1, 0], dtype=np.float32)
				return ups

			# points :  [BxNx3]
			# cam_locs : [Bx3],point to the (0,0,0)
			points_shape = tf.shape(points)
			# bs = points.shape[0]

			N = points.shape[1]
			out_img_size = 128
			# ---------------------------------------------------------------------------------------------#
			# subject mean of each dimension to ensure the center lies on the origin
			mean = tf.reduce_mean(points, axis=1)
			mean = tf.expand_dims(mean, axis=1)
			points = points - mean
			# ---------------------------------------------------------------------------------------------#
			# Guarantee the silhouettes rendered from the points generated by geometry image
			# match the ground truth after ordinary steps.
			sita_x = - np.pi / 2
			sita_y = 0
			sita_z = - np.pi / 2
			T_x = np.array([[1, 0, 0], [0, np.cos(sita_x), -np.sin(sita_x)], [0, np.sin(sita_x), np.cos(sita_x)]],
			               dtype=np.float32)
			T_y = np.array([[np.cos(sita_y), 0, np.sin(sita_y)], [0, 1, 0], [-np.sin(sita_y), 0, np.cos(sita_y)]],
			               dtype=np.float32)
			T_z = np.array([[np.cos(sita_z), -np.sin(sita_z), 0], [np.sin(sita_z), np.cos(sita_z), 0], [0, 0, 1]],
			               dtype=np.float32)
			T_x = tf.tile(T_x, [bs, 1])
			T_x = tf.reshape(T_x, [bs, 3, 3])
			T_y = tf.tile(T_y, [bs, 1])
			T_y = tf.reshape(T_y, [bs, 3, 3])
			T_z = tf.tile(T_z, [bs, 1])
			T_z = tf.reshape(T_z, [bs, 3, 3])

			points = tf.transpose(points, [0, 2, 1])  # Bsx3xN
			points = tf.matmul(T_z, points)  # revolve around z axis
			points = tf.matmul(T_y, points)  # revolve around y axis
			points = tf.matmul(T_x, points)  # revolve around x axis
			points = tf.transpose(points, [0, 2, 1])  # back to shape Bs x n x 3
			# return points        #have been chacked,correct till here:  check1
			# ---------------------------------------------------------------------------------------------#
			# map the points from world coordinate to a given camera coordinate
			points = points - tf.expand_dims(cam_locs, 1)  # translation from one world origin to camera origin
			up = get_up_direction(cam_locs)  # Bs x 3
			z_axis = - cam_locs  # Bs x 3
			x_axis = cross(up, z_axis)  # Have been normed, shape: Bs x 3
			y_axis = cross(z_axis, x_axis)  # Have been normed, shape: Bs x 3
			# camera_axis = np.concatenate([x_axis, y_axis, z_axis], 1)  # shape:Bs x 9
			camera_axis = tf.concat([x_axis, y_axis, z_axis], 1)  # shape:Bs x 9
			# camera_axis = np.reshape(camera_axis, [bs, 3, 3])  # shape:Bs x 3 x 3
			camera_axis = tf.reshape(camera_axis, [bs, 3, 3])  # shape:Bs x 3 x 3
			world_axis = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]], dtype=np.float32)  # shape:Bs x 3 x 3
			world_axis = np.tile(world_axis, [bs, 1, 1])  # shape:Bs x 3 x 3
			# rot_mat = np.matmul(world_axis, np.linalg.inv(camera_axis))  # bs x 3 x 3
			rot_mat = tf.matmul(world_axis, tf.matrix_inverse(camera_axis))  # bs x 3 x 3
			uvw = tf.matmul(points, rot_mat)  # scaled in (-1,1),shape :bs x n x 3
			uvw = uvw * out_img_size / 2  # scaled to out_imgsize/2 ,in the case the self.image_size is out_img_size
			# return uvw   #have been checked,correct till here: check 2
			# ---------------------------------------------------------------------------------------------#
			x, y, z = tf.split(uvw, 3, axis=2)  # each shape: bs x n x 1

			x = tf.reshape(x, [-1])  # bs x n
			y = tf.reshape(y, [-1])  # bs x n
			z = tf.reshape(z, [-1])  # bs x n

			scale_factor = 1.8
			x_int = tf.to_int32(tf.round(out_img_size / 2 - x * scale_factor))
			y_int = tf.to_int32(tf.round(out_img_size / 2 - y * scale_factor))
			depth_value = z  # bs x n
			# filter the out-range points
			# maskInside = (x_int>=0)&(x_int<128)

			batchIdx, _ = np.meshgrid(range(bs), range(128 * 128), indexing="ij")  # geometry image size
			batchIdx = batchIdx.reshape([-1])
			scatterIdx = tf.stack([batchIdx, y_int, x_int], axis=1)
			scatterContent = depth_value
			scatterShape = tf.constant([bs, out_img_size, out_img_size])  # silhouette image size
			newContent = tf.scatter_nd(scatterIdx, scatterContent, scatterShape)  # bs*w*h
			# newContent = tf.minimum(newContent, 255.0)  # scale pixel value to (0,255)
			newContent = tf.expand_dims(newContent, axis=3)

			# ---------------------------------------------------------------------------------------------#
			# filter the noise point from the reconstructed
			newContent = slim.conv2d(newContent, 32, [2, 2], stride=1, scope='smooth_conv1')
			newContent = slim.conv2d(newContent, 1, [2, 2], stride=1, scope='smooth_conv2')
			# ---------------------------------------------------------------------------------------------#
			newContent = tf.minimum(newContent, 255.0)  # scale pixel value to (0,255)
			return newContent / 255.0  # have been checked,correct till here: check 3; shape: bs*w*h*1

	def reconstruct_silhouette_from_geoimg(self, image, cam_locs):
		with tf.variable_scope('g_reconstruct_silhouette') as scope:
			bs, w, h, c = image.get_shape().as_list()
			cam_locs = tf.convert_to_tensor(cam_locs, tf.float32)
			bs1, _ = cam_locs.get_shape().as_list()
			assert bs == bs1
			points = tf.reshape(image, [bs, w * h, c])

			def cross(a, b):
				# calculate the cross product
				# shape a: Bs x 3
				# shape b: Bs x 3
				cross_product = tf.cross(a, b)
				nm = tf.sqrt(tf.reduce_sum(cross_product * cross_product, axis=1, keep_dims=True))
				return cross_product / nm

			def get_up_direction(cam_locs):
				bs = np.shape(cam_locs)[0]
				ups = np.ones([bs, 3], dtype=np.float32) * np.array([0, 1, 0], dtype=np.float32)
				return ups

			# points :  [BxNx3]
			# cam_locs : [Bx3],point to the (0,0,0)
			points_shape = tf.shape(points)
			# bs = points.shape[0]

			N = points.shape[1]
			out_img_size = self.image_size
			# ---------------------------------------------------------------------------------------------#
			# subject mean of each dimension to ensure the center lies on the origin
			mean = tf.reduce_mean(points, axis=1)
			mean = tf.expand_dims(mean, axis=1)
			points = points - mean
			# ---------------------------------------------------------------------------------------------#
			# Guarantee the silhouettes rendered from the points generated by geometry image
			# match the ground truth after ordinary steps.
			sita_x = - np.pi / 2
			sita_y = 0
			sita_z = - np.pi / 2
			T_x = np.array([[1, 0, 0], [0, np.cos(sita_x), -np.sin(sita_x)], [0, np.sin(sita_x), np.cos(sita_x)]],
			               dtype=np.float32)
			T_y = np.array([[np.cos(sita_y), 0, np.sin(sita_y)], [0, 1, 0], [-np.sin(sita_y), 0, np.cos(sita_y)]],
			               dtype=np.float32)
			T_z = np.array([[np.cos(sita_z), -np.sin(sita_z), 0], [np.sin(sita_z), np.cos(sita_z), 0], [0, 0, 1]],
			               dtype=np.float32)
			T_x = tf.tile(T_x, [bs, 1])
			T_x = tf.reshape(T_x, [bs, 3, 3])
			T_y = tf.tile(T_y, [bs, 1])
			T_y = tf.reshape(T_y, [bs, 3, 3])
			T_z = tf.tile(T_z, [bs, 1])
			T_z = tf.reshape(T_z, [bs, 3, 3])

			points = tf.transpose(points, [0, 2, 1])  # Bsx3xN
			points = tf.matmul(T_z, points)  # revolve around z axis
			points = tf.matmul(T_y, points)  # revolve around y axis
			points = tf.matmul(T_x, points)  # revolve around x axis
			points = tf.transpose(points, [0, 2, 1])  # back to shape Bs x n x 3
			# return points        #have been chacked,correct till here:  check1
			# ---------------------------------------------------------------------------------------------#
			# map the points from world coordinate to a given camera coordinate
			points = points - tf.expand_dims(cam_locs, 1)  # translation from one world origin to camera origin
			up = get_up_direction(cam_locs)  # Bs x 3
			z_axis = - cam_locs  # Bs x 3
			x_axis = cross(up, z_axis)  # Have been normed, shape: Bs x 3
			y_axis = cross(z_axis, x_axis)  # Have been normed, shape: Bs x 3
			# camera_axis = np.concatenate([x_axis, y_axis, z_axis], 1)  # shape:Bs x 9
			camera_axis = tf.concat([x_axis, y_axis, z_axis], 1)  # shape:Bs x 9
			# camera_axis = np.reshape(camera_axis, [bs, 3, 3])  # shape:Bs x 3 x 3
			camera_axis = tf.reshape(camera_axis, [bs, 3, 3])  # shape:Bs x 3 x 3
			world_axis = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]], dtype=np.float32)  # shape:Bs x 3 x 3
			world_axis = np.tile(world_axis, [bs, 1, 1])  # shape:Bs x 3 x 3
			# rot_mat = np.matmul(world_axis, np.linalg.inv(camera_axis))  # bs x 3 x 3
			rot_mat = tf.matmul(world_axis, tf.matrix_inverse(camera_axis))  # bs x 3 x 3
			uvw = tf.matmul(points, rot_mat)  # scaled in (-1,1),shape :bs x n x 3
			uvw = uvw * out_img_size / 2  # scaled to out_imgsize/2 ,in the case the self.image_size is out_img_size
			# return uvw   #have been checked,correct till here: check 2
			# ---------------------------------------------------------------------------------------------#
			x, y, z = tf.split(uvw, 3, axis=2)  # each shape: bs x n x 1

			x = tf.reshape(x, [-1])  # bs x n
			y = tf.reshape(y, [-1])  # bs x n
			z = tf.reshape(z, [-1])  # bs x n

			scale_factor = 1
			x_int = tf.to_int32(tf.round(out_img_size / 2 - x * scale_factor))
			y_int = tf.to_int32(tf.round(out_img_size / 2 - y * scale_factor))
			depth_value = z  # bs x n
			# filter the out-range points
			# maskInside = (x_int>=0)&(x_int<128)

			batchIdx, _ = np.meshgrid(range(bs), range(64 * 64), indexing="ij")  # geometry image size
			batchIdx = batchIdx.reshape([-1])
			scatterIdx = tf.stack([batchIdx, y_int, x_int], axis=1)
			scatterContent = depth_value
			scatterShape = tf.constant([bs, out_img_size, out_img_size])  # silhouette image size
			newContent = tf.scatter_nd(scatterIdx, scatterContent, scatterShape)  # bs*w*h
			# newContent = tf.minimum(newContent, 255.0)  # scale pixel value to (0,255)
			newContent = tf.expand_dims(newContent, axis=3)

			# ---------------------------------------------------------------------------------------------#
			# filter the noise point from the reconstructed
			# newContent = slim.conv2d(newContent,32,[2, 2],stride=1,scope='smooth_conv1')
			# newContent = slim.conv2d(newContent, 1, [2, 2], stride=1, scope='smooth_conv2')
			# ---------------------------------------------------------------------------------------------#
			newContent = tf.minimum(newContent, 255.0)  # scale pixel value to (0,255)
			return newContent / 255.0  # have been checked,correct till here: check 3; shape: bs*w*h*1

	def get_test_input(self):
		images = np.zeros((self.batch_size, self.image_size, self.image_size, self.input_c_dim + self.depth_num))
		camera_pose = []
		weighted_mask = np.zeros((self.batch_size, self.image_size, self.image_size, 1))
		for i in range(0, self.batch_size):
			cate_idx = i % len(self.cates)
			cate = self.cates[cate_idx]
			# idx = np.random.randint(981, 989 + 1)
			idx = i + 1
			geo_imgs = np.random.random((1, self.image_size, self.image_size, 3))  # original geometry scaled in 0 - 1
			images[i, :, :, :self.input_c_dim] = geo_imgs
			k = self.test_sample_idx[self.cate]
			start = 3
			img_name = '{}_{:04d}_{:d}.jpg'.format(cate, idx, k)
			file = os.path.join(self.depth_dir_test[cate],
			                    img_name)  # both train and test data are in 'train' director.
			depth_img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (self.image_size, self.image_size))
			images[i, :, :, start] = 1 - (depth_img / 255.0)
			mask = np.ones((self.image_size, self.image_size, 1))
			mask[images[i, :, :, start] != 0] = 3
			weighted_mask[i, :, :, :] = mask
			start += 1
			camera_pose.append(self.camera_poses[k])
		# images[20:self.batch_size,:,:,:] = np.random.random([self.batch_size-20,self.image_size, self.image_size, self.input_c_dim + self.depth_num])
		# scaled image data
		camera_pose = np.stack(camera_pose, 0)

		return images, camera_pose, weighted_mask

	def get_input(self):
		images = np.zeros((self.batch_size, self.image_size, self.image_size, self.input_c_dim + self.depth_num))
		geo_imgs_32 = np.zeros((self.batch_size, 32, 32, 3))
		geo_imgs_16 = np.zeros((self.batch_size, 16, 16, 3))
		normals_64 = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
		normals_32 = np.zeros((self.batch_size, 32, 32, 3))
		normals_16 = np.zeros((self.batch_size, 16, 16, 3))
		camera_pose = []
		sil_aux_gt = np.zeros((self.batch_size * 18, self.image_size, self.image_size, 1))
		weighted_mask = np.zeros((self.batch_size, self.image_size, self.image_size, 1))
		labels = np.zeros((self.batch_size), np.int32)
		for i in range(self.batch_size):
			cate_idx = np.random.randint(0, len(self.cates))
			cate = self.cates[cate_idx]
			rd_idx = np.random.randint(0, self.mat_data_size[cate])
			img_file = self.files_in_depth_dir[rd_idx]

			idx = int(img_file.split('_')[1])
			pose_idx = img_file.split('.')[0]
			pose_idx = int(pose_idx.split('_')[-1])
			labels[i] = self.cate2label[cate]

			geo_imgs = self.mat_data[cate][idx - 1] * 2  # original geometry scaled in 0 - 0.5
			images[i, :, :, :self.input_c_dim] = geo_imgs
			geo_imgs_32[i] = self.mat_data_32[cate][idx - 1] * 2
			geo_imgs_16[i] = self.mat_data_16[cate][idx - 1] * 2
			normals_64[i, :, :, :] = self.mat_normal[cate][idx - 1]
			normals_32[i, :, :, :] = self.mat_normal_32[cate][idx - 1]
			normals_16[i, :, :, :] = self.mat_normal_16[cate][idx - 1]

			start = 3

			file = os.path.join(self.depth_dir[cate], img_file)

			try:
				depth_img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (self.image_size, self.image_size))
			except Exception as ex:
				print('error occurs at ', file)
			# print('haha ',file)
			images[i, :, :, start] = 1 - (depth_img / 255.0)
			mask = np.ones((self.image_size, self.image_size, 1))
			mask[images[i, :, :, start] != 0] = 3
			weighted_mask[i, :, :, :] = mask
			start += 1
			camera_pose.append(self.camera_poses[pose_idx])

			# -----------------------------------------------------------------------#
			id = 0
			for k in self.view_points:
				img_name = '{}_{:04d}_{:d}.jpg'.format(cate, idx, k)
				file = os.path.join(self.depth_dir_old[cate], img_name)
				try:
					depth_img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (self.image_size, self.image_size))
				except Exception as ex:
					print('error occurs at ', file)
				sil_aux_gt[i * 18 + id, :, :, 0] = 1 - (depth_img / 255.0)
				id += 1
		# -----------------------------------------------------------------------#

		camera_pose = np.stack(camera_pose, 0)

		# scaled image data
		return images, camera_pose, sil_aux_gt, weighted_mask, normals_64, normals_32, normals_16, geo_imgs_32, geo_imgs_16, labels

	# return images, camera_pose
	def train(self, args):
		"""Train pix2pix"""
		# d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
		# g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

		optim_d = tf.train.AdamOptimizer(args.lr,beta1=args.beta1)
		optim_g = tf.train.AdamOptimizer(args.lr,beta1=args.beta1)

		g_output1_optim = tf.train.AdamOptimizer(args.lr,beta1=args.beta1).minimize(self.g_loss_for_output1)
		g_output2_optim = tf.train.AdamOptimizer(args.lr,beta1=args.beta1).minimize(self.g_loss_for_output2)

		self.grads_and_vars_d = optim_d.compute_gradients(self.d_loss, var_list=self.d_vars)
		self.grads_and_vars_g = optim_g.compute_gradients(self.g_loss, var_list=self.g_vars)

		for var in self.g_vars:
			print(var.name)
		'''
		gradients = []
		vars = []
		for grad ,var in self.grads_and_vars_g:
			if 'g_estimate_pose' in var.op.name:
				grad *= 0.1
			gradients.append(grad)
			vars.append(var)
		self.grads_and_vars_g = zip(gradients,vars)'''

		d_optim = optim_d.apply_gradients(self.grads_and_vars_d)
		g_optim = optim_g.apply_gradients(self.grads_and_vars_g)

		for grad, var in self.grads_and_vars_d:
			# print(var.op.name)
			tf.summary.scalar(var.op.name + '/d_gradients', tf.reduce_mean(tf.abs(grad)))

		for grad, var in self.grads_and_vars_g:
			# print(var.op.name)
			tf.summary.scalar(var.op.name + '/g_gradients', tf.reduce_mean(tf.abs(grad)))

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		# self.g_sum = tf.summary.merge([self.d__sum,self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
		# self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

		self.summary_op = tf.summary.merge_all()

		self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

		counter = 1
		start_time = time.time()

		if args.continue_train:
			pre_ckpt_dir = 'experiments/ckpt/' + 'ckpt_10_2' + '_' + self.cates[0]
			if self.load_for_continue_train(pre_ckpt_dir):
				print(" [*] Load SUCCESS")
			else:
				print(" [!] Load failed...train from scratch")

		# restore the checkpoint of PoseNet
		# self.posenet_saver.restore(self.sess,'./PoseNet10/pose_est_model-15001')
		# print('restore PoseNet checkpoint succesfully')

		for epoch in xrange(300,args.epoch):
			# data = glob('./data/{}/train/*.jpg'.format(self.dataset_name))
			# np.random.shuffle(data)
			batch_idxs = min(self.sample_number, args.train_size) // self.batch_size

			for idx in xrange(0, batch_idxs):

				batch_images, cam_poses, sil_aux, mask, batch_normal, batch_normal_32, batch_normal_16, batch_geoimg_32, batch_geoimg_16, labels = self.get_input()

				if epoch < 150:
					_,g_loss_1,chamfer_loss_1,edge_loss_1,normal_loss_1 = self.sess.run([g_output1_optim,
					                                                                     self.g_loss_for_output1,
					                                                                     self.chamfer_loss1,
					                                                                     self.edge_distance_loss1,
					                                                                     self.normal_loss1],

					                  feed_dict={self.real_data: batch_images, self.camera_pose_gt: cam_poses,
					                             self.sil_aux_gt: sil_aux, self.weighted_mask: mask,
					                             self.gt_normal: batch_normal, self.init_geo_img: self.mat,
					                             self.real_B_32: batch_geoimg_32, self.real_B_16: batch_geoimg_16,
					                             self.gt_normal_32: batch_normal_32, self.gt_normal_16: batch_normal_16,
					                             self.geoimg_labels: labels})

					print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss1: %.4f, CD_loss1: %.4f, edge_loss1: %.4f, normal_loss1: %.4f" \
						% (epoch, idx, batch_idxs, time.time() - start_time, g_loss_1,chamfer_loss_1,edge_loss_1,normal_loss_1))
					counter += 1
					if np.mod(counter, 200) == 0:
						test_images, camera_poses, mask = self.get_test_input()
						# training_sample = self.sess.run(self.fake_B,feed_dict={self.real_data: batch_images})
						training_sample = self.sess.run(self.fake_output1, feed_dict={self.real_data: test_images,
						                                                        self.camera_pose_gt: cam_poses,
						                                                        self.weighted_mask: mask,
						                                                        self.init_geo_img: self.mat})
						scio.savemat('./{}/test_sample_op1_{:02d}_{:04d}.mat'.format(self.sample_dir, epoch, idx),
						             {'geoimgs': training_sample[:32]})


				elif epoch < 300:
					_, g_loss_2, chamfer_loss_2, edge_loss_2, normal_loss_2 = self.sess.run([g_output2_optim,
					                                                                         self.g_loss_for_output2,
					                                                                         self.chamfer_loss2,
					                                                                         self.edge_distance_loss2,
					                                                                         self.normal_loss2],

					                                                                        feed_dict={
						                                                                        self.real_data: batch_images,
						                                                                        self.camera_pose_gt: cam_poses,
						                                                                        self.sil_aux_gt: sil_aux,
						                                                                        self.weighted_mask: mask,
						                                                                        self.gt_normal: batch_normal,
						                                                                        self.init_geo_img: self.mat,
						                                                                        self.real_B_32: batch_geoimg_32,
						                                                                        self.real_B_16: batch_geoimg_16,
						                                                                        self.gt_normal_32: batch_normal_32,
						                                                                        self.gt_normal_16: batch_normal_16,
						                                                                        self.geoimg_labels: labels})

					print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss2: %.4f, CD_loss2: %.4f, edge_loss2: %.4f, normal_loss1: %.4f" \
						% (epoch, idx, batch_idxs, time.time() - start_time, g_loss_2, chamfer_loss_2, edge_loss_2,normal_loss_2))
					counter += 1
					if np.mod(counter, 200) == 0:
						test_images, camera_poses, mask = self.get_test_input()
						# training_sample = self.sess.run(self.fake_B,feed_dict={self.real_data: batch_images})
						training_sample = self.sess.run(self.fake_output2, feed_dict={self.real_data: test_images,
						                                                        self.camera_pose_gt: cam_poses,
						                                                        self.weighted_mask: mask,
						                                                        self.init_geo_img: self.mat})
						scio.savemat('./{}/test_sample_op2_{:02d}_{:04d}.mat'.format(self.sample_dir, epoch, idx),
						             {'geoimgs': training_sample[:32]})
				else:

					# Update D network
					_ = self.sess.run([d_optim],
					                  feed_dict={self.real_data: batch_images, self.camera_pose_gt: cam_poses,
					                             self.sil_aux_gt: sil_aux, self.weighted_mask: mask,
					                             self.init_geo_img: self.mat, self.geoimg_labels: labels})

					# Update G network
					_ = self.sess.run([g_optim],
					                  feed_dict={self.real_data: batch_images, self.camera_pose_gt: cam_poses,
					                             self.sil_aux_gt: sil_aux, self.weighted_mask: mask,
					                             self.gt_normal: batch_normal, self.init_geo_img: self.mat,
					                             self.real_B_32: batch_geoimg_32, self.real_B_16: batch_geoimg_16,
					                             self.gt_normal_32: batch_normal_32, self.gt_normal_16: batch_normal_16,
					                             self.geoimg_labels: labels})

					# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
					_ = self.sess.run([g_optim],
					                  feed_dict={self.real_data: batch_images, self.camera_pose_gt: cam_poses,
					                             self.sil_aux_gt: sil_aux, self.weighted_mask: mask,
					                             self.gt_normal: batch_normal, self.init_geo_img: self.mat,
					                             self.real_B_32: batch_geoimg_32, self.real_B_16: batch_geoimg_16,
					                             self.gt_normal_32: batch_normal_32, self.gt_normal_16: batch_normal_16,
					                             self.geoimg_labels: labels})

					if counter % 10 == 1:
						summary_str = self.sess.run(self.summary_op, feed_dict={self.real_data: batch_images,
						                                                        self.camera_pose_gt: cam_poses,
						                                                        self.sil_aux_gt: sil_aux,
						                                                        self.weighted_mask: mask,
						                                                        self.gt_normal: batch_normal,
						                                                        self.init_geo_img: self.mat,
						                                                        self.real_B_32: batch_geoimg_32,
						                                                        self.real_B_16: batch_geoimg_16,
						                                                        self.gt_normal_32: batch_normal_32,
						                                                        self.gt_normal_16: batch_normal_16,
						                                                        self.geoimg_labels: labels})

						self.writer.add_summary(summary_str, counter)

					errD_fake, errD_real, errG, errL2, pose_est_loss, sil_reconstruct_loss, pose, sil_loss_aux, edge_loss, normal_loss = self.sess.run(
						[self.d_loss_fake, self.d_loss_real, self.g_loss, self.l2_loss,
						 self.pose_estimate_loss, self.reconstruct_sil_loss, self.camera_pose_pre,
						 self.reconstruct_sil_loss_aux, self.edge_distance_loss, self.normal_loss],
						{self.real_data: batch_images, self.camera_pose_gt: cam_poses, self.sil_aux_gt: sil_aux,
						 self.weighted_mask: mask, self.gt_normal: batch_normal, self.init_geo_img: self.mat,
						 self.real_B_32: batch_geoimg_32, self.real_B_16: batch_geoimg_16,
						 self.gt_normal_32: batch_normal_32, self.gt_normal_16: batch_normal_16,
						 self.geoimg_labels: labels})


					if self.use_chamfer:
						print(
							"Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, chamfer_loss: %.5f, "
							"pose_est_loss: %.5f, sil_recsturct_loss: %.5f,est_local_loss: %.5f,edge_loss: %.5f, normal_loss: %.5f, cate:%s, log:%s" \
							% (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG, errL2,
							   pose_est_loss, sil_reconstruct_loss, sil_loss_aux, edge_loss, normal_loss, self.cate,
							   self.checkpoint_dir.split('/')[-1]))

					else:
						print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, l2_loss: %.5f, "
						      "pose_est_loss: %.5f, sil_recsturct_loss: %.5f,est_local_loss: %.5f,edge_loss: %.5f, normal_loss: %.5f, cate:%s, log:%s" \
						      % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG, errL2,
						         pose_est_loss, sil_reconstruct_loss, sil_loss_aux, edge_loss, normal_loss, self.cate,
						         self.checkpoint_dir.split('/')[-1]))
					counter += 1
					# print('predict pose is ')
					# print(pose[:3]),
					# print('\ngt pose is ')
					# print(cam_poses[:3])
					if counter % 100 == 1:
						reconstructed_sil, gt_sil = self.sess.run(
							[self.reconstructed_silhouette, self.real_A],
							{self.real_data: batch_images, self.camera_pose_gt: cam_poses, self.init_geo_img: self.mat})

						pred_and_gt = np.concatenate([reconstructed_sil, gt_sil], 1)
						scio.savemat('{:s}/pred_and_gt_sil_{:d}.mat'.format(self.sample_dir, epoch),
						             {'geoimgs': pred_and_gt[:32]})

					if np.mod(counter, 200) == 0:
						test_images, camera_poses, mask = self.get_test_input()
						# training_sample = self.sess.run(self.fake_B,feed_dict={self.real_data: batch_images})
						training_sample = self.sess.run(self.fake_B, feed_dict={self.real_data: test_images,
						                                                        self.camera_pose_gt: cam_poses,
						                                                        self.weighted_mask: mask,
						                                                        self.init_geo_img: self.mat})
						scio.savemat('./{}/test_sample_{:02d}_{:04d}.mat'.format(self.sample_dir, epoch, idx),
						             {'geoimgs': training_sample[:32]})
						reconstructed_sil, gt_sil = self.sess.run(
							[self.reconstructed_silhouette, self.real_A],
							{self.real_data: batch_images, self.camera_pose_gt: cam_poses, self.sil_aux_gt: sil_aux,
							 self.init_geo_img: self.mat})

				if np.mod(counter, 500) == 2 and counter > 500:
					self.save(self.checkpoint_dir, counter)

	def estimate_pose(self, image):
		'''
		with tf.variable_scope('g_estimate_pose') as scope:
			net = slim.conv2d(image,32,[3,3],stride=2,scope ='conv1')  # 64 x 64
			net = slim.conv2d(net, 32, [3, 3], stride=2, scope='conv2')  # 32 x 32
			net = slim.conv2d(net,64,[3,3],stride=2,scope='conv3')   # 16 x 16
			net = slim.conv2d(net,64,[3,3],stride=2,scope='conv4')   # 8 x 8
			net = slim.conv2d(net,64,[3,3],stride=2,scope='conv5')  # 4 x 4
			net = tf.reshape(net,[self.batch_size,-1])
			net = slim.fully_connected(net,256,scope='fc1')
			net = slim.fully_connected(net,3,activation_fn=tf.nn.tanh,scope='fc2')
			return net
		'''
		with tf.variable_scope('g_estimate_pose') as scope:
			net = slim.conv2d(image, 32, [3, 3], stride=1, scope='conv1')  # 64 x 64
			net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pooling1')
			net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv2')  # 32 x 32
			net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pooling2')
			net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv3')  # 16 x 16
			net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pooling3')
			net = slim.conv2d(net, 192, [3, 3], stride=1, scope='conv4')  # 8 x 8
			net = slim.conv2d(net, 192, [3, 3], stride=1, scope='conv5')
			net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pooling4')  # 4 x 4
			net = tf.reshape(net, [self.batch_size, -1])
			net = slim.fully_connected(net, 512, scope='fc1')
			net = slim.fully_connected(net, 256, scope='fc2')
			net = slim.fully_connected(net, 3, activation_fn=tf.nn.tanh, scope='fc3')
			norm = tf.sqrt(tf.reduce_sum(net * net, axis=1, keep_dims=True))
			net = net / norm
			return net

	def my_discriminator(self, image, y=None, reuse=False):

		with tf.variable_scope("my_discriminator") as scope:

			# image is 128 x 128 x input_c_dim
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False

			# h0 is (128 x 128 x self.df_dim)
			f_conv3 = slim.conv2d(image, 32, [3, 3], padding='VALID', scope='my_conv3')
			f_conv5 = slim.conv2d(image, 32, [5, 5], padding='VALID', scope='my_conv5')
			f_conv7 = slim.conv2d(image, 32, [7, 7], padding='VALID', scope='my_conv7')

			return f_conv3, f_conv5, f_conv7

	# accuracy: 0.9076
	def discriminator(self, image, y=None, reuse=False):
		with tf.variable_scope("d_discriminator") as scope:
			# image is 128 x 128 x input_c_dim
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False

			bs, w, h, c = image.shape

			# print('batch size is ', bs)
			# extract feature at a receptive field of 3x3
			net1 = slim.conv2d(image, 32, [3, 3], scope='conv1')
			net1 = slim.conv2d(net1, 128, [3, 3], scope='conv2')
			net1 = slim.conv2d(net1, 128, [1, 1], scope='conv3')
			net1 = tf.concat([image, net1], axis=3)

			net1_rs = tf.reshape(net1, [-1, w * h, 131])  # bs * 16384 * 131
			net1_rs = tf.expand_dims(net1_rs, axis=2)  # bs * 16384 * 1 * 131

			# local feature selection
			net1 = slim.conv2d(net1, 64, [3, 3], scope='conv4')
			net1_max_pooling = slim.max_pool2d(net1, [3, 3], stride=1, padding='SAME')
			net1_max_pooling_rs = tf.reshape(net1_max_pooling, [-1, w * h, 64])  # bs * 16384 * 64

			net1_max_pooling_rs = tf.expand_dims(net1_max_pooling_rs, axis=2)  # bs * 16384 * 1 * 64

			net1_rs = slim.conv2d(net1_rs, 64, [1, 1], padding='VALID', scope='conv5')  # bs * 16384 * 1 * 64
			net1_rs = slim.conv2d(net1_rs, 128, [1, 1], padding='VALID', scope='conv6')  # bs * 16384 * 1 * 128

			net1 = tf.concat([net1_rs, net1_max_pooling_rs], axis=3)  # bs * 16384 * 1 * 192
			# print('net1 shape ', np.shape(net1))
			net1 = slim.conv2d(net1, 256, [1, 1], scope='conv7')  # bs * 16384 * 1 * 512
			net1 = slim.conv2d(net1, 512, [1, 1], scope='conv8')  # bs * 16384 * 1 * 512
			# net1 = slim.max_pool2d(net1,[w*h,1],stride=1,padding='VALID',scope='global_max_pooling') # bs * 1 * 1 * 512
			net1 = tf.nn.max_pool(net1, ksize=[1, w * h, 1, 1], strides=[1, 2, 2, 1], padding='VALID')

			net1 = tf.squeeze(net1, axis=[1, 2])  # bs * 512

			net = net1
			net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='fc1')

			if self.use_amgan:
				class_logit = slim.fully_connected(net, len(self.cates) + 2, activation_fn=None, scope='fc2')
				return class_logit[:, :len(self.cates)], class_logit[:, -2:-1], class_logit[:, -1:]
			else:
				net = slim.fully_connected(net, 1, activation_fn=None, scope='fc2')
				return tf.nn.sigmoid(net), net, net1

	

	def discriminator_pointnet(self, image, y=None, reuse=False):
		print('Using PointNet as discriminator')
		with tf.variable_scope("d_discriminator_pointnet") as scope:
			# image is 128 x 128 x input_c_dim
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False

			point_cloud = tf.reshape(image, (24, 64 * 64, 3))

			batch_size = point_cloud.get_shape()[0].value
			num_point = point_cloud.get_shape()[1].value
			end_points = {}

			is_training = tf.cast(True, dtype=tf.bool)
			bn_decay = None
			is_bn = False

			point_cloud_transformed = point_cloud
			input_image = tf.expand_dims(point_cloud_transformed, -1)

			net = tf_util.conv2d(input_image, 64, [1, 3],
			                     padding='VALID', stride=[1, 1],
			                     bn=is_bn, is_training=is_training,
			                     scope='conv1', bn_decay=bn_decay)
			net = tf_util.conv2d(net, 64, [1, 1],
			                     padding='VALID', stride=[1, 1],
			                     bn=is_bn, is_training=is_training,
			                     scope='conv2', bn_decay=bn_decay)

			net = tf_util.conv2d(net, 64, [1, 1],
			                     padding='VALID', stride=[1, 1],
			                     bn=is_bn, is_training=is_training,
			                     scope='conv3', bn_decay=bn_decay)
			net = tf_util.conv2d(net, 128, [1, 1],
			                     padding='VALID', stride=[1, 1],
			                     bn=is_bn, is_training=is_training,
			                     scope='conv4', bn_decay=bn_decay)
			net = tf_util.conv2d(net, 1024, [1, 1],
			                     padding='VALID', stride=[1, 1],
			                     bn=is_bn, is_training=is_training,
			                     scope='conv5', bn_decay=bn_decay)

			# Symmetric function: max pooling
			net = tf_util.max_pool2d(net, [num_point, 1],
			                         padding='VALID', scope='maxpool')

			net = tf.reshape(net, [batch_size, -1])
			net1 = net
			net = tf_util.fully_connected(net, 512, bn=is_bn, is_training=is_training,
			                              scope='fc1', bn_decay=bn_decay)
			net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
			                      scope='dp1')
			net = tf_util.fully_connected(net, 256, bn=is_bn, is_training=is_training,
			                              scope='fc2', bn_decay=bn_decay)
			net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
			                      scope='dp2')
			net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')

			return tf.nn.sigmoid(net), net, net1

	def discriminator_res(self, image, y=None, reuse=False):
		pass

	def generator(self, image, pose, y=None):
		with tf.variable_scope("g_generator") as scope:
			s = self.output_size
			s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
				s / 64), int(s / 128)

			# extract feature from pose
			pose = slim.fully_connected(pose, self.gf_dim * 2, scope='g_pose_fc1')
			pose = slim.fully_connected(pose, self.gf_dim, scope='g_pose_fc2')
			pose = tf.reshape(pose, [-1, 1, 1, self.gf_dim])

			# ------------------------------------------------------------------- #
			# encoder
			# image is (64 x 64 x input_c_dim)
			e1 = slim.conv2d(image, self.gf_dim * 2, kernel_size=[3, 3], stride=1, scope='g_e1_conv1')
			e1 = slim.conv2d(e1, self.gf_dim * 2, kernel_size=[3, 3], stride=1, scope='g_e1_conv2')
			e1 = slim.conv2d(e1, self.gf_dim * 4, kernel_size=[3, 3], stride=1, scope='g_e1_conv3')
			e1 = slim.max_pool2d(e1, kernel_size=[2, 2], stride=2, padding='SAME', scope='g_e1_maxpooling')
			# e1 is (32 x 32 x self.gf_dim*2)

			e2 = slim.conv2d(e1, self.gf_dim * 4, kernel_size=[3, 3], stride=1, scope='g_e2_conv1')
			e2 = slim.conv2d(e2, self.gf_dim * 4, kernel_size=[3, 3], stride=1, scope='g_e2_conv2')
			e2 = slim.conv2d(e2, self.gf_dim * 8, kernel_size=[3, 3], stride=1, scope='g_e2_conv3')
			e2 = slim.max_pool2d(e2, kernel_size=[2, 2], stride=2, padding='SAME', scope='g_e2_maxpooling')
			# e2 is (16 x 16 x self.gf_dim*4)

			e3 = slim.conv2d(e2, self.gf_dim * 8, kernel_size=[3, 3], stride=1, scope='g_e3_conv1')
			e3 = slim.conv2d(e3, self.gf_dim * 8, kernel_size=[3, 3], stride=1, scope='g_e3_conv2')
			e3 = slim.conv2d(e3, self.gf_dim * 8, kernel_size=[3, 3], stride=1, scope='g_e3_conv3')

			e4 = e3

			print('e4 shape is ', np.shape(e4))
			# e4 is (8 x 8 x self.gf_dim*8)
			e4_shape = e4.get_shape()
			pose_shape = pose.get_shape()
			e4 = tf.concat([e4, pose * tf.ones([e4_shape[0], e4_shape[1], e4_shape[2], pose_shape[3]])], 3)

			e4 = slim.conv2d(e4, self.gf_dim * 8, kernel_size=[1, 1], stride=1, scope='g_e4_conv1')
			e4 = slim.conv2d(e4, self.gf_dim * 16, kernel_size=[1, 1], stride=1, scope='g_e4_conv2')

			e5 = e4
			print('e5 shape is ', np.shape(e5))
			# e5 shape is (bs,16,16,224)
			# e5 = deconv2d(e4,[self.batch_size,s8,s8,self.gf_dim * 4-3],name='g_d0')
			# e5 shape is (bs,16,16,125)
			# choose 125 instead of 128 because of some tensorflow bug when use residual_block dealing
			# with feature map with 131 channels
			# decoder

			self.init_geo_img_concat = tf.concat((self.init_geo_img, e5), 3)
			self.init_geo_img_concat = tf.nn.relu(
				slim.conv2d(self.init_geo_img_concat, self.gf_dim * 3, [1, 1], activation_fn=None,
				            scope='g_conv_debug2'))
			self.decoder_block1 = tflearn.residual_block(self.init_geo_img_concat, 6, self.gf_dim * 3, batch_norm=False,
			                                             name='g_up_block1')
			# self.output1 = tflearn.residual_block(self.decoder_block1,2,3,batch_norm=True,name='g_output1',activation='sigmoid')
			self.output1 = slim.conv2d(self.decoder_block1, 3, [3, 3], activation_fn=tf.nn.sigmoid, scope='g_output1')
			# output1 shape is (bs,16,16,3)
			print('output1 shape is ', np.shape(self.output1))

			# deconv1 in decoder
			self.decoder_block1_upsample = deconv2d(self.output1, [self.batch_size, s4, s4, self.gf_dim * 4],name='g_d1')
			self.decoder_block1_upsample_bil = tf.image.resize_bilinear(self.output1,(32,32),align_corners=True)


			self.decoder_block1_concat = tf.concat((self.decoder_block1_upsample, e1), 3)
			# self.decoder_block1_concat = tf.nn.relu(tflearn.batch_normalization(slim.conv2d(self.decoder_block1_concat,self.gf_dim*4,[1,1],activation_fn=None,scope='g_conv_debug3')))
			self.decoder_block1_concat = tf.nn.relu(
				slim.conv2d(self.decoder_block1_concat, self.gf_dim * 4, [1, 1], activation_fn=None,
				            scope='g_conv_debug3'))

			self.decoder_block2 = tflearn.residual_block(self.decoder_block1_concat, 5, self.gf_dim * 4,
			                                             batch_norm=False, name='g_up_block2')
			# self.output2 = tflearn.residual_block(self.decoder_block2,2,3,batch_norm=True,name='g_output2',activation='sigmoid')
			self.output2 = slim.conv2d(self.decoder_block2, 3, [3, 3], activation_fn=tf.nn.sigmoid, scope='g_output2') * 0.5 \
						+ self.decoder_block1_upsample_bil * 0.5
			# decoder_block2 shape is (bs,32,32,3)

			# deconv2 in decoder
			self.decoder_block2_upsample = deconv2d(self.output2, [self.batch_size, s2, s2, self.gf_dim * 4],name='g_d2')
			self.decoder_block2_upsample_bil = tf.image.resize_bilinear(self.output2,(64,64),align_corners=False)


			self.decoder_block2_concat = tf.concat((self.decoder_block2_upsample, image), 3)
			self.decoder_block2_concat = tf.nn.relu(
				slim.conv2d(self.decoder_block2_concat, self.gf_dim * 4, [1, 1], activation_fn=None,
				            scope='g_conv_debug4'))
			self.decoder_block3 = tflearn.residual_block(self.decoder_block2_concat, 5, self.gf_dim * 4,
			                                             batch_norm=False, name='g_up_block3')
			# decoder_block3 shape is (bs,64,64,64)

			# self.decoder_block4 = tflearn.residual_block(self.decoder_block3,5,self.gf_dim,batch_norm=True,name='g_up_block4')

			self.output3 = slim.conv2d(self.decoder_block3, 3, [3, 3], activation_fn=tf.nn.sigmoid, scope='g_final_output') * 0.5 \
						+ self.decoder_block2_upsample_bil * 0.5

			return self.output3, self.output2, self.output1

	def save(self, checkpoint_dir, step):
		model_name = "pix2pix.model"
		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
		                os.path.join(checkpoint_dir, model_name),
		                global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		print('checkpoint dir {:s}'.format(checkpoint_dir))
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		restore_saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if 'd_' not in var.name])
		
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			print('Continue train: restore from {:s}'.format(os.path.join(checkpoint_dir, ckpt_name)))
			restore_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def load_for_continue_train(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		print('checkpoint dir {:s}'.format(checkpoint_dir))
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

		restore_saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if 'd_' not in var.name])
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			print('Continue train: restore from {:s}'.format(os.path.join(checkpoint_dir, ckpt_name)))
			restore_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def get_test_input1(self, id, cate):
		images = np.zeros((self.batch_size, self.image_size, self.image_size, self.input_c_dim + self.depth_num))
		camera_pose = []
		weighted_mask = np.zeros((self.batch_size, self.image_size, self.image_size, 1))
		for i in range(0, self.batch_size):
			# idx = np.random.randint(981, 989 + 1)
			idx = i + 1
			geo_imgs = np.random.random((1, self.image_size, self.image_size, 3))  # original geometry scaled in 0 - 1
			images[i, :, :, :self.input_c_dim] = geo_imgs
			k = self.test_sample_idx[self.cate]
			start = 3
			img_name = '{}_{:04d}_{:d}.jpg'.format(self.dataset_name, idx, k)
			file = os.path.join(self.depth_dir_test[cate],
			                    img_name)  # both train and test data are in 'train' director.
			# print(file)
			depth_img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (self.image_size, self.image_size))
			images[i, :, :, start] = 1 - (depth_img / 255.0)
			mask = np.ones((self.image_size, self.image_size, 1))
			mask[images[i, :, :, start] != 0] = 3
			weighted_mask[i, :, :, :] = mask
			start += 1
			camera_pose.append(self.camera_poses[k])
		# images[20:self.batch_size,:,:,:] = np.random.random([self.batch_size-20,self.image_size, self.image_size, self.input_c_dim + self.depth_num])
		# scaled image data
		camera_pose = np.stack(camera_pose, 0)

		img_name = '{}_{:04d}_{:d}.jpg'.format(self.dataset_name, id, k)
		file = os.path.join(self.depth_dir_test[cate], img_name)  # both train and test data are in 'train' director.
		# print(file)
		depth_img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (self.image_size, self.image_size))
		images[0, :, :, 3] = 1 - (depth_img / 255.0)

		return images, camera_pose, weighted_mask

	def test(self, args):
		print('Test mode')
		for i in range(len(self.cates)):
			cate = self.cates[i]

			if self.load(self.checkpoint_dir):
				print(" [*]{:s} Load SUCCESS".format(cate))
				test_samples = np.zeros((self.mat_data_size_test[cate], self.image_size, self.image_size, 3))
				for i in range(self.mat_data_size_test[cate]):
					print('{:d}th test sample'.format(i))
					batch_images, camera_poses, _ = self.get_test_input1(i + 1, cate)
					test_sample = self.sess.run(self.fake_B, feed_dict={self.real_data: batch_images,
					                                                    self.camera_pose_gt: camera_poses,
					                                                    self.init_geo_img: self.mat})
					test_samples[i, :, :, :] = test_sample[0, :, :, :]
				# print('haha test sample shape is ',np.shape(test_sample))

				scio.savemat('results/{:s}_{:s}_test_samples.mat'.format(self.experiment_name, cate),
				             {'geoimgs': test_samples})

			else:
				print('load failed')

	def padding_for_geoimg_old(self,tensor):
		bs, w, h, c = tensor.get_shape().as_list()
		# print('------------batch idx---------------')
		batchIdx, _ = np.meshgrid(range(bs), range(w * h * c), indexing="ij")
		batchIdx = batchIdx.reshape([-1])

		# print('------------yy---------------')
		y, _ = np.meshgrid(range(h), range(h * c), indexing="ij")
		y = y.reshape([-1]) + 1
		yy = []
		for i in range(bs):
			yy.extend(y)

		# print('------------xx---------------')
		x, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x = x.reshape([-1]) + 1
		xx = []
		for i in range(bs * w):
			xx.extend(x)

		# print('------------chanel idx---------------')
		c_idx = range(c)
		chanel_idx = []
		for i in range(bs * w * h):
			chanel_idx.extend(c_idx)

		# print('------------scatter idx---------------')
		scatterIdx = tf.stack([tf.cast(batchIdx,tf.int32), yy, xx, chanel_idx], axis=1)
		scatterContent = tf.reshape(tensor, [-1])
		scatterShape = tf.constant([bs, w + 2, h + 2, c])
		newContent = tf.scatter_nd(scatterIdx, scatterContent, scatterShape)  # padding with zero

		# ------------padding with reverse edge elements------------#
		# left colume
		slice1 = tf.slice(tensor, [0, 0, 0, 0], [bs, h, 1, c])  # left colume
		slice1 = tf.reverse(slice1, [1])
		batchIdx1, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx1 = batchIdx1.reshape([-1])
		y1, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y1 = y1.reshape([-1])
		yy1 = []
		for i in range(bs):
			yy1.extend(y1)
		x1, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x1 = x1.reshape([-1]) + 1
		xx1 = []
		for i in range(bs * 1):
			xx1.extend(x1)
		c_idx1 = range(c)
		chanel_idx1 = []
		for i in range(bs * 1 * h):
			chanel_idx1.extend(c_idx1)
		scatterIdx1 = tf.stack([tf.cast(batchIdx1,tf.int32), xx1, yy1, chanel_idx1], axis=1)
		slice1 = tf.reshape(slice1, [-1])
		delta1 = tf.SparseTensor(tf.cast(scatterIdx1, tf.int64), slice1, tf.cast(tf.shape(newContent), tf.int64))

		# right colume
		slice2 = tf.slice(tensor, [0, 0, w - 1, 0], [bs, h, 1, c])  # right colume
		slice2 = tf.reverse(slice2, [1])
		batchIdx2, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx2 = batchIdx2.reshape([-1])
		y2, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y2 = y2.reshape([-1]) + w + 1
		yy2 = []
		for i in range(bs):
			yy2.extend(y2)
		x2, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x2 = x2.reshape([-1]) + 1
		xx2 = []
		for i in range(bs * 1):
			xx2.extend(x2)
		c_idx2 = range(c)
		chanel_idx2 = []
		for i in range(bs * 1 * h):
			chanel_idx2.extend(c_idx2)
		scatterIdx2 = tf.stack([tf.cast(batchIdx2,tf.int32), xx2, yy2, chanel_idx2], axis=1)
		slice2 = tf.reshape(slice2, [-1])
		delta2 = tf.SparseTensor(tf.cast(scatterIdx2, tf.int64), slice2, tf.cast(tf.shape(newContent), tf.int64))

		# top row
		slice3 = tf.slice(tensor, [0, 0, 0, 0], [bs, 1, w, c])  # top row
		slice3 = tf.reverse(slice3, [2])
		batchIdx3, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx3 = batchIdx3.reshape([-1])
		y3, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y3 = y3.reshape([-1])
		yy3 = []
		for i in range(bs):
			yy3.extend(y3)
		x3, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x3 = x3.reshape([-1]) + 1
		xx3 = []
		for i in range(bs * 1):
			xx3.extend(x3)
		c_idx3 = range(c)
		chanel_idx3 = []
		for i in range(bs * 1 * h):
			chanel_idx3.extend(c_idx3)
		scatterIdx3 = tf.stack([tf.cast(batchIdx3,tf.int32), yy3, xx3, chanel_idx3], axis=1)
		slice3 = tf.reshape(slice3, [-1])
		delta3 = tf.SparseTensor(tf.cast(scatterIdx3, tf.int64), slice3, tf.cast(tf.shape(newContent), tf.int64))

		# bottom row
		slice4 = tf.slice(tensor, [0, h - 1, 0, 0], [bs, 1, w, c])  # bottom row
		slice4 = tf.reverse(slice4, [2])
		batchIdx4, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx4 = batchIdx4.reshape([-1])
		y4, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y4 = y4.reshape([-1]) + h + 1
		yy4 = []
		for i in range(bs):
			yy4.extend(y4)
		x4, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x4 = x4.reshape([-1]) + 1
		xx4 = []
		for i in range(bs * 1):
			xx4.extend(x4)
		c_idx4 = range(c)
		chanel_idx4 = []
		for i in range(bs * 1 * h):
			chanel_idx4.extend(c_idx4)
		scatterIdx4 = tf.stack([tf.cast(batchIdx4,tf.int32), yy4, xx4, chanel_idx4], axis=1)
		slice4 = tf.reshape(slice4, [-1])
		delta4 = tf.SparseTensor(tf.cast(scatterIdx4, tf.int64), slice4, tf.cast(tf.shape(newContent), tf.int64))

		new_t = newContent + tf.sparse_tensor_to_dense(delta1) + tf.sparse_tensor_to_dense(delta2) \
		        + tf.sparse_tensor_to_dense(delta3) + tf.sparse_tensor_to_dense(delta4)

		return new_t

	def padding_for_geoimg(self,tensor):
		bs, w, h, c = tensor.get_shape().as_list()
		# print('------------batch idx---------------')
		batchIdx, _ = np.meshgrid(range(bs), range(w * h * c), indexing="ij")
		batchIdx = batchIdx.reshape([-1])

		# print('------------yy---------------')
		y, _ = np.meshgrid(range(h), range(h * c), indexing="ij")
		y = y.reshape([-1]) + 1
		yy = []
		for i in range(bs):
			yy.extend(y)

		# print('------------xx---------------')
		x, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x = x.reshape([-1]) + 1
		xx = []
		for i in range(bs * w):
			xx.extend(x)

		# print('------------chanel idx---------------')
		c_idx = range(c)
		chanel_idx = []
		for i in range(bs * w * h):
			chanel_idx.extend(c_idx)

		# print('------------scatter idx---------------')
		scatterIdx = tf.stack([tf.cast(batchIdx, tf.int32), yy, xx, chanel_idx], axis=1)
		scatterContent = tf.reshape(tensor, [-1])
		scatterShape = tf.constant([bs, w + 2, h + 2, c])
		newContent = tf.scatter_nd(scatterIdx, scatterContent, scatterShape)  # padding with zero

		# ------------padding with reverse edge elements------------#
		# left colume
		slice1 = tf.slice(tensor, [0, 0, 0, 0], [bs, h, 1, c])  # left colume
		slice1 = tf.reverse(slice1, [1])
		batchIdx1, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx1 = batchIdx1.reshape([-1])
		y1, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y1 = y1.reshape([-1])
		yy1 = []
		for i in range(bs):
			yy1.extend(y1)
		x1, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x1 = x1.reshape([-1]) + 1
		xx1 = []
		for i in range(bs * 1):
			xx1.extend(x1)
		c_idx1 = range(c)
		chanel_idx1 = []
		for i in range(bs * 1 * h):
			chanel_idx1.extend(c_idx1)
		scatterIdx1 = tf.stack([tf.cast(batchIdx1, tf.int32), xx1, yy1, chanel_idx1], axis=1)
		slice1 = tf.reshape(slice1, [-1])
		delta1 = tf.SparseTensor(tf.cast(scatterIdx1, tf.int64), slice1, tf.cast(tf.shape(newContent), tf.int64))

		# right colume
		slice2 = tf.slice(tensor, [0, 0, w - 1, 0], [bs, h, 1, c])  # right colume
		slice2 = tf.reverse(slice2, [1])
		batchIdx2, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx2 = batchIdx2.reshape([-1])
		y2, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y2 = y2.reshape([-1]) + w + 1
		yy2 = []
		for i in range(bs):
			yy2.extend(y2)
		x2, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x2 = x2.reshape([-1]) + 1
		xx2 = []
		for i in range(bs * 1):
			xx2.extend(x2)
		c_idx2 = range(c)
		chanel_idx2 = []
		for i in range(bs * 1 * h):
			chanel_idx2.extend(c_idx2)
		scatterIdx2 = tf.stack([tf.cast(batchIdx2, tf.int32), xx2, yy2, chanel_idx2], axis=1)
		slice2 = tf.reshape(slice2, [-1])
		delta2 = tf.SparseTensor(tf.cast(scatterIdx2, tf.int64), slice2, tf.cast(tf.shape(newContent), tf.int64))

		# top row
		slice3 = tf.slice(tensor, [0, 0, 0, 0], [bs, 1, w, c])  # top row
		slice3 = tf.reverse(slice3, [2])
		batchIdx3, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx3 = batchIdx3.reshape([-1])
		y3, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y3 = y3.reshape([-1])
		yy3 = []
		for i in range(bs):
			yy3.extend(y3)
		x3, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x3 = x3.reshape([-1]) + 1
		xx3 = []
		for i in range(bs * 1):
			xx3.extend(x3)
		c_idx3 = range(c)
		chanel_idx3 = []
		for i in range(bs * 1 * h):
			chanel_idx3.extend(c_idx3)
		scatterIdx3 = tf.stack([tf.cast(batchIdx3, tf.int32), yy3, xx3, chanel_idx3], axis=1)
		slice3 = tf.reshape(slice3, [-1])
		delta3 = tf.SparseTensor(tf.cast(scatterIdx3, tf.int64), slice3, tf.cast(tf.shape(newContent), tf.int64))

		# bottom row
		slice4 = tf.slice(tensor, [0, h - 1, 0, 0], [bs, 1, w, c])  # bottom row
		slice4 = tf.reverse(slice4, [2])
		batchIdx4, _ = np.meshgrid(range(bs), range(1 * h * c), indexing="ij")
		batchIdx4 = batchIdx4.reshape([-1])
		y4, _ = np.meshgrid(range(1), range(h * c), indexing="ij")
		y4 = y4.reshape([-1]) + h + 1
		yy4 = []
		for i in range(bs):
			yy4.extend(y4)
		x4, _ = np.meshgrid(range(h), range(c), indexing="ij")
		x4 = x4.reshape([-1]) + 1
		xx4 = []
		for i in range(bs * 1):
			xx4.extend(x4)
		c_idx4 = range(c)
		chanel_idx4 = []
		for i in range(bs * 1 * h):
			chanel_idx4.extend(c_idx4)
		scatterIdx4 = tf.stack([tf.cast(batchIdx4, tf.int32), yy4, xx4, chanel_idx4], axis=1)
		slice4 = tf.reshape(slice4, [-1])
		delta4 = tf.SparseTensor(tf.cast(scatterIdx4, tf.int64), slice4, tf.cast(tf.shape(newContent), tf.int64))

		# --------------- Padding for four corners ----------------- #
		# Extract four corners and calculate mean value
		c1_slice = tf.slice(tensor, [0, 0, 0, 0], [bs, 1, 1, c])  # left-up corner
		c2_slice = tf.slice(tensor, [0, 0, h - 1, 0], [bs, 1, 1, c])  # right-up corner
		c3_slice = tf.slice(tensor, [0, w - 1, 0, 0], [bs, 1, 1, c])  # left-bottom corner
		c4_slice = tf.slice(tensor, [0, w - 1, h - 1, 0], [bs, 1, 1, c])  # right-bottom corner
		mean_slice = (c1_slice + c2_slice + c3_slice + c4_slice) / 4.0
		mean_slice = tf.reshape(mean_slice, [-1])

		# Fill four corner with mean value
		# left-up corner
		c1_batchIdx, _ = np.meshgrid(range(bs), range(1 * 1 * c), indexing="ij")
		c1_batchIdx = c1_batchIdx.reshape([-1])
		c1_y, _ = np.meshgrid(range(1), range(1 * c), indexing="ij")
		c1_y = c1_y.reshape([-1])
		c1_yy = []
		for i in range(bs):
			c1_yy.extend(c1_y)
		c1_x, _ = np.meshgrid(range(1), range(c), indexing="ij")
		c1_x = c1_x.reshape([-1])
		c1_xx = []
		for i in range(bs * 1):
			c1_xx.extend(c1_x)
		c1_idx = range(c)
		c1_chanel_idx = []
		for i in range(bs * 1 * 1):
			c1_chanel_idx.extend(c1_idx)
		c1_scatterIdx = tf.stack([tf.cast(c1_batchIdx, tf.int32), c1_xx, c1_yy, c1_chanel_idx], axis=1)
		c1_delta = tf.SparseTensor(tf.cast(c1_scatterIdx, tf.int64), mean_slice,
		                           tf.cast(tf.shape(newContent), tf.int64))

		# right-up corner
		c2_batchIdx, _ = np.meshgrid(range(bs), range(1 * 1 * c), indexing="ij")
		c2_batchIdx = c2_batchIdx.reshape([-1])
		c2_y, _ = np.meshgrid(range(1), range(1 * c), indexing="ij")
		c2_y = c2_y.reshape([-1]) + 1 + w
		c2_yy = []
		for i in range(bs):
			c2_yy.extend(c2_y)
		c2_x, _ = np.meshgrid(range(1), range(c), indexing="ij")
		c2_x = c2_x.reshape([-1])
		c2_xx = []
		for i in range(bs * 1):
			c2_xx.extend(c2_x)
		c2_idx = range(c)
		c2_chanel_idx = []
		for i in range(bs * 1 * 1):
			c2_chanel_idx.extend(c2_idx)
		c2_scatterIdx = tf.stack([tf.cast(c2_batchIdx, tf.int32), c2_xx, c2_yy, c2_chanel_idx], axis=1)
		c2_delta = tf.SparseTensor(tf.cast(c2_scatterIdx, tf.int64), mean_slice,
		                           tf.cast(tf.shape(newContent), tf.int64))

		# left-bottom corner
		c3_batchIdx, _ = np.meshgrid(range(bs), range(1 * 1 * c), indexing="ij")
		c3_batchIdx = c3_batchIdx.reshape([-1])
		c3_y, _ = np.meshgrid(range(1), range(1 * c), indexing="ij")
		c3_y = c3_y.reshape([-1])
		c3_yy = []
		for i in range(bs):
			c3_yy.extend(c3_y)
		c3_x, _ = np.meshgrid(range(1), range(c), indexing="ij")
		c3_x = c3_x.reshape([-1]) + 1 + h
		c3_xx = []
		for i in range(bs * 1):
			c3_xx.extend(c3_x)
		c3_idx = range(c)
		c3_chanel_idx = []
		for i in range(bs * 1 * 1):
			c3_chanel_idx.extend(c3_idx)
		c3_scatterIdx = tf.stack([tf.cast(c3_batchIdx, tf.int32), c3_xx, c3_yy, c3_chanel_idx], axis=1)
		c3_delta = tf.SparseTensor(tf.cast(c3_scatterIdx, tf.int64), mean_slice,
		                           tf.cast(tf.shape(newContent), tf.int64))

		# right-bottom corner
		c4_batchIdx, _ = np.meshgrid(range(bs), range(1 * 1 * c), indexing="ij")
		c4_batchIdx = c4_batchIdx.reshape([-1])
		c4_y, _ = np.meshgrid(range(1), range(1 * c), indexing="ij")
		c4_y = c4_y.reshape([-1]) + 1 + w
		c4_yy = []
		for i in range(bs):
			c4_yy.extend(c4_y)
		c4_x, _ = np.meshgrid(range(1), range(c), indexing="ij")
		c4_x = c1_x.reshape([-1]) + 1 + h
		c4_xx = []
		for i in range(bs * 1):
			c4_xx.extend(c4_x)
		c4_idx = range(c)
		c4_chanel_idx = []
		for i in range(bs * 1 * 1):
			c4_chanel_idx.extend(c4_idx)
		c4_scatterIdx = tf.stack([tf.cast(c4_batchIdx, tf.int32), c4_xx, c4_yy, c4_chanel_idx], axis=1)
		c4_delta = tf.SparseTensor(tf.cast(c4_scatterIdx, tf.int64), mean_slice,
		                           tf.cast(tf.shape(newContent), tf.int64))

		new_t = newContent + tf.sparse_tensor_to_dense(delta1) + tf.sparse_tensor_to_dense(delta2) \
		        + tf.sparse_tensor_to_dense(delta3) + tf.sparse_tensor_to_dense(delta4) \
		        + tf.sparse_tensor_to_dense(c1_delta) + tf.sparse_tensor_to_dense(c2_delta) \
		        + tf.sparse_tensor_to_dense(c3_delta) + tf.sparse_tensor_to_dense(c4_delta)

		return new_t

	def build_dict(self):
		self.matfile = {}
		self.matfile_normal = {}
		self.matfile_test = {}
		self.mat_data = {}
		self.mat_data_test = {}
		self.mat_normal = {}
		self.mat_data_size = {}
		self.mat_data_size_test = {}
		self.depth_dir = {}
		self.depth_dir_test = {}
		self.depth_dir_old = {}

		self.matfile_32 = {}
		self.matfile_normal_32 = {}
		self.matfile_test_32 = {}
		self.mat_data_32 = {}
		self.mat_data_test_32 = {}
		self.mat_normal_32 = {}
		self.mat_data_size_32 = {}

		self.matfile_16 = {}
		self.matfile_normal_16 = {}
		self.matfile_test_16 = {}
		self.mat_data_16 = {}
		self.mat_data_test_16 = {}
		self.mat_normal_16 = {}
		self.mat_data_size_16 = {}

		self.cate2label = {}

		self.test_sample_idx = {}
		self.test_sample_idx['airplane'] = 50
		self.test_sample_idx['bench'] = 56
		self.test_sample_idx['cabinet'] = 9
		self.test_sample_idx['car'] = 8
		self.test_sample_idx['chair'] = 38
		self.test_sample_idx['display'] = 5
		self.test_sample_idx['lamp'] = 0
		self.test_sample_idx['loudspeaker'] = 5
		self.test_sample_idx['rifle'] = 0
		self.test_sample_idx['sofa'] = 58
		self.test_sample_idx['table'] = 25
		self.test_sample_idx['telephone'] = 1