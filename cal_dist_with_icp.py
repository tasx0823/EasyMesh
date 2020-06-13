import os
import numpy as np
import scipy.io as scio

def best_fit_transform(A, B):
	'''
	Calculates the least-squares best-fit transform between corresponding 3D points A->B
	Input:
	  A: Nx3 numpy array of corresponding 3D points
	  B: Nx3 numpy array of corresponding 3D points
	Returns:
	  T: 4x4 homogeneous transformation matrix
	  R: 3x3 rotation matrix
	  t: 3x1 column vector
	'''

	assert len(A) == len(B)

	# translate points to their centroids
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
	AA = A - centroid_A
	BB = B - centroid_B

	# rotation matrix
	W = np.dot(BB.T, AA)
	U, s, VT = np.linalg.svd(W)
	R = np.dot(U, VT)

	# special reflection case

	if np.linalg.det(R) < 0:
		VT[2, :] *= -1
		R = np.dot(U, VT)

	# translation
	t = centroid_B.T - np.dot(R, centroid_A.T)

	# homogeneous transformation
	T = np.identity(4)
	T[0:3, 0:3] = R
	T[0:3, 3] = t

	return T, R, t

def nearest_neighbor(x, c):
	'''
	Find the nearest (Euclidean) neighbor in dst for each point in src
	Input:
		x: Nx3 array of points
		c: Nx3 array of points
	Output:
		distances: Euclidean distances (errors) of the nearest neighbor
		indecies: dst indecies of the nearest neighbor
	'''

	ndata, dimx = np.shape(x)
	ncentres, dimc = np.shape(c)
	assert dimx == dimc
	n2 = np.transpose(np.matmul(np.ones((ncentres, 1)), np.expand_dims(np.sum(np.transpose(x * x), 0), 0))) \
	     + np.matmul(np.ones((ndata, 1)), np.expand_dims(np.sum(np.transpose(c * c), 0), 0)) \
	     - 2.0 * (np.matmul(x, np.transpose(c)))
	n2[n2 < 0] = 0
	return np.sqrt(np.min(n2, 1)), np.argmin(n2, 1)

def icp(A, B, init_pose=None, max_iterations=100, tolerance=0.001):
	'''
	The Iterative Closest Point method
	Input:
		A: Nx3 numpy array of source 3D points
		B: Nx3 numpy array of destination 3D point
		init_pose: 4x4 homogeneous transformation
		max_iterations: exit algorithm after max_iterations
		tolerance: convergence criteria
	Output:
		T: final homogeneous transformation
		distances: Euclidean distances (errors) of the nearest neighbor
	'''

	# make points homogeneous, copy them so as to maintain the originals
	src = np.ones((4, A.shape[0]))
	dst = np.ones((4, B.shape[0]))
	src[0:3, :] = np.copy(A.T)   # 4 x N
	dst[0:3, :] = np.copy(B.T)   # 4 x N

	# apply the initial pose estimation, important !!!!!!!

	if init_pose is not None:
		src = np.dot(init_pose, src)

	prev_error = 0
	accum_T = np.diag((1,1,1,1))
	for i in range(max_iterations):

		# find the nearest neighbours between the current source and destination points
		distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

		# compute the transformation between the current source and nearest destination points
		T, _, _ = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T)

		# update the current source
		# refer to "Introduction to Robotics" Chapter2 P28. Spatial description and transformations
		src = np.dot(T, src)
		accum_T = np.dot(T,accum_T)
		# check error
		mean_error = np.sum(distances) / distances.size
		if abs(prev_error - mean_error) < tolerance:
			break
		prev_error = mean_error

	# calculcate final tranformation
	T, _, _ = best_fit_transform(A, src[0:3, :].T)


	#In fact, accum_T is equal  to T
	return T, distances

def save_pc(filename,pc):
	with open(filename,'w') as f:
		f.write('OFF\n')
		f.write('{:d} {:d} {:d}\n'.format(np.shape(pc)[0],0,0))
		for i in range(np.shape(pc)[0]):
			f.writelines(str(pc[i,:]).strip('[').strip(']')+'\n')

def save_pc_txt(filename,pc):
	np.savetxt(filename,pc)

def dist2(x,c):
	ndata,dimx = np.shape(x)
	ncentres,dimc = np.shape(c)
	assert dimx == dimc
	n2 = np.transpose(np.matmul(np.ones((ncentres,1)) , np.expand_dims(np.sum(np.transpose(x*x),0),0) ) )\
	     + np.matmul(np.ones((ndata,1)) , np.expand_dims(np.sum( np.transpose(c*c),0),0) )\
	     - 2.0 * (np.matmul(x,np.transpose(c)))
	n2[n2<0] = 0
	return n2


experiment_name = ['10_2','NoEdge','NoGAN','NoMultiChamfer','NoNormal','NoPadding','NoPESRM','PointDis','Res_D']

experiment_name = 'Res_D'
cates = ['airplane', 'bench', 'cabinet','car', 'chair', 'display', 'lamp', 'loudspeaker', 'rifle','sofa','table', 'telephone']
#cates = [ 'car', 'chair', 'display', 'lamp', 'rifle','sofa','table']

#cates = ['loudspeaker']
print('Experiment name:{:s}'.format(experiment_name))
for cate in cates:
	print('calculating {:s}'.format(cate))
	#load ground truth point cloud
	gt_path = r'/data1/sunx/CVPR2019/data/{:s}/{:s}_test_geoimg_64.mat'.format(cate,cate)
	#print(gt_path)
	gt_data = scio.loadmat(gt_path)['geoimgs']

	#load generated point cloud
	pre_path = r'results/{:s}_{:s}_test_samples.mat'.format(experiment_name,cate)
	pre_data = scio.loadmat(pre_path)['geoimgs']

	#print('Total sample ',pre_data.shape[0])
	assert gt_data.shape == pre_data.shape

	pre2gts = 0
	gt2pres = 0

	for i in range(gt_data.shape[0]):
		#print('{:s} {:d}'.format(cate,i))
		gt = gt_data[i,:,:,:] * 2
		gt = np.reshape(gt,(64*64,3))
		gt_zhixin = (np.max(gt, 0) + np.min(gt, 0)) / 2.0
		gt = gt - gt_zhixin

		pre = pre_data[i,:,:,:]
		pre = np.reshape(pre,(64*64,3))
		pre_zhixin = (np.max(pre,0) + np.min(pre,0)) / 2.0
		pre = pre - pre_zhixin

		r_idx1 = np.random.choice(range(np.shape(pre)[0]), 500, replace=False)
		A_sample = pre[r_idx1, :]
		r_idx2 = np.random.choice(range(np.shape(gt)[0]), 500, replace=False)
		B_sample = gt[r_idx2, :]
		T, distances = icp(A_sample, B_sample)
		A_hom = np.ones((4, pre.shape[0]))
		A_hom[0: 3, :] = np.copy(pre.T)
		A_transform = np.dot(T, A_hom)[0:3, :].T  # rotate A towards B

		pre = A_transform

		'''
		gt_dif = np.max(gt, 0) - np.min(gt, 0, keepdims=True)
		pre_dif = np.max(pre, 0) - np.min(pre, 0, keepdims=True)
		# print('hahaha',np.shape(pre_dif))
		dif = gt_dif / pre_dif
		pre = pre * dif'''

		dist_mat = dist2(gt,pre)

		pre2gt = np.mean(np.sqrt(np.min(dist_mat,0)))
		gt2pre = np.mean(np.sqrt(np.min(dist_mat,1)))

		gt2pres += gt2pre
		pre2gts += pre2gt

	pre2gts = pre2gts / gt_data.shape[0]
	gt2pres = gt2pres / gt_data.shape[0]

	print('----------------------------------------------')
	print('experiment name:{:s}, {:s}'.format(experiment_name,cate))
	print('pre2gts is {:.6f}'.format(pre2gts))
	print('gt2pres is {:.6f}'.format(gt2pres))


