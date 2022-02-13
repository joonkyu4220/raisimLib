import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
import sys
sys.path.append('/home/zhaoming/Documents/open_robot/raisim_build/lib')
import raisimpy as raisim

from fairmotion.data import bvh
BVH_FILENAME = "../../../../../Downloads/MotionCapture/Mocap/Loco/RunSideBack1.bvh"
motion = bvh.load(BVH_FILENAME)

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()
virtual_human_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/virtual_human.urdf"
virtual_human = world.addArticulatedSystem(virtual_human_urdf_file)
virtual_human.setName("virtual_human")
virtual_human_nominal_joint_config = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
 0., 0., 0.,])

# visBox = server.addVisualBox("v_box", 1, 1, 1, 1, 1, 1, 1)
# visAngularVelBox = server.addVisualBox("v_box2", 1, 1, 1, 1, 1, 0, 1)

def coordinate_transform(q):
	# q[0:4] = q[[0, 3, 1, 2]]
	# q[1] = -q[1]
	# q[3] = -q[3]
	return q

def process_mocap(mocap_file):
	motion = bvh.load(BVH_FILENAME)
	matrix = motion.to_matrix(local=False)[:, :, :, :]
	local_matrix = motion.to_matrix(local=True)[:, :, : ,:]
	pos = matrix[:, :, :3, 3]
	num_frames = matrix.shape[0]

	#generate virtual root pose
	root_pos = np.zeros((num_frames, 3))
	root_rot = np.zeros((num_frames, 3, 3))
	root_pos[:, 0] = pos[:, 0, 0]
	root_pos[:, 2] = pos[:, 0, 2]
	root_rot[:, :, 0] = np.cross((pos[:, 15, :] - pos[:, 19, :]), (pos[:, 0, :] - pos[:, 1, :]))
	root_rot[:, 1, 0] = 0
	norms = np.apply_along_axis(np.linalg.norm, 1, root_rot[:, :, 0])
	root_rot[:, :, 0] /= norms[:, np.newaxis]
	root_rot[:, :, 1] = np.array([0, 1, 0])
	root_rot[:, :, 2] = np.cross(root_rot[:, :, 0], root_rot[:, :, 1])

	#root velocity
	root_linear_vel = np.zeros((num_frames, 3))
	root_angular_vel = np.zeros((num_frames, 3, 3))
	root_angular_vel[-1, :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	for i in range(num_frames-1):
		root_linear_vel[i] = root_rot[i, :, :].T.dot(root_pos[i+1, :] - root_pos[i, :])
		root_angular_vel[i, :, :] = root_rot[i+1, :, :].dot(root_rot[i, :, :].T)

	#generate bone data w.r.t virtual root
	store_data = np.zeros((num_frames, 23, 3))
	bone_velocity = np.zeros((num_frames, 23, 3))

	#bone position w.r.t the root
	for i in range(23):
		A = np.transpose(root_rot[:, : :], axes=[0, 2, 1])
		x = pos[:, i, :] - root_pos
		# store_data[:, i, :] = np.einsum('ijk, ij->k', A, x)
		store_data[:, i, :] = np.matmul(A, x[:, :, None]).squeeze()
	#bone velocity
	for i in range(num_frames-1):
		for j in range(23):
			bone_velocity[i, j, :] = store_data[i+1, j, :] - store_data[i, j, :]

	return root_pos, root_rot, root_linear_vel, root_angular_vel, store_data, bone_velocity


def set_human_pose(index, hip_pose=None, hip_rot=None):
	reference =  np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0.,])
	base_rot = R.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()
	#base_rot = R.from_quat([0.707, 0.0, 0.0, 0.707]).as_matrix()
		#r = R.from_matrix(matrix[j, 0, 0:3, 0:3])
		#rotation = base_rot.dot(r.as_matrix().T)
		#translation = r.as_matrix().T.dot(matrix[j, 0, 0:3, 3])

	#hip rot
	if hip_pose is None:
		reference[0:3] = base_rot.dot(matrix[j, 0, 0:3, 3])/100#(r.as_matrix().dot(matrix[j, 0, 0:3, 3]) - translation)[[0, 2, 1]] / 100 + np.array([0, 0, 1])
	else:
		reference[0:3] = (hip_pose)[[0, 2, 1]] / 100 - np.array([0, 0, 0.05])
		reference[1] *= -1
	if hip_rot is None:
		hip_rot = R.from_matrix(base_rot.dot(matrix[index, 0, 0:3, 0:3])).as_quat()
	else:
		hip_rot = R.from_matrix(base_rot.dot(hip_rot)).as_quat()
	reference[3:7] = hip_rot[[3, 0, 1, 2]]

	#set left leg
	reference[7:11] = coordinate_transform(R.from_matrix(local_matrix[index, 19, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[11:15] = coordinate_transform(R.from_matrix(local_matrix[index, 20, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[15:19] = coordinate_transform(R.from_matrix(local_matrix[index, 21, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set right leg
	reference[19:23] = coordinate_transform(R.from_matrix(local_matrix[index, 15, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[23:27] = coordinate_transform(R.from_matrix(local_matrix[index, 16, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[27:31] = coordinate_transform(R.from_matrix(local_matrix[index, 17, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set body
	reference[31:35] = coordinate_transform(R.from_matrix(local_matrix[index, 1, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[35:39] = coordinate_transform(R.from_matrix(local_matrix[index, 2, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[39:43] = coordinate_transform(R.from_matrix(local_matrix[index, 3, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[43:47] = coordinate_transform(R.from_matrix(local_matrix[index, 4, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[47:51] = coordinate_transform(R.from_matrix(local_matrix[index, 5, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set left arms
	reference[51:55] = coordinate_transform(R.from_matrix(local_matrix[index, 11, 0:3, 0:3]).as_quat()[[3,0,2,1]])
	reference[55:59] = coordinate_transform(R.from_matrix(local_matrix[index, 12, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[59:63] = coordinate_transform(R.from_matrix(local_matrix[index, 13, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set right arms
	reference[63:67] = coordinate_transform(R.from_matrix(local_matrix[index, 7, 0:3, 0:3]).as_quat()[[3,0,2,1]])
	reference[67:71] = coordinate_transform(R.from_matrix(local_matrix[index, 8, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[71:75] = coordinate_transform(R.from_matrix(local_matrix[index, 9, 0:3, 0:3]).as_quat()[[3,0,1,2]])


	virtual_human.setState(reference, np.zeros([virtual_human.getDOF()]))


balls = []
for i in range(23):
	balls.append(world.addSphere(0.02, 0.8))
balls.append(world.addSphere(0.1, 0.8))


matrix = motion.to_matrix(local=False)[:, :, :, :]
local_matrix = motion.to_matrix(local=True)[:, :, : ,:]
pos = matrix[:, :, :3, 3]

num_frames = matrix.shape[0]

root_pos = np.zeros((num_frames, 3))
root_rot = np.zeros((num_frames, 3, 3))
root_pos[:, 0] = pos[:, 0, 0]
root_pos[:, 2] = pos[:, 0, 2]

root_rot[:, :, 0] = np.cross((pos[:, 15, :] - pos[:, 19, :]), (pos[:, 0, :] - pos[:, 1, :]))
root_rot[:, 1, 0] = 0
norms = np.apply_along_axis(np.linalg.norm, 1, root_rot[:, :, 0])
root_rot[:, :, 0] /= norms[:, np.newaxis]
root_rot[:, :, 1] = np.array([0, 1, 0])
root_rot[:, :, 2] = np.cross(root_rot[:, :, 0], root_rot[:, :, 1])

store_data = np.zeros((num_frames, 23, 3))
bone_velocity = np.zeros((num_frames, 23, 3))


#bone position w.r.t the root
for i in range(23):
	A = np.transpose(root_rot[:, : :], axes=[0, 2, 1])
	x = pos[:, i, :] - root_pos
	# store_data[:, i, :] = np.einsum('ijk, ij->k', A, x)
	store_data[:, i, :] = np.matmul(A, x[:, :, None]).squeeze()
#bone velocity
for i in range(num_frames-1):
	for j in range(23):
		bone_velocity[i, j, :] = store_data[i+1, j, :] - store_data[i, j, :]


#root velocity
root_linear_vel = np.zeros((num_frames, 3))
root_angular_vel = np.zeros((num_frames, 3, 3))
root_angular_vel[-1, :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
for i in range(num_frames-1):
	root_linear_vel[i] = root_rot[i, :, :].T.dot(root_pos[i+1, :] - root_pos[i, :])
	root_angular_vel[i, :, :] = root_rot[i+1, :, :].dot(root_rot[i, :, :].T)

#build feature space for kd tree
features = np.zeros((num_frames - 60, 3 * 4 + 3 * 5))
for i in range(num_frames - 60):
	frames = [19, 39, 59]
	root_pos_feature = np.zeros(3)
	for j in range(3):
		features[i, j*4:j*4+2] = root_rot[i, :, :].T.dot(root_pos[frames[j] + i] - root_pos[i])[[0, 2]]
		features[i, j*4+2:j*4+4] = root_rot[i+frames[j], :, :].dot(root_rot[i, :, :].T).dot(np.array([1, 0, 0]))[[0, 2]]
		#features[i, j*9+6:j*9+9] = root_angular_vel[i+j, :, 1].copy()
	features[i, 3 * 4: 3 * 4 + 3] = bone_velocity[i, 0, :].flatten().copy()
	features[i, 3 * 4 + 3: 3 * 4 + 6] = store_data[i, 17, :].flatten().copy()
	features[i, 3 * 4 + 6: 3 * 4 + 9] = store_data[i, 21, :].flatten().copy()
	features[i, 3 * 4 + 9: 3 * 4 + 12] = bone_velocity[i, 17, :].flatten().copy()
	features[i, 3 * 4 + 12: 3 * 4 + 15] = bone_velocity[i, 21, :].flatten().copy()

feature_mean = np.mean(features, axis=0)
feature_std = np.std(features, axis=0)
feature_std[0:12] /= 1
for i in range(feature_std.shape[0]):
	if abs(feature_std[i]) < 0.00001:
		feature_std[i] = 1
features = (features - feature_mean) / feature_std

#build kd tree
tree = KDTree(features)
dist, ind = tree.query(features[2:3, :], k=10)
print(dist, ind)

server.launchServer(8081)
world.integrate1()

world_vel = np.zeros(3)

#kd tree query
current_root_pose = np.zeros(3)
desired_root_linear_velocity = np.array([10, 0, 0])
current_bone_pos = store_data[0, :, :].copy()
current_bone_velocity = bone_velocity[0, :, :].copy()
current_root_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
desired_root_delta_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
while True:
	#build feature for query
	query_feature = np.zeros((1, 3 * 4 + 3 * 5))
	frames = [19, 39, 59]
	for i in range(3):
		query_feature[:, i*4:i*4+2] = np.array([frames[i] * desired_root_linear_velocity[0], frames[i] * desired_root_linear_velocity[2]])
		query_feature[:, i*4+2:i*4+4] = current_root_orientation.T.dot(np.array([1, 0, 0]))[[0, 2]]
	query_feature[0, 3 * 4:3 * 4+3] = current_bone_velocity[0, :].flatten().copy()
	query_feature[0, 3 * 4 + 3:3 * 4 + 6] = current_root_orientation.T.dot(current_bone_pos[17, :].flatten().copy() - current_root_pose)
	query_feature[0, 3 * 4 + 6:3 * 4 + 9] = current_root_orientation.T.dot(current_bone_pos[21, :].flatten().copy() - current_root_pose)
	query_feature[0, 3 * 4 + 9:3 * 4 + 12] = current_bone_velocity[17, :].copy()
	query_feature[0, 3 * 4 + 12:3 * 4 + 15] = current_bone_velocity[21, :].copy()
	dist, ind = tree.query((query_feature-feature_mean) / feature_std, k=9)
	for t in range(20):
		# current_root_pose += desired_root_linear_velocity / 10
		# current_root_orientation = current_root_orientation.dot(desired_root_delta_orientation)
		
		current_root_pose += current_root_orientation.dot(root_linear_vel[ind[0,0]+t-1])
		current_root_orientation = root_angular_vel[ind[0,0]+t-1, :, :].dot(current_root_orientation)
		
		print(dist, ind)
		for i in range(23):
			current_bone_pos[i, :] = ((store_data[ind[0,0]+1+t, i, :] + current_root_pose))
			balls[i].setPosition(current_bone_pos[i, 0]/200, current_bone_pos[i, 2]/200, current_bone_pos[i, 1]/200)
		current_bone_velocity[0, :] = bone_velocity[ind[0, 0]+1+t, 0, :].copy()
		current_bone_velocity[17, :] = bone_velocity[ind[0, 0]+1+t, 17, :].copy()
		current_bone_velocity[21, :] = bone_velocity[ind[0, 0]+1+t, 21, :].copy()

		set_human_pose(ind[0, 0]+1+t, current_bone_pos[0, :], current_root_orientation)

		world.integrate()
		import time; time.sleep(0.016)

while True:
	for j in range(num_frames):
		#set skeleton pose
		set_human_pose(j)

		# set point cloud pose
		# for i in range(23):
		# 	bone_pos = root_rot[j,:,:].dot(store_data[j, i, :]) + root_pos[j, :]
		# 	balls[i].setPosition(bone_pos[0]/200, bone_pos[2]/200, bone_pos[1]/200)
		# 	# balls[i].setPosition(pos[j, i, 0]/200, pos[j, i, 2] / 200, pos[j, i, 1] / 200)

		# balls[23].setPosition(root_pos[j, 0] / 200, root_pos[j, 2] / 200, root_pos[j, 1] / 200)

		# r = R.from_matrix(root_rot[j, :, :]).as_quat()
		# displacement = root_rot[j, :, :].dot(np.array([0.1, 0, 0])) + root_pos[j, :] / 200
		# direction.setPosition(displacement[0], displacement[2], displacement[1] + 0.1)
		# direction.setOrientation(r[3], r[0], r[2], -r[1])
		

		# #visualize linear velocity
		# world_vel = world_vel * 0.9 + 0.1 * root_rot[j, :, :].dot(root_linear_vel[j])
		# world_orientation = np.zeros((3, 3))
		# normalized_world_vel = world_vel / np.linalg.norm(world_vel)
		# world_orientation[:, 0] = normalized_world_vel
		# world_orientation[:, 1] = np.array([0, 1, 0])
		# world_orientation[:, 2] = np.cross(world_orientation[:, 0], world_orientation[:, 1])
		# world_quat = R.from_matrix(world_orientation).as_quat()
		# visBox.setBoxSize(np.linalg.norm(world_vel), 0.01, 0.01)
		# displacement = root_pos[j, :] / 200 + world_orientation.dot(np.array([1, 0, 0]) * np.linalg.norm(world_vel) / 2)
		# visBox.setPosition(np.array([displacement[0], displacement[2], displacement[1] + 0.1]))
		# visBox.setOrientation(np.array([world_quat[3], world_quat[0], world_quat[2], -world_quat[1]]))

		# #visualzie angular velocity
		# visAngularVelBox.setBoxSize(0.4, 0.1, 0.1)
		# displacement = root_rot[j, :, :].dot(np.array([0.1, 0, 0])) + root_pos[j, :] / 200
		# visAngularVelBox.setPosition(np.array([displacement[0], displacement[2], displacement[1] + 0.1]))
		# next_orientation = root_rot[j, : ,:].copy()
		# for k in range(10):
		# 	if k + j < num_frames-1:
		# 		next_orientation = root_angular_vel[j + k, :, :].dot(next_orientation)
		# next_quat = R.from_matrix(next_orientation).as_quat()
		# visAngularVelBox.setOrientation(np.array([next_quat[3], next_quat[0], next_quat[2], -next_quat[1]]))

		import time; time.sleep(0.016)

		world.integrate()