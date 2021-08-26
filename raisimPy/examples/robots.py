import os
import numpy as np
import sys
sys.path.append('/home/zhaoming/Documents/open_robot/raisim_build/lib')
import raisimpy as raisim
import time

raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
anymal_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/anymal/urdf/anymal.urdf"
laikago_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/laikago/laikago.urdf"
atlas_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/atlas/robot.urdf"
monkey_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/monkey/monkey.obj"
solo8_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/solo8_URDF_v6/solo8.urdf"
dummy_inertia = np.zeros([3, 3])
np.fill_diagonal(dummy_inertia, 0.1)

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()

# anymal = world.addArticulatedSystem(anymal_urdf_file)
# anymal.setName("anymal")
# anymal_nominal_joint_config = np.array([0, -1.5, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8,
#                                         -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
# anymal.setGeneralizedCoordinate(anymal_nominal_joint_config)
# anymal.setPdGains(200*np.ones([18]), np.ones([18]))
# anymal.setPdTarget(anymal_nominal_joint_config, np.zeros([18]))

# laikago = world.addArticulatedSystem(laikago_urdf_file)
# laikago.setName("laikago")
# laikago_nominal_joint_config = np.array([0, 1.5, 0.48, 1, 0.0, 0.0, 0.0, 0.0, 0.5, -1, 0, 0.5, -1,
#                                          0.00, 0.5, -1, 0, 0.5, -0.7])
# laikago.setGeneralizedCoordinate(laikago_nominal_joint_config)
# laikago.setPdGains(200*np.ones([18]), np.ones([18]))
# laikago.setPdTarget(laikago_nominal_joint_config, np.zeros([18]))

# atlas = world.addArticulatedSystem(atlas_urdf_file)
# atlas.setName("atlas")
# atlas_nominal_joint_config = np.zeros(atlas.getGeneralizedCoordinateDim())
# atlas_nominal_joint_config[2] = 1.5
# atlas_nominal_joint_config[3] = 1
# atlas.setGeneralizedCoordinate(atlas_nominal_joint_config)

solo8 = world.addArticulatedSystem(solo8_urdf_file)
solo8.setName("solo8")
solo8_nominal_joint_config = np.zeros(solo8.getGeneralizedCoordinateDim())
print(solo8_nominal_joint_config)
solo8_nominal_joint_config[0] = 0
solo8_nominal_joint_config[2] = 0.35
solo8_nominal_joint_config[3] = 1
solo8.setGeneralizedCoordinate(solo8_nominal_joint_config)
solo8.setPdGains(2*np.ones([14]), 0.2*np.ones([14]))
solo8.setPdTarget(solo8_nominal_joint_config, np.zeros([14]))

server.launchServer(8080)
obj = world.addSphere(0.2, 0.8)
obj.setPosition(0, 0, 0.2)
# import ipdb; ipdb.set_trace()

# for i in range(5):
#     for j in range(5):
#         object_type = (i + j*6) % 5

#         if object_type == 0:
#             obj = world.addMesh(monkey_file, 5.0, dummy_inertia, np.array([0, 0, 0]), 0.3)
#         elif object_type == 1:
#             obj = world.addCylinder(0.2, 0.3, 2.0)
#         elif object_type == 2:
#             obj = world.addCapsule(0.2, 0.3, 2.0)
#         elif object_type == 3:
#             obj = world.addBox(0.4, 0.4, 0.4, 2.0)
#         else:
#             obj = world.addSphere(0.3, 2.0)

#         obj.setPosition(i-2.5, j-2.5, 5)

time.sleep(2)
world.integrate1()

### get dynamic properties
# mass matrix
# mass_matrix = anymal.getMassMatrix()
# non-linear term (gravity+coriolis)
# non_linearities = anymal.getNonlinearities()
# Jacobians
# jaco_foot_lh_linear = anymal.getDenseFrameJacobian("LF_ADAPTER_TO_FOOT")
# jaco_foot_lh_angular = anymal.getDenseFrameRotationalJacobian("LF_ADAPTER_TO_FOOT")

reference = np.zeros(15)
reference[2] = 0.55
reference[3] = 0.7071068
reference[5] = -0.7071068
t = 0
period = 30

for i in range(5000000):
	for j in range(4):
		if t <= period / 2:
			if j == 0 or j == 3:
				reference[7+j*2] = 0.65
				reference[8+j*2] = -1.0
			else:
				reference[7+j*2] = 0.65
				reference[8+j*2] = -1.0 - 0.7 * np.sin(2.0*np.pi*t/period)
		else:
			if j == 0 or j == 3:
				reference[7+j*2] = 0.65
				reference[8+j*2] = -1.0 - 0.7 * np.sin(2.0*np.pi*t/period - np.pi)
			else:
				reference[7+j*2] = 0.65
				reference[8+j*2] = -1.0
	reference[7+4] = 1.57
	reference[7+6] = 1.57
	reference[8+4] = 0
	reference[8+6] = 0
	reference[2] = 0.53 + 0.4
	if t <= period / 2:
		reference[7+4] = 1.57 - 0.7 * np.sin(2.0*np.pi*t/period)
		reference[8+4] = 0 + 0.7 * np.sin(2.0*np.pi*t/period)
	else:
		reference[7+6] = 1.57 - 0.7 * np.sin(2.0*np.pi*t/period - np.pi)
		reference[8+6] = 0 + 0.7 * np.sin(2.0*np.pi*t/period - np.pi)
	reference[7] = 1.57 + np.sin(2.0*np.pi*t/period)
	reference[9] = 1.57 + np.sin(2.0*np.pi*t/period + np.pi)
	if i == 0:
		solo8.setGeneralizedCoordinate(reference)
	solo8.setPdTarget(reference, np.zeros([14]))

	fl_id = solo8.getFrameIdxByName("FL_ANKLE")

	print(solo8.getFrameVelocity(fl_id))

	# import ipdb; ipdb.set_trace()

	# for _ in range(10):
	world.integrate()
	import time; time.sleep(0.02)
	t += 1
	if t > period:
		t = 0


server.killServer()
