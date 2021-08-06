import numpy as np

from pydrake.all import MultibodyPlant, FindResourceOrThrow, Parser, InverseKinematics, Solve
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer
from pydrake.all import RigidTransform, Simulator, PidController, RotationMatrix, RollPitchYaw

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess()


builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
# plant = MultibodyPlant(0.001)
filename = FindResourceOrThrow("drake/examples/atlas/urdf/atlas_convex_hull.urdf")
Parser(plant).AddModelFromFile(filename)
plant.Finalize()


visualizer = ConnectMeshcatVisualizer(builder=builder, 
    scene_graph=scene_graph, 
    zmq_url=zmq_url)

diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)


ik = InverseKinematics(plant=plant, with_joint_limits=False)
epsilon = 1e-2

r_foot_position = np.array([0.0, -0.15, 0.])
ik.AddPositionConstraint(
        frameB=plant.GetFrameByName("r_foot"),
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_upper=r_foot_position+epsilon,
        p_AQ_lower=r_foot_position-epsilon)


l_foot_position = np.array([0.0, 0.15, 0.])
ik.AddPositionConstraint(
        frameB=plant.GetFrameByName("l_foot"),
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_upper=l_foot_position+epsilon,
        p_AQ_lower=l_foot_position-epsilon)

com_position = np.array([0.0, 0., 0.5])
ik.AddPositionConstraint(
        frameB=plant.GetFrameByName("pelvis"),
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_upper=com_position+epsilon,
        p_AQ_lower=com_position-epsilon)

rotation = RotationMatrix(RollPitchYaw(0,0,0))
ik.AddOrientationConstraint(
        plant.world_frame(), rotation,
        plant.GetFrameByName("r_foot"),rotation,
        epsilon)

ik.AddOrientationConstraint(
        plant.world_frame(), rotation,
        plant.GetFrameByName("l_foot"),rotation,
        epsilon)

ik.AddOrientationConstraint(
        plant.world_frame(), rotation,
        plant.GetFrameByName("pelvis"),rotation,
        epsilon)

lknee = plant.GetJointByName(name="l_leg_kny")
rknee = plant.GetJointByName(name="r_leg_kny")
lknee.set_angle(plant_context,0.2)
rknee.set_angle(plant_context,0.2)

joint1 = plant.GetJointByName(name="back_bkx")
joint1.set_angle(plant_context,0)
joint2 = plant.GetJointByName(name="back_bky")
joint2.set_angle(plant_context,0)
joint3 = plant.GetJointByName(name="back_bkz")
joint3.set_angle(plant_context,0)

q0 = plant.GetPositions(plant_context)

# q0 = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,
#       0,0.2, 0.2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
#       0. ]


print('type q0:', type(q0))
print(q0)

result = Solve(ik.prog(), q0)
q_sol = result.GetSolution()

print('type result:',type(result))
print(q_sol)

print(f"Success? {result.is_success()}")


visualizer.load()
visualizer.start_recording()
for i in np.arange(10):
  t = i*visualizer.draw_period
  context.SetTime(t)
  plant.SetPositions(plant_context, q0)
  diagram.Publish(context)
for i in np.arange(10):
  t = i*visualizer.draw_period+10*visualizer.draw_period
  context.SetTime(t)
  plant.SetPositions(plant_context, q_sol)
  diagram.Publish(context)

visualizer.stop_recording()
visualizer.publish_recording()

import time
time.sleep(1e5)

