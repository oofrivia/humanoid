import importlib
import sys
from urllib.request import urlretrieve

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(
    server_args=['--ngrok_http_tunnel'] if 'google.colab' in sys.modules else [])

from pydrake.common import set_log_level
set_log_level('off');

import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController
from pydrake.all import FindResourceOrThrow

def set_home(plant, context):
    hip_roll = .1;
    hip_pitch = 1;
    knee = 1.55;
    plant.GetJointByName("front_right_hip_roll").set_angle(context, -hip_roll)
    plant.GetJointByName("front_right_hip_pitch").set_angle(context, hip_pitch)
    plant.GetJointByName("front_right_knee").set_angle(context, -knee)
    plant.GetJointByName("front_left_hip_roll").set_angle(context, hip_roll)
    plant.GetJointByName("front_left_hip_pitch").set_angle(context, hip_pitch)
    plant.GetJointByName("front_left_knee").set_angle(context, -knee)
    plant.GetJointByName("back_right_hip_roll").set_angle(context, -hip_roll)
    plant.GetJointByName("back_right_hip_pitch").set_angle(context, -hip_pitch)
    plant.GetJointByName("back_right_knee").set_angle(context, knee)
    plant.GetJointByName("back_left_hip_roll").set_angle(context, hip_roll)
    plant.GetJointByName("back_left_hip_pitch").set_angle(context, -hip_pitch)
    plant.GetJointByName("back_left_knee").set_angle(context, knee)
    plant.SetFreeBodyPose(context, plant.GetBodyByName("body"), RigidTransform([0, 0, 0.146]))

def run_pid_control():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-4)
    parser = Parser(plant)
    parser.AddModelFromFile('robots/littledog/LittleDog.urdf')
    parser.AddModelFromFile('robots/littledog/ground.urdf')
    plant.Finalize()

    # Add a PD Controller
    kp = 0.1*np.ones(12)
    ki = 0.01*np.ones(12)
    kd = 0.5*np.ones(12)
    kd[-4:] = .16 # use lower gain for the knee joints
    # Select the joint states (and ignore the floating-base states)
    S = np.zeros((24, 37))
    S[:12, 7:19] = np.eye(12)
    S[12:, 25:] = np.eye(12)
    control = builder.AddSystem(PidController(
        kp=kp, ki=ki, kd=kd, 
        state_projection=S, 
        output_projection=plant.MakeActuationMatrix()[6:,:].T))

    builder.Connect(plant.get_state_output_port(), control.get_input_port_estimated_state())
    builder.Connect(control.get_output_port(), plant.get_actuation_input_port())

    visualizer = ConnectMeshcatVisualizer(builder, scene_graph=scene_graph, zmq_url=zmq_url)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    set_home(plant, plant_context)
    x0 = S @ plant.get_state_output_port().Eval(plant_context)
    control.get_input_port_desired_state().FixValue(control.GetMyContextFromRoot(context), x0)

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(20.0)
    print('Done simulation!')

run_pid_control()

import time
time.sleep(1e7)
