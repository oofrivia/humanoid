import importlib
import sys
from urllib.request import urlretrieve

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(
    server_args=['--ngrok_http_tunnel'] if 'google.colab' in sys.modules else [])

from pydrake.common import set_log_level
set_log_level('off')

import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController

from functools import partial
from pydrake.all import (
    MultibodyPlant, JointIndex, RotationMatrix, PiecewisePolynomial, JacobianWrtVariable,
    MathematicalProgram, Solve, eq, AutoDiffXd, autoDiffToGradientMatrix, SnoptSolver,
    initializeAutoDiffGivenGradientMatrix, autoDiffToValueMatrix, autoDiffToGradientMatrix,
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
)
from pydrake.common.containers import namedview

# Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
# That's the behavior of AutoDiffXd in C++, also.
def autoDiffArrayEqual(a,b):
    return np.array_equal(a, b) and np.array_equal(autoDiffToGradientMatrix(a), autoDiffToGradientMatrix(b))


#import

############################################################################################################################################

def set_home(plant, context):
    hip = 0.
    knee = 0.
    ankle = 0
    plant.GetJointByName("joint_left_hip").set_angle(context, hip)
    plant.GetJointByName("joint_right_hip").set_angle(context, -hip)
    plant.GetJointByName("joint_left_knee").set_angle(context, knee)
    plant.GetJointByName("joint_right_knee").set_angle(context, -knee)
    plant.GetJointByName("joint_left_ankle").set_angle(context, ankle)
    plant.GetJointByName("joint_right_ankle").set_angle(context, -ankle)
    plant.SetFreeBodyPose(context, plant.GetBodyByName("body"), RigidTransform([0, 0, 0.39]))

def main():
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = Parser(plant)
    BipedRobot1015=parser.AddModelFromFile('robots/BipedRobot1015/BipedRobot1015.urdf')
    plant.Finalize()

    visualizer = ConnectMeshcatVisualizer(builder, 
        scene_graph=scene_graph, 
        zmq_url=zmq_url)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    set_home(plant, plant_context)

    mu = 1.0 # Coefficient of friction
    N_d = 4 # friction cone approximated as a i-pyramid
    N_f = 3 # contact force dimension
    N_c = 2 # num contact points****

    ''' Eq(9)'''
    '''
    # Assume flat ground for now
    n = np.array([
        [0],
        [0],
        [1.0]])
    d = np.array([
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, 0.0, 0.0]])
    v = np.zeros((N_d, N_c, N_f))
    for i in range(N_d):
        for j in range(N_c):
            v[i,j] = (n+mu*d)[:,i]
    '''
    
    H = plant.CalcMassMatrixViaInverseDynamics(plant_context)
    #Coriolis, centripetal, and gyroscopic effects - Gravity
    C = plant.CalcBiasTerm(plant_context) - plant.CalcGravityGeneralizedForces(plant_context)
    # Calculate values that don't depend on context
    B = plant.MakeActuationMatrix()
    # From np.sort(np.nonzero(B_7)[0]) we know that indices 0-5 are the unactuated 6 DOF floating base and 6-35 are the actuated 30 DOF robot joints
    v_idx_act = 6 # Start index of actuated joints in generalized velocities
    H_f = H[0:v_idx_act,:]
    H_a = H[v_idx_act:,:]
    C_f = C[0:v_idx_act]
    C_a = C[v_idx_act:]
    B_a = B[v_idx_act:,:]
    B_a_inv = np.linalg.inv(B_a)
    #Phi_f_T = Phi.T[0:v_idx_act:,:]
    #Phi_a_T = Phi.T[v_idx_act:,:]
    q = plant.GetPositions(plant_context)
    qd = plant.GetVelocities(plant_context)
    nv = plant.num_velocities()
    vd_d = np.zeros(nv)



    tau = B_a_inv.dot(H_a.dot(np.zeros(12)) + C_a - np.zeros(6))
    #tau=B_a_inv.dot(H_a.dot(np.zeros(12)) + C_a - [98.1, 127.53, 142.245, 98.1, 127.53, 142.245])
    #tauG = plant.CalcInverseDynamics(plant_context,[0,0,0,0,0,0],)




    # print('MassMatrix:',H_a)
    # print(H_a.shape)
    # print('Bia:',plant.CalcBiasTerm(plant_context))
    # print(plant.CalcBiasTerm(plant_context).shape)
    # print('Gravity:',plant.CalcGravityGeneralizedForces(plant_context))
    # print(plant.CalcGravityGeneralizedForces(plant_context).shape)
    # print('B_a:',B_a)
    # print('B_a_inv:',B_a_inv)
    # print('tau:',tau)






    B_inv = np.vstack([0*np.identity(6),B_a_inv])
    # tauAll = B_inv.dot(H.dot(np.zeros(12)) + C - np.zeros(6))
    print(H.shape)
    print('C_f:',C_f)
    print('B:',B)
    print('B_inv:',B_inv)
    # print('tauAll:',tauAll)

    np.save('TrajData/C_f.npy',C_f)
    fileC_f = np.load('TrajData/C_f.npy')
    print('fileC_f:',fileC_f)

'''
    visualizer.load()
    diagram.Publish(context)

    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))

    visualizer.start_recording()
    num_strides = 1
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides
    for t in (np.arange(t0, T, visualizer.draw_period)):
        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)
        qt = PositionView(q_sol.value(ts))
        plant.SetPositions(plant_context, qt)
        diagram.Publish(context)
    visualizer.stop_recording()
    
    visualizer.publish_recording()
'''
if __name__ == "__main__":
    main()


import time
time.sleep(1e7)

