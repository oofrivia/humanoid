import importlib
import sys
from tokenize import Double
import numpy as np

from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController
from pydrake.all import (
    MultibodyPlant, JointIndex, RotationMatrix, PiecewisePolynomial, JacobianWrtVariable,
    MathematicalProgram, Solve, eq, AutoDiffXd, autoDiffToGradientMatrix, SnoptSolver,
    initializeAutoDiffGivenGradientMatrix, autoDiffToValueMatrix, autoDiffToGradientMatrix,
    PiecewiseQuaternionSlerp, AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
)
from pydrake.common.eigen_geometry import Quaternion

from pydrake.common.containers import namedview

from pydrake.common import set_log_level
set_log_level('off');

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(
    server_args=['--ngrok_http_tunnel'] if 'google.colab' in sys.modules else [])




# Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
# That's the behavior of AutoDiffXd in C++, also.
def autoDiffArrayEqual(a,b):
    return np.array_equal(a, b) and np.array_equal(autoDiffToGradientMatrix(a), autoDiffToGradientMatrix(b))

# TODO: promote this to drake (and make a version with model_instance)
def MakeNamedViewPositions(mbp, view_name):
    names = [None]*mbp.num_positions()
    for ind in range(mbp.num_joints()): 
        joint = mbp.get_joint(JointIndex(ind))
        # TODO: Handle planar joints, etc.
        assert(joint.num_positions() == 1)
        names[joint.position_start()] = joint.name()
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_positions_start()
        body_name = body.name()
        names[start] = body_name+'_qw'
        names[start+1] = body_name+'_qx'
        names[start+2] = body_name+'_qy'
        names[start+3] = body_name+'_qz'
        names[start+4] = body_name+'_x'
        names[start+5] = body_name+'_y'
        names[start+6] = body_name+'_z'
    return namedview(view_name, names)

def MakeNamedViewVelocities(mbp, view_name):
    names = [None]*mbp.num_velocities()
    for ind in range(mbp.num_joints()): 
        joint = mbp.get_joint(JointIndex(ind))
        # TODO: Handle planar joints, etc.
        assert(joint.num_velocities() == 1)
        names[joint.velocity_start()] = joint.name()
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_velocities_start() - mbp.num_positions()
        body_name = body.name()
        names[start] = body_name+'_wx'
        names[start+1] = body_name+'_wy'
        names[start+2] = body_name+'_wz'
        names[start+3] = body_name+'_vx'
        names[start+4] = body_name+'_vy'
        names[start+5] = body_name+'_vz'
    return namedview(view_name, names)


def set_home(plant, context):
    hip = 0.
    knee = -0.
    ankle = -0.
    plant.GetJointByName("joint_hip").set_angle(context, hip)
    plant.GetJointByName("joint_knee").set_angle(context, knee)
    plant.GetJointByName("joint_ankle").set_angle(context, ankle)
    plant.SetFreeBodyPose(context, plant.GetBodyByName("body"), RigidTransform([0, 0, 0.13]))


def JumpControl():

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = Parser(plant)
    littledog = parser.AddModelFromFile('robots/singleleg/singlelegtwocontact.urdf')
    plant.Finalize()
    visualizer = ConnectMeshcatVisualizer(builder, 
        scene_graph=scene_graph, 
        zmq_url=zmq_url)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    set_home(plant, plant_context)
    visualizer.load()
    diagram.Publish(context)

    q0 = plant.GetPositions(plant_context)
    v0 = plant.GetVelocities(plant_context)
    com_q0 = plant.CalcCenterOfMassPositionInWorld(plant_context)

    PositionView = MakeNamedViewPositions(plant, "Positions")
    VelocityView = MakeNamedViewVelocities(plant, "Velocities")

    mu = 1 # rubber on rubber
    gravity = plant.gravity_field().gravity_vector()
    total_mass = sum(plant.get_body(index).get_mass(context) for index in plant.GetBodyIndices(littledog))
    # print(f'total_mass: {total_mass}')

    body_frame = plant.GetFrameByName("body")
    foot_frame = [
        plant.GetFrameByName('frame_toe'),
        plant.GetFrameByName('frame_heel')]
    num_contacts = len(foot_frame)

    # jump
    T = 1.5 # total time
    dt = 0.001

    N = 25
    Stride = 0.05
    LiftKont = 7
    TouchKont = 16
    in_stance = np.zeros((2, N))
    in_stance[0, :LiftKont] = 1
    in_stance[1, :LiftKont] = 1

    in_stance[0, TouchKont:] = 1
    in_stance[1, TouchKont:] = 1


    Traj = np.load('TrajData/Traj.npz')
    h_sol = Traj['h_sol']
    myq_sol = Traj['myq_sol']
    myv_sol = Traj['myv_sol']
    com_sol = Traj['com_sol']
    comdot_sol = Traj['comdot_sol']
    comddot_sol = Traj['comddot_sol']
    H_sol = Traj['H_sol']
    Hdot_sol = Traj['Hdot_sol']
    contact_force_sol = Traj['contact_force_sol']

    t_sol = np.cumsum(np.concatenate(([0],h_sol)))
    Ncntl = t_sol[-1]/dt

    q_sol = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_sol, myq_sol, False)

    Q_sol = []
    for i in range(N):
        Q = Quaternion(myq_sol[:4,i])
        Q_sol.append(Q)
    Q_slerp = PiecewiseQuaternionSlerp(t_sol.tolist(), Q_sol)



    q_poly = []
    v_poly = []
    t_poly = []
    for t in (np.arange(0, t_sol[-1], dt)):
        qt_minus = q_sol.value(t)
        qt_plus = q_sol.value(t+dt)
        q_poly.append(qt_minus.squeeze().transpose())

        vt = (qt_plus-qt_minus)/dt
        vt = vt.squeeze().transpose()
        angularVel = Q_slerp.angular_velocity(t)
        vt = np.delete(vt, [0,1,2,3])
        vt = np.hstack((angularVel, vt))

        v_poly.append(vt)
        t_poly.append(t)

    acc_poly = []
    for i in range(len(v_poly)-1):
        acc = (v_poly[i+1]-v_poly[i])/dt
        acc_poly.append(acc)

    t_poly = np.array(t_poly)
    q_poly = np.array(q_poly)
    v_poly = np.array(v_poly)
    acc_poly = np.array(acc_poly)
    acc_poly = np.vstack((acc_poly, acc_poly[-1]))






    qv = np.concatenate((q_poly[0], v_poly[0]))
    # qv = np.concatenate((q0, v0))
    plant.SetPositionsAndVelocities(plant_context, qv)
    
    H = plant.CalcMassMatrixViaInverseDynamics(plant_context)
    C = plant.CalcBiasTerm(plant_context) - plant.CalcGravityGeneralizedForces(plant_context)
    B = plant.MakeActuationMatrix()

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
    tau = B_a_inv.dot(H_a.dot(np.zeros(9)) + C_a - np.zeros(3))

    print('q0 ',q0)
    print('v0 ',v0)
    print('C ',C)
    print('tau ',tau)












    # print('h_sol lift: ',sum(h_sol[0:LiftKont]))
    # print('h_sol fly: ',sum(h_sol[LiftKont:TouchKont]))
    # print('h_sol touch: ',sum(h_sol[TouchKont:-1]))

    # print(h_sol)
    # print(t_sol)
    # print(type(t_poly))
    # print(type(v_poly))
    # print(len(v_poly))
    # print(len(acc_poly))



    # import matplotlib.pyplot as plt
    # plt.ion()
    # ax1 = plt.subplot(221)
    # plt.plot(t_poly , 10*q_poly[:,6] )
    # plt.plot(t_poly , v_poly[:,6])
    # plt.plot(t_poly , 0.1*acc_poly[:,6])
    # plt.legend(['pos', 'vel', 'acc'])
    # ax1.set_title('Torso z')

    # ax2 = plt.subplot(222)
    # plt.plot(t_poly , 10*q_poly[:,7] )
    # plt.plot(t_poly , v_poly[:,7])
    # plt.plot(t_poly , 0.1*acc_poly[:,7])
    # plt.legend(['pos', 'vel', 'acc'])
    # ax2.set_title('hip')

    # ax3 = plt.subplot(223)
    # plt.plot(t_poly , 10*q_poly[:,8] )
    # plt.plot(t_poly , v_poly[:,8])
    # plt.plot(t_poly , 0.1*acc_poly[:,8])
    # plt.legend(['pos', 'vel', 'acc'])
    # ax3.set_title('knee')

    # ax4 = plt.subplot(224)
    # plt.plot(t_poly , 10*q_poly[:,8] )
    # plt.plot(t_poly , v_poly[:,8])
    # plt.plot(t_poly , 0.1*acc_poly[:,8])
    # plt.legend(['pos', 'vel', 'acc'])
    # ax4.set_title('ankle')
    # plt.ioff()

    # plt.show()





    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.array([0.1]*24)
    visualizer.start_recording()
    num_strides = 1
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides
    for t in (np.arange(t0, T, visualizer.draw_period)):

        context.SetTime(t)
        plant.SetPositions(plant_context, q0)
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()


    # import matplotlib.pyplot as plt
    # comddot_sol = np.hstack((comddot_sol, np.array([[0],[0],[0]])))
    # plt.plot(t_sol , 0.1*comddot_sol[2,:], '*-')
    # plt.plot(t_sol , comdot_sol[2,:], '*-')
    # plt.plot(t_sol , 10*com_sol[2,:], '*-')
    # plt.legend(['comddot', 'comdot', 'com'])
    # plt.show()




JumpControl()

import time
time.sleep(1e7)

