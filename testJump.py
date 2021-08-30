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


def set_home(plant, context):
    hip_roll = .1;
    # hip_pitch = 1;
    # knee = 1.55;
    hip_pitch = 1.5;
    knee = 2;
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


from functools import partial
from pydrake.all import (
    MultibodyPlant, JointIndex, RotationMatrix, PiecewisePolynomial, JacobianWrtVariable,
    MathematicalProgram, Solve, eq, AutoDiffXd, autoDiffToGradientMatrix, SnoptSolver,
    initializeAutoDiff, initializeAutoDiffGivenGradientMatrix, autoDiffToValueMatrix, autoDiffToGradientMatrix,
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
)
from pydrake.common.containers import namedview

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

def gait_optimization(gait = 'walking_trot'):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = Parser(plant)
    littledog = parser.AddModelFromFile('robots/littledog/LittleDog.urdf')
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
    body_frame = plant.GetFrameByName("body")

    PositionView = MakeNamedViewPositions(plant, "Positions")
    VelocityView = MakeNamedViewVelocities(plant, "Velocities")

    mu = 1 # rubber on rubber
    total_mass = sum(plant.get_body(index).get_mass(context) for index in plant.GetBodyIndices(littledog))
    gravity = plant.gravity_field().gravity_vector()

    nq = 12
    foot_frame = [
        plant.GetFrameByName('front_left_foot_center'),
        plant.GetFrameByName('front_right_foot_center'),
        plant.GetFrameByName('back_left_foot_center'),
        plant.GetFrameByName('back_right_foot_center')]

    # setup gait
    is_laterally_symmetric = False
    check_self_collision = False

    # jump
    N = 25
    LiftKont = 7
    TouchKont = 16
    in_stance = np.zeros((4, N))
    in_stance[0, :LiftKont] = 1
    in_stance[1, :LiftKont] = 1
    in_stance[2, :LiftKont] = 1
    in_stance[3, :LiftKont] = 1

    in_stance[0, TouchKont:] = 1
    in_stance[1, TouchKont:] = 1
    in_stance[2, TouchKont:] = 1
    in_stance[3, TouchKont:] = 1

    stride_length = 0.
    check_self_collision = True


    T = 1.5

    prog = MathematicalProgram()        

    # Time steps    
    h = prog.NewContinuousVariables(N-1, "h")
    prog.AddBoundingBoxConstraint(0.01, 2.0*T/N, h)
    prog.AddLinearConstraint(sum(h[LiftKont:TouchKont]) >= .5*T)
    prog.AddLinearConstraint(sum(h) >= .9*T)
    prog.AddLinearConstraint(sum(h) <= 1.1*T)

    # Create one context per timestep (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant:
    ad_plant = plant.ToAutoDiffXd()

    # Contact forces
    contact_force = [prog.NewContinuousVariables(3, N-1, f"foot{foot}_contact_force") for foot in range(4)]
    for n in range(N-1):
        for foot in range(4):
            # Linear friction cone
            prog.AddLinearConstraint(contact_force[foot][0,n] <= mu*contact_force[foot][2,n])
            prog.AddLinearConstraint(-contact_force[foot][0,n] <= mu*contact_force[foot][2,n])
            prog.AddLinearConstraint(contact_force[foot][1,n] <= mu*contact_force[foot][2,n])
            prog.AddLinearConstraint(-contact_force[foot][1,n] <= mu*contact_force[foot][2,n])
            # normal force >=0, normal_force == 0 if not in_stance
            prog.AddBoundingBoxConstraint(0, in_stance[foot,n]*4*9.81*total_mass, contact_force[foot][2,n])            

    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N-1, "comddot")
    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2,0])
    prog.AddBoundingBoxConstraint(0, 0, com[:2,-1])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,0])
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,-1])
    # End CoM acc == 0
    prog.AddBoundingBoxConstraint(0, 0, comddot[2,-1])


    # CoM height for # Kinematic constraints
    for n in range(LiftKont):
      prog.AddBoundingBoxConstraint(.025 , 0.15, com[2,n])
    for n in range(LiftKont, TouchKont, 1):
      prog.AddBoundingBoxConstraint(.025 , np.inf, com[2,n])
    for n in range(TouchKont, N, 1):
      prog.AddBoundingBoxConstraint(.025 , 0.15, com[2,n])

    # CoM dynamics
    for n in range(N-1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddConstraint(eq(com[:, n+1], com[:,n] + h[n]*comdot[:,n]))
        prog.AddConstraint(eq(comdot[:, n+1], comdot[:,n] + h[n]*comddot[:,n]))
        prog.AddConstraint(eq(total_mass*comddot[:,n], sum(contact_force[i][:,n] for i in range(4)) + total_mass*gravity))


    def jerk(vars):
        comddot,comddotN = np.split(vars, [3])
        jerk = comddotN - comddot
        return (jerk[0]*jerk[0] + jerk[1]*jerk[1] + jerk[2]*jerk[2])*1e4

    for n in range(N-2):
        prog.AddCost(jerk, vars=np.concatenate((comddot[:,n],comddot[:,n+1])))

    # for n in range(9,16,1):
    #     prog.AddLinearCost(-1e3*com[2,n])

    for i in range(4):
      for n in range(N-1):
          prog.AddQuadraticErrorCost(np.diag([1e2,1e2,1e2]), [0]*3, contact_force[i][:,n])





    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N-1, "Hdot")
    prog.SetInitialGuess(H, np.zeros((3, N)))
    prog.SetInitialGuess(Hdot, np.zeros((3,N-1)))
    def angular_momentum_constraint(vars, context_index):
        com, Hdot, contact_force = np.split(vars, [3, 6])
        contact_force = contact_force.reshape(3, 4, order='F')
        if isinstance(vars[0], AutoDiffXd):
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(plant_context, foot_frame[i], [0,0,0], plant.world_frame())
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    plant_context, JacobianWrtVariable.kQDot,
                    foot_frame[i], [0, 0, 0], plant.world_frame(), plant.world_frame())
                print('Jq_WF.shape: ',Jq_WF.shape)
                ad_p_WF = initializeAutoDiffGivenGradientMatrix(p_WF, np.hstack((Jq_WF, np.zeros((3, 18)))))
                # ad_p_WF = initializeAutoDiff(p_WF)
                print('ad_p_WF.shape: ',ad_p_WF.shape)
                GM = autoDiffToGradientMatrix(ad_p_WF)
                print('GM.shape: ',GM.shape)
                print('GM: ',GM)
                print('\n ')

                torque = torque     + np.cross(ad_p_WF.reshape(3) - com, contact_force[:,i])
                # print('ad torque: ',i , type(torque))
                # print('torque: ', torque)

        else:
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(plant_context, foot_frame[i], [0,0,0], plant.world_frame())
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:,i])
                # print('noad torque: ',i , type(torque))
                # print('torque: ', torque)
        # return (torque[0]*torque[0] + torque[1]*torque[1] + torque[2]*torque[2])*1e4
        return Hdot - torque

    for n in range(N-1):
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hdot[:,n]))
        Fn = np.concatenate([contact_force[i][:,n] for i in range(4)])
        prog.AddConstraint(partial(angular_momentum_constraint, context_index=n), lb=np.zeros(3), ub=np.zeros(3), 
                           vars=np.concatenate((com[:,n], Hdot[:,n], Fn)))
        # prog.AddCost(partial(angular_momentum_constraint, context_index=n),
        #                    vars=np.concatenate(( com[:,n], Hdot[:,n], Fn)))

    for n in range(N):
        prog.AddQuadraticErrorCost(np.diag([1e2,1e2,1e2]), [0]*3, H[:,n])












    # TODO: Set solver parameters (mostly to make the worst case solve times less bad)
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 1e6 )
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 1e4 )
    prog.SetSolverOption(snopt, 'Major Feasibility Tolerance', 5e-6)
    prog.SetSolverOption(snopt, 'Major Optimality Tolerance', 1e-4)
    prog.SetSolverOption(snopt, 'Superbasics limit', 2000)
    prog.SetSolverOption(snopt, 'Linesearch tolerance', 0.9)
    # prog.SetSolverOption(snopt, 'Scale option', 2)
    prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

    from shutil import copyfile
    source = 'snopt.out'
    target = 'snopt.out_old'
    copyfile(source, target)

    f=open('snopt.out','w')
    f.truncate()

    # TODO a few more costs/constraints from 
    # from https://github.com/RobotLocomotion/LittleDog/blob/master/gaitOptimization.m 

    result = Solve(prog)

    infeasible_constraints = result.GetInfeasibleConstraints(prog)
    for c in infeasible_constraints:
      # print(f"infeasible constraint: {c.evaluator().get_description()}")
      print(f"infeasible constraint: {c}")
    print(result.get_solver_id().name(),': ', result.is_success())



    h_sol = result.GetSolution(h)

    contact_force_sol = [result.GetSolution(contact_force[i]) for i in range(4)]
    com_sol = result.GetSolution(com)
    comdot_sol = result.GetSolution(comdot)
    comddot_sol = result.GetSolution(comddot)


    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    i = 0
    print('h_sol sum: \n',sum(h_sol[LiftKont:TouchKont]))
    print('h_sol: \n',h_sol)
    print('contact_force_sol0 z: \n',contact_force_sol[0][2,:])
    print('contact_force_sol1 z: \n',contact_force_sol[1][2,:])
    print('contact_force_sol2 z: \n',contact_force_sol[2][2,:])
    print('contact_force_sol3 z: \n',contact_force_sol[3][2,:])
    print('contact_force_sol3 x: \n',contact_force_sol[3][0,:])
    print('contact_force_sol3 y: \n',contact_force_sol[3][1,:])
    print('com_sol x: \n',com_sol[0])
    print('com_sol y: \n',com_sol[1])
    print('com_sol z: \n',com_sol[2])
    print('comdot_sol z: \n',comdot_sol[2])
    print('comddot_sol z: \n',comddot_sol[2])















    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    # q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    visualizer.start_recording()
    num_strides = 1
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
    for t in (np.arange(t0, T, visualizer.draw_period)):
        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)
        # qt = PositionView(q_sol.value(ts))

        # qt.body_x += stride*stride_length
        # plant.SetPositions(plant_context, qt)
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()
    
# Try them all!  The last two could use a little tuning.

gait_optimization('jump')
# gait_optimization('walking_trot')
# gait_optimization('running_trot')
# gait_optimization('rotary_gallop')  
# gait_optimization('bound')

import time
time.sleep(1e7)

