'''
Adapted from http://underactuated.mit.edu/humanoids.html#example1

Implements paper:
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''
from LittleDog import LittleDog
from Atlas import Atlas

import pdb
from functools import partial
import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController
from pydrake.all import (
    MultibodyPlant, JointIndex, RollPitchYaw,  
    PiecewisePolynomial, JacobianWrtVariable, InverseKinematics, RotationMatrix,
    MathematicalProgram, Solve, eq, AutoDiffXd, autoDiffToGradientMatrix, SnoptSolver,
    initializeAutoDiffGivenGradientMatrix, autoDiffToValueMatrix, autoDiffToGradientMatrix,
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
)

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess()

# Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
# That's the behavior of AutoDiffXd in C++, also.
def autoDiffArrayEqual(a,b):
    return np.array_equal(a, b) and np.array_equal(autoDiffToGradientMatrix(a), autoDiffToGradientMatrix(b))

def gait_optimization(robot_ctor):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    robot = robot_ctor(plant)
    visualizer = ConnectMeshcatVisualizer(builder, 
        scene_graph=scene_graph, 
        zmq_url=zmq_url)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    robot.set_home(plant, plant_context)
    visualizer.load()
    diagram.Publish(context)

    q0 = plant.GetPositions(plant_context)
    body_frame = plant.GetFrameByName(robot.get_body_name())

    PositionView = robot.PositionView()
    VelocityView = robot.VelocityView()

    mu = 1 # rubber on rubber
    total_mass = robot.get_total_mass(context)
    gravity = plant.gravity_field().gravity_vector()
    g = 9.81
    
    contact_frame = robot.get_contact_frames()

    in_stance = robot.get_stance_schedule()
    N = robot.get_num_timesteps()
    is_laterally_symmetric = robot.get_laterally_symmetric()
    check_self_collision = robot.get_check_self_collision()
    stride_length = robot.get_stride_length()
    speed = robot.get_speed()
    
    T = 1.5  #stride_length / speed
    # if is_laterally_symmetric:
    #     T = T / 2.0

    prog = MathematicalProgram()        

    # Time steps    
    h = prog.NewContinuousVariables(N-1, "h")
    prog.AddBoundingBoxConstraint(0.5*T/N, 2.0*T/N, h).evaluator().set_description("dt")
    prog.AddLinearConstraint(sum(h) >= .9*T).evaluator().set_description("T")
    prog.AddLinearConstraint(sum(h) <= 1.1*T).evaluator().set_description("T")

    # Create one context per timestep (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant:
    ad_plant = plant.ToAutoDiffXd()

    # Joint positions and velocities
    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    # Joint costs
    q_cost = robot.get_position_cost()
    v_cost = robot.get_velocity_cost()
    for n in range(N):
        # Joint limits
        prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q[:,n]).evaluator().set_description(f"q[:,{n}]")
        # Joint velocity limits
        prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits(), v[:,n]).evaluator().set_description(f"v[:,{n}]")
        # Unit quaternions
        AddUnitQuaternionConstraintOnPlant(plant, q[:,n], prog)
        # Body orientation
        prog.AddConstraint(OrientationConstraint(plant, 
                                                 body_frame, RotationMatrix(),
                                                 plant.world_frame(), RotationMatrix(), 
                                                 robot.max_body_rotation(), context[n]), q[:,n]).evaluator().set_description(f"Body ori q[:,{n}]")
        # Initial guess for all joint angles is the home position
        # prog.SetInitialGuess(q[:,n], q0)  # Solvers get stuck if the quaternion is initialized with all zeros.

        # Running costs:
        prog.AddQuadraticErrorCost(np.diag(q_cost), q0, q[:,n])
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0]*nv, v[:,n])


    lknee = plant.GetJointByName(name="l_leg_kny")
    rknee = plant.GetJointByName(name="r_leg_kny")
    lknee.set_angle(plant_context,0.2)
    rknee.set_angle(plant_context,0.2)
    q0comh = plant.GetPositions(plant_context)

    epsilon = 1e-2
    def setpos(ik, name,pos):
        ik.AddPositionConstraint(
            frameB=plant.GetFrameByName(name),
            p_BQ=np.zeros(3),
            frameA=plant.world_frame(),
            p_AQ_upper=pos+epsilon,
            p_AQ_lower=pos-epsilon)
    def setori(ik, name,rpy):
      rotation = RotationMatrix(rpy)
      ik.AddOrientationConstraint(
              plant.world_frame(), rotation,
              plant.GetFrameByName(name),rotation,
              epsilon)

    def get_q(comh):
        ik = InverseKinematics(plant=plant, with_joint_limits=False)
        setpos(ik, "pelvis",np.array([0.0, 0.0, comh]))
        setpos(ik, "r_foot",np.array([0.0, -0.15, 0.1]))
        setpos(ik, "l_foot",np.array([0.0, 0.15, 0.1]))

        setori(ik, "pelvis",RollPitchYaw(0,0,0))
        setori(ik, "r_foot",RollPitchYaw(0,0,0))
        setori(ik, "l_foot",RollPitchYaw(0,0,0))

        result = Solve(ik.prog(), q0comh)
        q_sol = result.GetSolution()
        # print(f'comh: {comh} \n q_sol: \n {q_sol} \n\n')

        return q_sol


    initcom = np.zeros((3, N))
    comN = 11
    comT = 0.5
    comv0 = 4.9*comT
    h0 = 0.6
    a1 = 2*0.4/comT**2
    comt = [i for i in range(comN)]
    comt = np.array(comt)
    comt = 0.5/(comN-1)*comt
    h1 = h0 + 0.5*a1*comt*comt
    h2 = [h1[-1]+comv0*comt[i] - 0.5*9.8*comt[i]*comt[i] for i in range(comN)]
    h3 = [h2[-1]]*comN
    h1 = np.delete(h1, -1)
    h2 = np.delete(h2, -1)
    comh = np.hstack((h1,h2,h3))

    for n in range(N):
        qcomh = get_q(comh[n])
        prog.SetInitialGuess(q[:,n], qcomh)  # Solvers get stuck if the quaternion is initialized with all zeros.


    qcomh = get_q(comh[0])
    # print(f'comh: {comh[0]} \n qcomh: \n {qcomh} \n\n')
    q_selector = robot.get_initpos_view()
    prog.AddLinearConstraint(eq(q[q_selector,0], qcomh[q_selector])).evaluator().set_description("period")

    # qcomh = get_q(comh[-1])
    # q_selector = robot.get_initpos_view()
    # prog.AddLinearConstraint(eq(q[q_selector,-1], qcomh[q_selector])).evaluator().set_description("period")


    # Make a new autodiff context for this constraint (to maximize cache hits)
    ad_velocity_dynamics_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    def velocity_dynamics_constraint(vars, context_index):
        h, q, v, qn = np.split(vars, [1, 1+nq, 1+nq+nv])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(q, ad_plant.GetPositions(ad_velocity_dynamics_context[context_index])):
                ad_plant.SetPositions(ad_velocity_dynamics_context[context_index], q)
            v_from_qdot = ad_plant.MapQDotToVelocity(ad_velocity_dynamics_context[context_index], (qn - q)/h)
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            v_from_qdot = plant.MapQDotToVelocity(context[context_index], (qn - q)/h)
        return v - v_from_qdot
    for n in range(N-1):
        prog.AddConstraint(
            partial(velocity_dynamics_constraint, context_index=n), 
            lb=[0]*nv, ub=[0]*nv, 
            vars=np.concatenate(([h[n]], q[:,n], v[:,n], q[:,n+1]))).evaluator().set_description(f"veldyn h[{n}]")


    # Contact forces
    num_contacts = robot.get_num_contacts()
    contact_force = [prog.NewContinuousVariables(3, N-1, f"contact{contact}_contact_force") for contact in range(num_contacts)]
    for n in range(N-1):
        for contact in range(num_contacts):
            # Linear friction cone
            prog.AddLinearConstraint(contact_force[contact][0,n] <= mu*contact_force[contact][2,n]).evaluator().set_description("Contactforces")
            prog.AddLinearConstraint(-contact_force[contact][0,n] <= mu*contact_force[contact][2,n]).evaluator().set_description("Contactforces")
            prog.AddLinearConstraint(contact_force[contact][1,n] <= mu*contact_force[contact][2,n]).evaluator().set_description("Contactforces")
            prog.AddLinearConstraint(-contact_force[contact][1,n] <= mu*contact_force[contact][2,n]).evaluator().set_description("Contactforces")
            # normal force >=0, normal_force == 0 if not in_stance
            # max normal force assumed to be 4mg
            prog.AddBoundingBoxConstraint(0, in_stance[contact,n]*4*g*total_mass, contact_force[contact][2,n]).evaluator().set_description("Contactforces")

    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N-1, "comddot")


    initcom[2,:] = comh
    prog.SetInitialGuess(com, initcom)

    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2,0]).evaluator().set_description("InitialCoM")
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,0]).evaluator().set_description("InitialCoM")
    # CoM height
    prog.AddBoundingBoxConstraint(robot.min_com_height(), np.inf, com[2,:]).evaluator().set_description("InitialCoMH")
    # CoM x velocity >= 0
    prog.AddBoundingBoxConstraint(0, np.inf, comdot[0,:]).evaluator().set_description("COMV")
    # CoM final x position
    if is_laterally_symmetric:
        prog.AddBoundingBoxConstraint(stride_length/2.0, stride_length/2.0, com[0,-1]).evaluator().set_description("is_laterally_symmetric")
    else:
        prog.AddBoundingBoxConstraint(stride_length, stride_length, com[0,-1])
    # CoM dynamics
    for n in range(N-1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddConstraint(eq(com[:, n+1], com[:,n] + h[n]*comdot[:,n])).evaluator().set_description("COMdyn")
        prog.AddConstraint(eq(comdot[:, n+1], comdot[:,n] + h[n]*comddot[:,n])).evaluator().set_description("COMdyn")
        prog.AddConstraint(eq(total_mass*comddot[:,n], sum(contact_force[i][:,n] for i in range(4)) + total_mass*gravity)).evaluator().set_description("COMdyn")

    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N-1, "Hdot")
    prog.SetInitialGuess(H, np.zeros((3, N)))
    prog.SetInitialGuess(Hdot, np.zeros((3,N-1)))
    # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
    def angular_momentum_constraint(vars, context_index):
        q, com, Hdot, contact_force = np.split(vars, [nq, nq+3, nq+6])
        contact_force = contact_force.reshape(3, 4, order='F')
        if isinstance(vars[0], AutoDiffXd):
            q = autoDiffToValueMatrix(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[i], [0,0,0], plant.world_frame())
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index], JacobianWrtVariable.kQDot,
                    contact_frame[i], [0, 0, 0], plant.world_frame(), plant.world_frame())
                ad_p_WF = initializeAutoDiffGivenGradientMatrix(p_WF, np.hstack((Jq_WF, np.zeros((3, 18)))))
                torque = torque     + np.cross(ad_p_WF.reshape(3) - com, contact_force[:,i])
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[i], [0,0,0], plant.world_frame())
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:,i])
        return Hdot - torque
    for n in range(N-1):
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hdot[:,n])).evaluator().set_description("centroidal")
        Fn = np.concatenate([contact_force[i][:,n] for i in range(4)])
        lblim = np.array([0,-np.inf,0])
        ublim = np.array([0,np.inf,0])
        prog.AddConstraint(partial(angular_momentum_constraint, context_index=n), lb=lblim, ub=ublim, 
                           vars=np.concatenate((q[:,n], com[:,n], Hdot[:,n], Fn))).evaluator().set_description("centroidal")

    # com == CenterOfMass(q), H = SpatialMomentumInWorldAboutPoint(q, v, com)
    # Make a new autodiff context for this constraint (to maximize cache hits)
    com_constraint_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    def com_constraint(vars, context_index):
        qv, com, H = np.split(vars, [nq+nv, nq+nv+3])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(qv, ad_plant.GetPositionsAndVelocities(com_constraint_context[context_index])):
                ad_plant.SetPositionsAndVelocities(com_constraint_context[context_index], qv)
            com_q = ad_plant.CalcCenterOfMassPositionInWorld(com_constraint_context[context_index])
            H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(com_constraint_context[context_index], com).rotational()
        else:
            if not np.array_equal(qv, plant.GetPositionsAndVelocities(context[context_index])):
                plant.SetPositionsAndVelocities(context[context_index], qv)
            com_q = plant.CalcCenterOfMassPositionInWorld(context[context_index])
            H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(context[context_index], com).rotational()
        return np.concatenate((com_q - com, H_qv - H))
    for n in range(N):
        prog.AddConstraint(partial(com_constraint, context_index=n), 
            lb=np.zeros(6), ub=np.zeros(6), vars=np.concatenate((q[:,n], v[:,n], com[:,n], H[:,n]))).evaluator().set_description("centroidal")

    # TODO: Add collision constraints

    # Kinematic constraints
    def fixed_position_constraint(vars, context_index, frame):
        q, qn = np.split(vars, [nq])
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[context_index+1])):
            plant.SetPositions(context[context_index+1], qn)
        p_WF = plant.CalcPointsPositions(context[context_index], frame, [0,0,0], plant.world_frame())
        p_WF_n = plant.CalcPointsPositions(context[context_index+1], frame, [0,0,0], plant.world_frame())
        if isinstance(vars[0], AutoDiffXd):
            J_WF = plant.CalcJacobianTranslationalVelocity(context[context_index], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            J_WF_n = plant.CalcJacobianTranslationalVelocity(context[context_index+1], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            return initializeAutoDiffGivenGradientMatrix(
                p_WF_n - p_WF, J_WF_n @ autoDiffToGradientMatrix(qn) - J_WF @ autoDiffToGradientMatrix(q))
        else:
            return p_WF_n - p_WF
    for i in range(robot.get_num_contacts()):
        for n in range(N):
            if in_stance[i, n]:
                # foot should be on the ground (world position z=0)
                prog.AddConstraint(PositionConstraint(
                    plant, plant.world_frame(), [-np.inf,-np.inf,0], [np.inf,np.inf,0], 
                    contact_frame[i], [0,0,0], context[n]), q[:,n])
                if n > 0 and in_stance[i, n-1]:
                    # feet should not move during stance.
                    prog.AddConstraint(partial(fixed_position_constraint, context_index=n-1, frame=contact_frame[i]),
                                       lb=np.zeros(3), ub=np.zeros(3), vars=np.concatenate((q[:,n-1], q[:,n]))).evaluator().set_description("kine")
            else:
                min_clearance = 0.01
                prog.AddConstraint(PositionConstraint(plant, plant.world_frame(), [-np.inf,-np.inf,min_clearance], [np.inf,np.inf,np.inf],contact_frame[i],[0,0,0],context[n]), q[:,n])


    # # Periodicity constraints
    # if is_laterally_symmetric:
    #     robot.add_periodic_constraints(prog, q_view, v_view)
    #     print('q_view type:', type(q_view))
    #     # CoM velocity
    #     prog.AddLinearEqualityConstraint(comdot[0,0] == comdot[0,-1]).evaluator().set_description("period")
    #     prog.AddLinearEqualityConstraint(comdot[1,0] == -comdot[1,-1]).evaluator().set_description("period")
    #     prog.AddLinearEqualityConstraint(comdot[2,0] == comdot[2,-1]).evaluator().set_description("period")
    # else:
    #     # Everything except body_x is periodic
    #     q_selector = robot.get_periodic_view()
    #     prog.AddLinearConstraint(eq(q[q_selector,0], q[q_selector,-1])).evaluator().set_description("period")
    #     prog.AddLinearConstraint(eq(v[:,0], v[:,-1])).evaluator().set_description("period")

    # TODO: Set solver parameters (mostly to make the worst case solve times less bad)
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 3e5)
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 200)
    prog.SetSolverOption(snopt, 'Major Feasibility Tolerance', 5e-6)
    prog.SetSolverOption(snopt, 'Major Optimality Tolerance', 1e-4)
    prog.SetSolverOption(snopt, 'Superbasics limit', 2000)
    prog.SetSolverOption(snopt, 'Linesearch tolerance', 0.9)
    # prog.SetSolverOption(snopt, 'Scale option', 2)
    prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

    # TODO a few more costs/constraints from 
    # from https://github.com/RobotLocomotion/LittleDog/blob/master/gaitOptimization.m 

    result = Solve(prog)
    print(f"{result.get_solver_id().name()}: {result.is_success()}")
    # infeasible_constraints = result.GetInfeasibleConstraints(prog)
    # for c in infeasible_constraints:
    #   print(f"infeasible constraint: {c.evaluator().get_description()}")

    #print(result.is_success())  # We expect this to be false if iterations are limited.

    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    visualizer.start_recording()
    num_strides = 1
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
    for t in np.hstack((np.arange(t0, T, visualizer.draw_period), T)):
        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)
        qt = PositionView(q_sol.value(ts))
        if is_laterally_symmetric:
            if stride % 2 == 1:
                qt = robot.HalfStrideToFullStride(qt)
                robot.increment_periodic_view(qt, stride_length/2.0)
            stride = stride // 2
        robot.increment_periodic_view(qt, stride*stride_length)
        plant.SetPositions(plant_context, qt)
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()

gait_optimization(partial(Atlas, simplified=True))

import time
time.sleep(1e7)
