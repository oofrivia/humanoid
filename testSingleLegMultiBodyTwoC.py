import importlib
import sys
from urllib.request import urlretrieve

from numpy.core.arrayprint import printoptions

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(
    server_args=['--ngrok_http_tunnel'] if 'google.colab' in sys.modules else [])

from pydrake.common import set_log_level
set_log_level('off');

import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController


def set_home(plant, context):
    hip = 0.2
    knee = -0.4
    ankle = -0.2
    plant.GetJointByName("joint_hip").set_angle(context, hip)
    plant.GetJointByName("joint_knee").set_angle(context, knee)
    plant.GetJointByName("joint_ankle").set_angle(context, ankle)
    plant.SetFreeBodyPose(context, plant.GetBodyByName("body"), RigidTransform([0, 0, 0.13]))


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
    print(f'total_mass: {total_mass}')

    body_frame = plant.GetFrameByName("body")
    foot_frame = [
        plant.GetFrameByName('frame_toe'),
        plant.GetFrameByName('frame_heel')]
    num_contacts = len(foot_frame)




    # jump
    T = 1.5 # total time
    N = 25
    Stride = 0.05
    LiftKont = 7
    TouchKont = 16
    in_stance = np.zeros((2, N))
    in_stance[0, :LiftKont] = 1
    in_stance[1, :LiftKont] = 1

    in_stance[0, TouchKont:] = 1
    in_stance[1, TouchKont:] = 1


    ##################################################################################################################
    prog = MathematicalProgram()        

    # Time steps    
    h = prog.NewContinuousVariables(N-1, "h")
    prog.AddBoundingBoxConstraint(0.01, 2.0*T/N, h)
    prog.AddLinearConstraint(sum(h[LiftKont:TouchKont]) >= 0.35)
    prog.AddLinearConstraint(sum(h) >= .9*T)
    prog.AddLinearConstraint(sum(h) <= 1.1*T)


    # Create one context per timestep (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant:
    ad_plant = plant.ToAutoDiffXd()

    # Joint positions and velocities
    nq = plant.num_positions()
    nv = plant.num_velocities()
    print(f'nq: {nq}   nv: {nv}')

    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    # Joint costs
    q_cost = PositionView([1]*nq)
    v_cost = VelocityView([1]*nv)
    q_cost.body_z = 0
    q_cost.body_qx = 0
    q_cost.body_qy = 0
    q_cost.body_qz = 0
    q_cost.body_qw = 0
    q_cost.joint_hip = 5
    q_cost.joint_knee = 5
    q_cost.joint_ankle = 5
    # v_cost.body_vx = 0
    # v_cost.body_vz = 0
    v_cost.body_wx = 0
    v_cost.body_wy = 0
    v_cost.body_wz = 0


    for n in range(N):
        # Joint limits
        prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q[:,n])
        # Joint velocity limits
        # prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits(), v[:,n])
        # Unit quaternions
        AddUnitQuaternionConstraintOnPlant(plant, q[:,n], prog)
        # Body orientation
        prog.AddConstraint(OrientationConstraint(plant, 
                                                 body_frame, RotationMatrix(),
                                                 plant.world_frame(), RotationMatrix(), 
                                                 0.1, context[n]), q[:,n])
        # Initial guess for all joint angles is the home position

        qt = q0
        qt[4] = 1.0*n*Stride/(N-1)
        prog.SetInitialGuess(q[:,n], qt)  # Solvers get stuck if the quaternion is initialized with all zeros.
        prog.SetInitialGuess(v[:,n], v0)

        # Running costs:
        prog.AddQuadraticErrorCost(np.diag(q_cost), qt, q[:,n])
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0]*nv, v[:,n])





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

    veloCon = []
    for n in range(N-1):
        velocon = prog.AddConstraint(
            partial(velocity_dynamics_constraint, context_index=n), 
            lb=[0]*nv, ub=[0]*nv, 
            vars=np.concatenate(([h[n]], q[:,n], v[:,n], q[:,n+1])))
        veloCon.append(velocon)







    ##################################################################################################################
    # Contact forces
    contact_force = [prog.NewContinuousVariables(3, N-1, f"foot{foot}_contact_force") for foot in range(num_contacts)]
    forceScale = 9.81*total_mass
    for n in range(N-1):
        for foot in range(num_contacts):
            # Linear friction cone
            prog.AddLinearConstraint(contact_force[foot][0,n] <= mu*contact_force[foot][2,n])
            prog.AddLinearConstraint(-contact_force[foot][0,n] <= mu*contact_force[foot][2,n])
            prog.AddLinearConstraint(contact_force[foot][1,n] <= mu*contact_force[foot][2,n])
            prog.AddLinearConstraint(-contact_force[foot][1,n] <= mu*contact_force[foot][2,n])
            # normal force >=0, normal_force == 0 if not in_stance
            prog.AddBoundingBoxConstraint(0, in_stance[foot,n]*10, contact_force[foot][2,n]) 

    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N-1, "comddot")
    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2,0])
    prog.AddBoundingBoxConstraint(0.05, 0.05, com[0,-1])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,0])
    # All CoM x vel >= 0
    prog.AddBoundingBoxConstraint(0, np.inf, comdot[0,:])
    # Final CoMddot z == 0
    prog.AddBoundingBoxConstraint(0, 0, comddot[2,-1])

    # CoM height for # Kinematic constraints
    for n in range(LiftKont+2):
      prog.AddBoundingBoxConstraint(.05 , 0.12, com[2,n])
    for n in range(LiftKont+2, TouchKont, 1):
      prog.AddBoundingBoxConstraint(.05 , np.inf, com[2,n])
    for n in range(TouchKont, N, 1):
      prog.AddBoundingBoxConstraint(.05 , 0.12, com[2,n])


    # CoM dynamics
    for n in range(N-1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddConstraint(eq(com[:, n+1], com[:,n] + h[n]*comdot[:,n] + 1/2*h[n]*h[n]*comddot[:,n]))
        prog.AddConstraint(eq(comdot[:, n+1], comdot[:,n] + h[n]*comddot[:,n]))
        prog.AddConstraint(eq(total_mass*comddot[:,n], sum(forceScale*contact_force[i][:,n] for i in range(num_contacts)) + total_mass*gravity))



    for i in range(num_contacts):
      for n in range(N-1):
          prog.AddQuadraticErrorCost(np.diag([1e2,1e2,1e2]), [0]*3, contact_force[i][:,n])

    def jerk(vars):
        comddot,comddotN = np.split(vars, [3])
        jerk = comddotN - comddot
        return (jerk[0]*jerk[0] + jerk[1]*jerk[1] + jerk[2]*jerk[2])*1e3

    for n in range(N-2):
        prog.AddCost(jerk, vars=np.concatenate((comddot[:,n],comddot[:,n+1])))

    def nvjerk(vars):
        jointv, jointvn, jointvn1 = np.split(vars, [nv,2*nv])
        acc = jointvn - jointv
        acc1 = jointvn1 - jointvn
        jerk = acc1-acc
        norm = 0
        for i in range(nv-6):
          norm += jerk[i+6]*jerk[i+6]*1e3
        return norm

    for n in range(N-3):
        prog.AddCost(nvjerk, vars=np.concatenate((v[:,n],v[:,n+1],v[:,n+2])))

    # for n in range(9,16,1):
    #     prog.AddLinearCost(-1e3*com[2,n])






    ##################################################################################################################
    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N-1, "Hdot")
    prog.SetInitialGuess(H, np.zeros((3, N)))
    prog.SetInitialGuess(Hdot, np.zeros((3,N-1)))
    # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
    for n in range(N-1):
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hdot[:,n]))


    def angular_momentum_constraint(vars, context_index):
        q, com, Hdot, contact_force = np.split(vars, [nq, nq+3, nq+6])
        contact_force = contact_force.reshape(3, num_contacts, order='F')
        if isinstance(vars[0], AutoDiffXd):
            q = autoDiffToValueMatrix(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(num_contacts):
                p_WF = plant.CalcPointsPositions(context[context_index], foot_frame[i], [0,0,0], plant.world_frame())
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index], JacobianWrtVariable.kQDot,
                    foot_frame[i], [0, 0, 0], plant.world_frame(), plant.world_frame())
                ad_p_WF = initializeAutoDiffGivenGradientMatrix(p_WF, np.hstack((Jq_WF, np.zeros((3, 12)))))
                torque = torque     + np.cross(ad_p_WF.reshape(3) - com, forceScale*contact_force[:,i])
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(num_contacts):
                p_WF = plant.CalcPointsPositions(context[context_index], foot_frame[i], [0,0,0], plant.world_frame())
                torque += np.cross(p_WF.reshape(3) - com, forceScale*contact_force[:,i])
        return Hdot - torque
    for n in range(N-1):
        Fn = np.concatenate([contact_force[i][:,n] for i in range(num_contacts)])
        prog.AddConstraint(partial(angular_momentum_constraint, context_index=n), lb=np.zeros(3), ub=np.zeros(3), 
                           vars=np.concatenate((q[:,n], com[:,n], Hdot[:,n], Fn)))

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
            lb=np.zeros(6), ub=np.zeros(6), vars=np.concatenate((q[:,n], v[:,n], com[:,n], H[:,n])))









    ##################################################################################################################
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
    for i in range(num_contacts):
        for n in range(N):
            if in_stance[i, n] or in_stance[i, n-1] :
                # foot should be on the ground (world position z=0)
                prog.AddConstraint(PositionConstraint(
                    plant, plant.world_frame(), [-np.inf,-np.inf,0], [np.inf,np.inf,0], 
                    foot_frame[i], [0,0,0], context[n]), q[:,n])
                if n>0 and in_stance[i, n-1]:
                    # feet should not move during stance.
                    prog.AddConstraint(partial(fixed_position_constraint, context_index=n-1, frame=foot_frame[i]),
                                       lb=np.zeros(3), ub=np.zeros(3), vars=np.concatenate((q[:,n-1], q[:,n])))
            else:
                min_clearance = 0.01
                prog.AddConstraint(PositionConstraint(plant, plant.world_frame(), [-np.inf,-np.inf,min_clearance], [np.inf,np.inf,np.inf],foot_frame[i],[0,0,0],context[n]), q[:,n])



    def support_ploygon_constraint(vars, context_index):
        q, com = np.split(vars, [nq])
        if isinstance(vars[0], AutoDiffXd):
            q = autoDiffToValueMatrix(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            p_WF = plant.CalcPointsPositions(context[context_index], foot_frame[0], [0,0,0], plant.world_frame())
            Jq_WF = plant.CalcJacobianTranslationalVelocity(
                context[context_index], JacobianWrtVariable.kQDot,
                foot_frame[0], [0, 0, 0], plant.world_frame(), plant.world_frame())
            ad_p_WF = initializeAutoDiffGivenGradientMatrix(p_WF, np.hstack((Jq_WF, np.zeros((3, 12)))))
            distance = ad_p_WF.reshape(3) - com
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            p_WF = plant.CalcPointsPositions(context[context_index], foot_frame[0], [0,0,0], plant.world_frame())
            distance = p_WF.reshape(3) - com
        return distance

    for n in range(N):
        prog.AddConstraint(partial(support_ploygon_constraint, context_index=n),
                            lb=np.array([0.02,-np.inf,-np.inf]), ub=np.array([0.04,np.inf,np.inf]), vars=np.concatenate((q[:,n], com[:,n])))


    # prog.AddBoundingBoxConstraint(0.01, 0.01, q_view.body_x[[-1]])





    ##################################################################################################################
    # TODO: Set solver parameters (mostly to make the worst case solve times less bad)
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 1e6 )
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 1000 )
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
    print(f"optimal cost {result.get_optimal_cost()}")
    # print(prog.EvalBinding(veloCon[16], result.GetSolution()))

    # prog.SetInitialGuessForAllVariables(result.GetSolution())
    # result = Solve(prog)
    # print(f"optimal cost {result.get_optimal_cost()}")




    contact_force_sol = [result.GetSolution(contact_force[i]) for i in range(num_contacts)]
    myv_sol = result.GetSolution(v)
    myq_sol = result.GetSolution(q)
    h_sol = result.GetSolution(h)
    H_sol = result.GetSolution(H)
    Hdot_sol = result.GetSolution(Hdot)
    com_sol = result.GetSolution(com)
    comdot_sol = result.GetSolution(comdot)
    comddot_sol = result.GetSolution(comddot)


    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)

    print('h_sol lift: \n',sum(h_sol[0:LiftKont]))
    print('h_sol fly: \n',sum(h_sol[LiftKont:TouchKont]))
    print('h_sol touch: \n',sum(h_sol[TouchKont:-1]))
    print('h_sol : \n',h_sol)

    print('\ncontact_force_sol x: \n',contact_force_sol[0][0,:])
    print('contact_force_sol x: \n',contact_force_sol[1][0,:])
    # print('contact_force_sol y: \n',contact_force_sol[0][1,:])
    print('contact_force_sol z: \n',contact_force_sol[0][2,:])
    print('contact_force_sol z: \n',contact_force_sol[1][2,:])


    # print('\nH_sol z: \n',H_sol[0])
    # print('H_sol z: \n',H_sol[1])
    # print('H_sol z: \n',H_sol[2])



    print('')
    for i in range(N):
        plant.SetPositions(plant_context, myq_sol[:,i])
        p_WF = plant.CalcPointsPositions(plant_context, foot_frame[0], [0,0,0], plant.world_frame())
        print(p_WF[2], end=' ')
    print('')
    for i in range(N):
        plant.SetPositions(plant_context, myq_sol[:,i])
        p_WF = plant.CalcPointsPositions(plant_context, foot_frame[1], [0,0,0], plant.world_frame())
        print(p_WF[2], end=' ')
    print('')


    t_sol = np.zeros(N)
    t_sol[0] = 0
    for n in range(N-1):
        t_sol[n+1] = t_sol[n] + h_sol[n]





    print('\nt_sol z: \n',t_sol)
    print('\ncom_sol z: \n',com_sol[2])
    print('comdot_sol x: \n',comdot_sol[0])
    # print('comdot_sol y: \n',comdot_sol[1])
    print('comdot_sol z: \n',comdot_sol[2])
    print('comddot_sol z: \n',comddot_sol[2])











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
    index_h = 0
    for t in (np.arange(t0, T, visualizer.draw_period)):
        if t>sum(h_sol[:index_h]) :
          index_h = index_h+1

        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)
        qt = PositionView(q_sol.value(ts))
        # plant.SetPositions(plant_context, myq_sol[:,index_h])
        plant.SetPositions(plant_context, qt)
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()


    import matplotlib.pyplot as plt
    comddot_sol = np.hstack((comddot_sol, np.array([[0],[0],[0]])))
    plt.plot(t_sol , 0.1*comddot_sol[2,:], '*-')
    plt.plot(t_sol , comdot_sol[2,:], '*-')
    plt.plot(t_sol , 10*com_sol[2,:], '*-')
    plt.legend(['comddot', 'comdot', 'com'])
    plt.show()




gait_optimization('jump')

import time
time.sleep(1e7)

