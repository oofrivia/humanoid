#!/usr/bin/python3

'''
This implements the paper
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''

from load_atlas import load_atlas, set_atlas_initial_pose
from load_atlas import getSortedJointLimits, getActuatorIndex, getActuatorIndices, getJointValues
from load_atlas import JOINT_LIMITS, lfoot_full_contact_points, rfoot_full_contact_points, FLOATING_BASE_DOF, FLOATING_BASE_QUAT_DOF, NUM_ACTUATED_DOF, TOTAL_DOF, M
from pydrake.all import eq, le, ge, PiecewisePolynomial, PiecewiseTrajectory
from pydrake.geometry import ConnectDrakeVisualizer, SceneGraph
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.systems.framework import BasicVector, LeafSystem
from utility import calcPoseError_Expression
from balance import HumanoidController
import numpy as np
import pdb

mbp_time_step = 1.0e-3
N_d = 4 # friction cone approximated as a i-pyramid
N_f = 3 # contact force dimension

num_contact_points = lfoot_full_contact_points.shape[0]+rfoot_full_contact_points.shape[0]
mu = 1.0 # Coefficient of friction, same as in load_atlas.py
n = np.array([
    [0],
    [0],
    [1.0]])
d = np.array([
    [1.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 0.0, 0.0]])
# Equivalent to v in balance.py
friction_cone_components = np.zeros((N_d, num_contact_points, N_f))
for i in range(N_d):
    for j in range(num_contact_points):
        friction_cone_components[i,j] = (n+mu*d)[:,i]

class Interpolator(LeafSystem):
    def __init__(self, r_traj, rd_traj, rdd_traj, dt_traj):
        LeafSystem.__init__(self)
        self.input_t_idx = self.DeclareVectorInputPort("t", BasicVector(1)).get_index()
        self.output_r_idx = self.DeclareVectorOutputPort("r", BasicVector(3), self.get_r).get_index()
        self.output_rd_idx = self.DeclareVectorOutputPort("rd", BasicVector(3), self.get_rd).get_index()
        self.output_rdd_idx = self.DeclareVectorOutputPort("rdd", BasicVector(3), self.get_rdd).get_index()
        self.trajectory_polynomial = PiecewisePolynomial()
        t = 0.0
        for i in range(len(r_traj)-1):
            # CubicHermite assumes samples are column vectors
            r = np.reshape(r_traj[i], (3,1))
            rd = np.reshape(rd_traj[i], (3,1))
            rdd = np.reshape(rdd_traj[i], (3,1))
            dt = np.reshape(dt_traj[i+1], (3,1))
            r_next = np.reshape(r_traj[i+1], (3,1))
            rd_next = np.reshape(rd_traj[i+1], (3,1))
            rdd_next = np.reshape(rdd_traj[i+1], (3,1))
            self.trajectory_polynomial.ConcatenateInTime(
                    CubicHermite(
                        breaks=[t, t+dt],
                        samples=np.hstack([r, r_next],
                        sample_dot=[rd, rd_next])))
            t += dt
        self.trajectory = PiecewiseTrajectory(self.trajectory_polynomial)

    def get_r(self, context, output):
        t = self.EvalVectorInput(context, self.input_t_idx).get_value()
        output.SetFromVector(self.trajectory.get_position(t))

    def get_rd(self, context, output):
        t = self.EvalVectorInput(context, self.input_t_idx).get_value()
        output.SetFromVector(self.trajectory.get_velocity(t))

    def get_rdd(self, context, output):
        t = self.EvalVectorInput(context, self.input_t_idx).get_value()
        output.SetFromVector(self.trajectory.get_acceleration(t))

def calcTrajectory(q_init, q_final):
    plant = MultibodyPlant(mbp_time_step)
    load_atlas(plant)
    plant_autodiff = plant.ToAutoDiffXd()
    upright_context = plant.CreateDefaultContext()
    q_nom = plant.GetPositions(upright_context)

    def get_contact_positions(q):
        context = plant_autodiff.CreateDefaultContext()
        plant_autodiff.SetPositions(context, q[k])
        plant_autodiff.SetVelocities(context, v[k])
        lfoot_full_contact_positions = plant_autodiff.CalcPointsPositions(
                context, plant_autodiff.GetFrameByName("l_foot"),
                lfoot_full_contact_points, plant_autodiff.world_frame())
        rfoot_full_contact_positions = plant_autodiff.CalcPointsPositions(
                context, plant_autodiff.GetFrameByName("r_foot"),
                rfoot_full_contact_points, plant_autodiff.world_frame())
        return np.concatenate([lfoot_full_contact_points, rfoot_full_contact_points], axis=1)

    N = 50
    T = 10.0 # 10 seconds

    sorted_joint_position_lower_limits = np.array([entry[1].lower for entry in getSortedJointLimits(plant)])
    sorted_joint_position_upper_limits = np.array([entry[1].upper for entry in getSortedJointLimits(plant)])
    sorted_joint_velocity_limits = np.array([entry[1].velocity for entry in getSortedJointLimits(plant)])

    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(rows=N, cols=plant.num_positions(), name="q")
    v = prog.NewContinuousVariables(rows=N, cols=plant.num_velocities(), name="v")
    dt = prog.NewContinuousVariables(N, name="dt")
    r = prog.NewContinuousVariables(rows=N, cols=3, name="r")
    rd = prog.NewContinuousVariables(rows=N, cols=3, name="rd")
    rdd = prog.NewContinuousVariables(rows=N, cols=3, name="rdd")
    contact_dim = 3*num_contact_points
    # The cols are ordered as
    # [contact1_x, contact1_y, contact1_z, contact2_x, contact2_y, contact2_z, ...]
    c = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="c")
    F = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="F")
    tau = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="tau")
    h = prog.NewContinuousVariables(rows=N, cols=3, name="h")
    hd = prog.NewContinuousVariables(rows=N, cols=3, name="hd")

    ''' Additional variables not explicitly stated '''
    # Friction cone scale
    beta = prog.NewContinuousVariables(rows=N, cols=num_contact_points*N_d, name="beta")

    for k in range(N):
        ''' Eq(7a) '''
        g = np.array([0, 0, -9.81])
        Fj = np.reshape(F[k], (num_contact_points, 3))
        prog.AddLinearConstraint(eq(M*rdd[k], np.sum(Fj, axis=0) + M*g))
        ''' Eq(7b) '''
        cj = np.reshape(c[k], (num_contact_points, 3))
        tauj = np.reshape(tau[k], (num_contact_points, 3))
        prog.AddConstraint(eq(hd[k], np.sum(np.cross(cj - r[k], Fj) + tauj, axis=0)))
        ''' Eq(7c) '''
        # https://stackoverflow.com/questions/63454077/how-to-obtain-centroidal-momentum-matrix/63456202#63456202
        # TODO

        ''' Eq(7h) '''
        def eq7h(q_r):
            q, r = np.split(plant.num_positions())
            context = plant_autodiff.CreateDefaultContext()
            plant_autodiff.SetPositions(context, q)
            plant_autodiff.SetVelocities(context, v)
            return plant_autodiff.CalcCenterOfMassPosition(context) - r
        # COM position has dimension 3
        prog.AddConstraint(eq7h, lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], r[k]]))
        ''' Eq(7i) '''
        def eq7i(q_ck):
            q, ck = np.split(q_ck, plant.num_positions())
            cj = np.reshape(ck, (num_contact_points, 3))
            contact_positions = get_contact_positions(q)
            return (contact_positions - cj).flatten()
        # np.concatenate cannot work q, cj since they have different dimensions
        prog.AddConstraint(eq7i, lb=np.zeros(cj.shape).flatten(), ub=np.zeros(cj.shape).flatten(), vars=np.concatenate([q[k], c[k]]))
        ''' Eq(7j) '''
        ''' We don't constrain the contact point positions for now... '''

        ''' Eq(7k) '''
        ''' Constrain admissible posture '''
        prog.AddLinearConstraint(le(q[k, FLOATING_BASE_QUAT_DOF:], sorted_joint_position_upper_limits))
        prog.AddLinearConstraint(ge(q[k, FLOATING_BASE_QUAT_DOF:], sorted_joint_position_lower_limits))
        ''' Constrain velocities '''
        prog.AddLinearConstraint(le(v[k, FLOATING_BASE_DOF:], sorted_joint_velocity_limits))
        prog.AddLinearConstraint(ge(v[k, FLOATING_BASE_DOF:], -sorted_joint_velocity_limits))
        ''' Constrain forces within friction cone '''
        beta_k = np.reshape(beta[k], (num_contact_points, N_d))
        for i in range(num_contact_points):
            beta_v = beta_k[i].dot(friction_cone_components[:,i,:])
            prog.AddLinearConstraint(eq(Fj[i], beta_v))
        ''' Constrain torques - assume no torque allowed for now '''
        for i in range(num_contact_points):
            prog.AddLinearConstraint(eq(tauj[i], np.array([0.0, 0.0, 0.0])))

        ''' Assume flat ground for now... '''
        def get_contact_positions_z(q):
            return get_contact_positions(q)[2,:]
        ''' Eq(8a) '''
        def eq8a_lhs(q_F):
            q, F = np.split(q_F, plant.num_positions())
            Fj = np.reshape(F, (num_contact_points, 3))
            return Fj[:,2].dot(get_contact_positions_z(q))
        prog.AddConstraint(eq8a_lhs, lb=[0.0], ub=[0.0], vars=np.concatenate([q[k], F[k]]))
        ''' Eq(8b) '''
        def eq8b_lhs(q_tau):
            q, tau = np.split(q_tau, plant.num_positions())
            tauj = np.reshape(tau, (num_contact_points, 3))
            return tauj.dot(tauj).dot(get_contact_positions_z(q))
        prog.AddConstraint(eq8b_lhs, lb=[0.0], ub=[0.0], vars=np.concatenate([q[k], tau[k]]))
        ''' Eq(8c) '''
        prog.AddLinearConstraint(ge(Fj[:,2], 0.0))
        prog.AddConstraint(get_contact_positions_z, lb=[0.0], ub=[float('inf')], vars=q[k])

    for k in range(1, N):
        ''' Eq(7d) '''
        '''
        Constrain rotation
        Taken from Practical Methods for Optimal Control and Estimation by ...
        Section 6.8 Reorientation of an Asymmetric Rigid Body
        '''
        q1 = q[k,0]
        q2 = q[k,1]
        q3 = q[k,2]
        q4 = q[k,3]
        w1 = v[k,0]
        w2 = v[k,1]
        w3 = v[k,2]
        # Not sure why reshape is necessary
        prog.AddConstraint(eq(q[k,0] - q[k-1,0], 0.5*(w1*q4 - w2*q3 + w3*q2)).reshape((1,)))
        prog.AddConstraint(eq(q[k,1] - q[k-1,1], 0.5*(w1*q3 + w2*q4 - w3*q1)).reshape((1,)))
        prog.AddConstraint(eq(q[k,2] - q[k-1,2], 0.5*(-w1*q2 + w2*q1 + w3*q4)).reshape((1,)))
        prog.AddConstraint(eq(q[k,3] - q[k-1,3], 0.5*(-w1*q1 - w2*q2 - w3*q3)).reshape((1,)))
        ''' Constrain other positions '''
        prog.AddConstraint(eq(q[k, 4:] - q[k-1, 4:], v[k, 3:]*dt[k]))
        ''' Eq(7e) '''
        prog.AddConstraint(eq(h[k] - h[k-1], hd[k]*dt[k]))
        ''' Eq(7f) '''
        prog.AddConstraint(eq(r[k] - r[k-1], (rd[k] + rd[k-1])/2*dt[k]))
        ''' Eq(7g) '''
        prog.AddConstraint(eq(rd[k] - rd[k-1], rdd[k]*dt[k]))

        Fj = np.reshape(F[k], (num_contact_points, 3))
        cj = np.reshape(c[k], (num_contact_points, 3))
        cj_prev = np.reshape(c[k-1], (num_contact_points, 3))
        for i in range(num_contact_points):
            ''' Assume flat ground for now... '''
            ''' Eq(9a) '''
            prog.AddConstraint(Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([1.0, 0.0, 0.0])) == 0.0)
            ''' Eq(9b) '''
            prog.AddConstraint(Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([0.0, 1.0, 0.0])) == 0.0)
    ''' Eq(10) '''
    Q_q = 0.1 * np.identity(plant.num_velocities())
    Q_v = 0.2 * np.identity(plant.num_velocities())
    for k in range(N):
        # TODO: Convert this to polynomial expression
        q_err = calcPoseError_Expression(q[k], q_nom)
        prog.AddCost(dt[k]*(
                q_err.dot(Q_q).dot(q_err)
                + v[k].dot(Q_v).dot(v[k])
                + rdd[k].dot(rdd[k])))

    ''' Additional constraints not explicitly stated '''
    ''' Constrain initial pose '''
    prog.AddLinearConstraint(eq(q[0], q_init))
    ''' Constrain initial velocity '''
    prog.AddLinearConstraint(eq(v[0], 0.0))
    ''' Constrain final pose '''
    prog.AddLinearConstraint(eq(q[-1], q_final))
    ''' Constrain final velocity '''
    prog.AddLinearConstraint(eq(v[0], 0.0))
    ''' Constrain time taken '''
    prog.AddLinearConstraint(le(np.sum(dt), T))

    ''' Solve '''
    start_solve_time = time.time()
    result = Solve(prog)
    print(f"Solve time: {time.time() - start_solve_time}s")
    if not result.is_success():
        print(f"FAILED")
        pdb.set_trace()
        exit(-1)
    print(f"Cost: {result.get_optimal_cost()}")
    r_sol = result.GetSolution(r)
    rd_sol = result.GetSolution(rd)
    rdd_sol = result.GetSolution(rdd)
    kt_sol = result.GetSolution(kt)

    return r_sol, rd_sol, rdd_sol, kt_sol

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()

    q_init = plant.GetPositions(plant_context)
    q_final = q_init
    q_final[4] = 4 # x position of pelvis

    r_traj, rd_traj, rdd_traj, kt_traj = calcTrajectory(q_init, q_final)

    controller = builder.AddSystem(HumanoidController(is_wbc=True))
    controller.set_name("HumanoidController")

    ''' Connect atlas plant to controller '''
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), plant.get_actuation_input_port())

    interpolator = builder.AddSystem(Interpolator(r_traj, rd_traj, rdd_traj, kt_traj))
    interpolator.set_name("Interpolator")
    ''' Connect interpolator to controller '''
    builder.Connect(interpolator.GetOutputPort("r"), controller.GetInputPort("r"))
    builder.Connect(interpolator.GetOutputPort("rd"), controller.GetInputPort("rd"))
    builder.Connect(interpolator.GetOutputPort("rdd"), controller.GetInputPort("rdd"))

    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    set_atlas_initial_pose(plant, plant_context)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.1)
    simulator.AdvanceTo(5.0)

if __name__ == "__main__":
    main()
