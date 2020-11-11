#!/usr/bin/python3

'''
This implements the paper
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''

from Atlas import load_atlas, set_atlas_initial_pose
from Atlas import getSortedJointLimits, getActuatorIndex, getActuatorIndices, getJointValues
from Atlas import Atlas
from pydrake.all import Quaternion
from pydrake.all import Multiplexer
from pydrake.all import PiecewisePolynomial, PiecewiseTrajectory, PiecewiseQuaternionSlerp, TrajectorySource
from pydrake.all import ConnectDrakeVisualizer, ConnectContactResultsToDrakeVisualizer, Simulator
from pydrake.all import DiagramBuilder, MultibodyPlant, AddMultibodyPlantSceneGraph, BasicVector, LeafSystem
from pydrake.all import MathematicalProgram, Solve, eq, le, ge, SolverOptions
# from pydrake.all import IpoptSolver
from pydrake.all import SnoptSolver
from pydrake.all import Quaternion_, AutoDiffXd
from HumanoidController import HumanoidController
import numpy as np
import time
import pdb
import pickle
from collections.abc import Iterable 

mbp_time_step = 1.0e-3
N_f = 3 # contact force dimension
mu = 1.0 # Coefficient of friction
epsilon = 1e-9
quaternion_epsilon = 1e-5
PLAYBACK_ONLY = False
ENABLE_COMPLEMENTARITY_CONSTRAINTS = True
MAX_GROUND_PENETRATION = 0.0
MAX_JOINT_ACCELERATION = 20.0
g = np.array([0, 0, -Atlas.g])
'''
Slack for the complementary constraints
Same value used in drake/multibody/optimization/static_equilibrium_problem.cc
'''
# slack = 1e-3
slack = 1e-2

def create_q_interpolation(plant, context, q_traj, v_traj, dt_traj):
    t_traj = np.cumsum(dt_traj)
    quaternions = [Quaternion(q[0:4] / np.linalg.norm(q[0:4])) for q in q_traj]
    quaternion_poly = PiecewiseQuaternionSlerp(t_traj, quaternions)
    position_poly = PiecewisePolynomial.FirstOrderHold(t_traj, q_traj[:, 4:].T)
    return quaternion_poly, position_poly
    # qd_poly = q_poly.derivative()
    # qdd_poly = qd_poly.derivative()
    # return q_poly, v_poly, vd_poly

def create_r_interpolation(r_traj, rd_traj, rdd_traj, dt_traj):
    r_poly = PiecewisePolynomial()
    t = 0.0
    for i in range(len(dt_traj)-1):
        # CubicHermite assumes samples are column vectors
        r = np.array([r_traj[i]]).T
        rd = np.array([rd_traj[i]]).T
        rdd = np.array([rdd_traj[i]]).T
        dt = dt_traj[i+1]
        r_next = np.array([r_traj[i+1]]).T
        rd_next = np.array([rd_traj[i+1]]).T
        rdd_next = np.array([rdd_traj[i+1]]).T
        r_poly.ConcatenateInTime(
                PiecewisePolynomial.CubicHermite(
                    breaks=[t, t+dt],
                    samples=np.hstack([r, r_next]),
                    samples_dot=np.hstack([rd, rd_next])))
        t += dt
    rd_poly = r_poly.derivative()
    rdd_poly = rd_poly.derivative()
    return r_poly, rd_poly, rdd_poly

def apply_angular_velocity_to_quaternion(q, w, t):
    # This currently returns a runtime warning of division by zero
    # https://github.com/RobotLocomotion/drake/issues/10451
    norm_w = np.linalg.norm(w)
    if norm_w <= epsilon:
        return q
    norm_q = np.linalg.norm(q)
    if abs(norm_q - 1.0) > quaternion_epsilon:
        print(f"WARNING: Quaternion {q} with norm {norm_q} not normalized!")
    a = w / norm_w
    if q.dtype == AutoDiffXd:
        delta_q = Quaternion_[AutoDiffXd](np.hstack([np.cos(norm_w * t/2.0), a*np.sin(norm_w * t/2.0)]).reshape((4,1)))
        return Quaternion_[AutoDiffXd](q/norm_q).multiply(delta_q).wxyz()
    else:
        delta_q = Quaternion(np.hstack([np.cos(norm_w * t/2.0), a*np.sin(norm_w * t/2.0)]).reshape((4,1)))
        return Quaternion(q/norm_q).multiply(delta_q).wxyz()

def get_index_of_variable(variables, variable_name):
    return [str(element) for element in variables].index(variable_name)

def create_constraint_input_array(constraint, name_value_map):
    processed_name_value_map = name_value_map.copy()
    ret = np.zeros(len(constraint.variables()))
    # Fill array with NaN values first
    ret.fill(np.nan)
    for name, value in name_value_map.items():
        if isinstance(value, Iterable):
            # Expand vectors into individual entries
            it = np.nditer(value, flags=['multi_index', 'refs_ok'])
            while not it.finished:
                if len(it.multi_index) == 1:
                    # Convert x(1,) to x(1)
                    element_name = name + f"({it.multi_index[0]})"
                else:
                    element_name = name + str(it.multi_index).replace(" ","")
                processed_name_value_map[element_name] = it.value
                it.iternext()
            del processed_name_value_map[name]
        else:
            # Rename scalars from 'x' to 'x(0)'
            element_name = name + "(0)"
            processed_name_value_map[element_name] = value
            del processed_name_value_map[name]

    for name, value in processed_name_value_map.items():
        try:
            ret[get_index_of_variable(constraint.variables(), name)] = value
        except ValueError:
            pass
    # Make sure all values are filled
    assert(not np.isnan(ret).any())
    return ret

class HumanoidPlanner:
    def __init__(self, plant_float, contacts_per_frame, q_nom):
        self.plant_float = plant_float
        self.context_float = plant_float.CreateDefaultContext()
        self.plant_autodiff = self.plant_float.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()
        self.q_nom = q_nom

        self.sorted_joint_position_lower_limits = np.array([entry[1].lower for entry in getSortedJointLimits(self.plant_float)])
        self.sorted_joint_position_upper_limits = np.array([entry[1].upper for entry in getSortedJointLimits(self.plant_float)])
        self.sorted_joint_velocity_limits = np.array([entry[1].velocity for entry in getSortedJointLimits(self.plant_float)])

        self.contacts_per_frame = contacts_per_frame
        self.num_contacts = sum([contact_points.shape[1] for contact_points in contacts_per_frame.values()])
        self.contact_dim = 3*self.num_contacts

        self.N_d = 4 # friction cone approximated as a i-pyramid
        n = np.array([
            [0],
            [0],
            [1.0]])
        d = np.array([
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]])
        # Equivalent to v in HumanoidController.py
        self.friction_cone_components = np.zeros((self.N_d, self.num_contacts, N_f))
        for i in range(self.N_d):
            for j in range(self.num_contacts):
                self.friction_cone_components[i,j] = (n+mu*d)[:,i]

    def getPlantAndContext(self, q, v):
        assert(q.dtype == v.dtype)
        if q.dtype == np.object:
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff, self.context_autodiff
        else:
            self.plant_float.SetPositions(self.context_float, q)
            self.plant_float.SetVelocities(self.context_float, v)
            return self.plant_float, self.context_float

    '''
    Creates an np.array of shape [num_contacts, 3] where first 2 rows are zeros
    since we only care about tau in the z direction
    '''
    def toTauj(self, tau_k):
        return np.hstack([np.zeros((self.num_contacts, 2)), np.reshape(tau_k, (self.num_contacts, 1))])

    '''
    Returns contact positions in the shape [3, num_contacts]
    '''
    def get_contact_positions(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        contact_positions_per_frame = []
        for frame, contacts in self.contacts_per_frame.items():
            contact_positions = plant.CalcPointsPositions(
                context, plant.GetFrameByName(frame),
                contacts, plant.world_frame())
            contact_positions_per_frame.append(contact_positions)
        return np.concatenate(contact_positions_per_frame, axis=1)

    ''' Assume flat ground for now... '''
    def get_contact_positions_z(self, q, v):
        return self.get_contact_positions(q, v)[2,:]

    # https://stackoverflow.com/questions/63454077/how-to-obtain-centroidal-momentum-matrix/63456202#63456202
    def calc_h(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        return plant.CalcSpatialMomentumInWorldAboutPoint(context, plant.CalcCenterOfMassPosition(context)).rotational()

    def calc_r(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        return plant.CalcCenterOfMassPosition(context)

    def eq7c(self, q_v_h):
        q, v, h = np.split(q_v_h, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        return self.calc_h(q, v) - h

    def eq7d(self, q_qprev_v_dt):
        q, qprev, v, dt = np.split(q_qprev_v_dt, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_positions() + self.plant_float.num_velocities()])
        plant, context = self.getPlantAndContext(q, v)
        qd = plant.MapVelocityToQDot(context, v*dt[0])
        # return q - qprev - qd
        '''
        As advised in
        https://stackoverflow.com/a/63510131/3177701
        '''
        ret_quat = q[0:4] - apply_angular_velocity_to_quaternion(qprev[0:4], v[0:3], dt[0])
        ret_linear = (q - qprev - qd)[4:]
        ret = np.hstack([ret_quat, ret_linear])
        return ret

    def eq7h(self, q_v_r):
        q, v, r = np.split(q_v_r, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        return  self.calc_r(q, v) - r

    def eq7i(self, q_v_ck):
        q, v, ck = np.split(q_v_ck, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        cj = np.reshape(ck, (self.num_contacts, 3))
        # print(f"q = {q}\nv={v}\nck={ck}")
        contact_positions = self.get_contact_positions(q, v).T
        return (contact_positions - cj).flatten()

    def eq8a_lhs(self, q_v_F):
        q, v, F = np.split(q_v_F, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        Fj = np.reshape(F, (self.num_contacts, 3))
        return [Fj[:,2].dot(self.get_contact_positions_z(q, v))] # Constraint functions must output vectors

    def eq8b_lhs(self, q_v_tau):
        q, v, tau = np.split(q_v_tau, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        tauj = self.toTauj(tau)
        return (tauj**2).T.dot(self.get_contact_positions_z(q, v)) # Outputs per axis sum of torques of all contact points

    def eq8c_2(self, q_v):
        q, v = np.split(q_v, [self.plant_float.num_positions()])
        return self.get_contact_positions_z(q, v)

    ''' Assume flat ground for now... '''
    def eq9a_lhs(self, F_c_cprev, i):
        F, c, c_prev = np.split(F_c_cprev, [
            self.contact_dim,
            self.contact_dim + self.contact_dim])
        Fj = np.reshape(F, (self.num_contacts, 3))
        cj = np.reshape(c, (self.num_contacts, 3))
        cj_prev = np.reshape(c_prev, (self.num_contacts, 3))
        return [Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([1.0, 0.0, 0.0]))]

    def eq9b_lhs(self, F_c_cprev, i):
        F, c, c_prev = np.split(F_c_cprev, [
            self.contact_dim,
            self.contact_dim + self.contact_dim])
        Fj = np.reshape(F, (self.num_contacts, 3))
        cj = np.reshape(c, (self.num_contacts, 3))
        cj_prev = np.reshape(c_prev, (self.num_contacts, 3))
        return [Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([0.0, 1.0, 0.0]))]

    def pose_error_cost(self, q_v_dt):
        q, v, dt = np.split(q_v_dt, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        plant, context = self.getPlantAndContext(q, v)
        Q_q = 1.0 * np.identity(plant.num_velocities())
        q_err = plant.MapQDotToVelocity(context, q-self.q_nom)
        return (dt*(q_err.dot(Q_q).dot(q_err)))[0] # AddCost requires cost function to return scalar, not array

    def add_eq7a_constraints(self):
        F = self.F
        rdd = self.rdd
        self.eq7a_constraints = []
        for k in range(self.N):
            Fj = np.reshape(F[k], (self.num_contacts, 3))
            constraint = self.prog.AddLinearConstraint(
                    eq(Atlas.M*rdd[k], np.sum(Fj, axis=0) + Atlas.M*g))
            constraint.evaluator().set_description(f"Eq(7a)[{k}]")
            self.eq7a_constraints.append(constraint)

    def check_eq7a_constraints(self, F, rdd):
        for k in range(self.N):
            constraint = self.eq7a_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "F": F,
                "rdd": rdd
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7b_constraints(self):
        F = self.F
        c = self.c
        tau = self.tau
        hd = self.hd
        r = self.r
        self.eq7b_constraints = []
        for k in range(self.N):
            Fj = np.reshape(F[k], (self.num_contacts, 3))
            cj = np.reshape(c[k], (self.num_contacts, 3))
            tauj = self.toTauj(self.tau[k])
            constraint = self.prog.AddConstraint(
                    eq(hd[k], np.sum(np.cross(cj - r[k], Fj) + tauj, axis=0)))
            constraint.evaluator().set_description(f"Eq(7b)[{k}]")
            self.eq7b_constraints.append(constraint)

    def check_eq7b_constraints(self, F, c, tau, hd, r):
        for k in range(self.N):
            constraint = self.eq7b_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "F": F,
                "c": c,
                "tau": tau,
                "hd": hd,
                "r": r
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7c_constraints(self):
        q = self.q
        v = self.v
        h = self.h
        self.eq7c_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddConstraint(self.eq7c,
                    lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], h[k]]))
            constraint.evaluator().set_description(f"Eq(7c)[{k}]")
            self.eq7c_constraints.append(constraint)

    def check_eq7c_constraints(self, q, v, h)
        for k in range(self.N):
            constraint = self.eq7c_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "q", q,
                "v", v,
                "h", h
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7d_constraints(self):
        q = self.q
        v = self.v
        dt = self.dt
        self.eq7d_constraints = []
        for k in range(1, self.N):
            # dt[k] must be converted to an array
            constraint = self.prog.AddConstraint(self.eq7d,
                    lb=[0.0]*self.plant_float.num_positions(),
                    ub=[0.0]*self.plant_float.num_positions(),
                    vars=np.concatenate([q[k], q[k-1], v[k], [dt[k]]]))
            constraint.evaluator().set_description(f"Eq(7d)[{k}]")

            # Deprecated
            # '''
            # Constrain rotation
            # Taken from Practical Methods for Optimal Control and Estimation by ...
            # Section 6.8 Reorientation of an Asymmetric Rigid Body
            # '''
            # q1 = q[k,0]
            # q2 = q[k,1]
            # q3 = q[k,2]
            # q4 = q[k,3]
            # w1 = v[k,0]*dt[k]
            # w2 = v[k,1]*dt[k]
            # w3 = v[k,2]*dt[k]
            # # Not sure why reshape is necessary
            # self.prog.AddConstraint(eq(q[k,0] - q[k-1,0], 0.5*(w1*q4 - w2*q3 + w3*q2)).reshape((1,)))
            # self.prog.AddConstraint(eq(q[k,1] - q[k-1,1], 0.5*(w1*q3 + w2*q4 - w3*q1)).reshape((1,)))
            # self.prog.AddConstraint(eq(q[k,2] - q[k-1,2], 0.5*(-w1*q2 + w2*q1 + w3*q4)).reshape((1,)))
            # self.prog.AddConstraint(eq(q[k,3] - q[k-1,3], 0.5*(-w1*q1 - w2*q2 - w3*q3)).reshape((1,)))
            # ''' Constrain other positions '''
            # self.prog.AddConstraint(eq(q[k, 4:] - q[k-1, 4:], v[k, 3:]*dt[k]))
            self.eq7d_constraints.append(constraint)

    def check_eq7d_constraints(self, q, v, dt):
        for k in range(1, self.N):
            constraint = self.eq7d_constraints[k-1]
            input_array = create_constraint_input_array(constraint, {
                "q": q,
                "v": v,
                "dt": dt
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7e_constraints(self):
        h = self.h
        hd = self.hd
        dt = self.dt
        self.eq7e_constraints = []
        for k in range(1, self.N):
            constraint = self.prog.AddConstraint(eq(h[k] - h[k-1], hd[k]*dt[k]))
            constraint.evaluator().set_description(f"Eq(7e)[{k}]")
            self.eq7e_constraints.append(constraint)

    def check_eq7e_constraints(self, h, hd, dt):
        for k in range(1, self.N):
            constraint = self.eq7e_constraints[k-1]
            input_array = create_constraint_input_array(constraint, {
                "h": h,
                "hd": hd,
                "dt": dt
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7f_constraints(self):
        r = self.r
        rd = self.rd
        dt = self.dt
        self.eq7f_constraints = []
        for k in range(1, self.N):
            constraint = self.prog.AddConstraint(
                    eq(r[k] - r[k-1], (rd[k] + rd[k-1])/2*dt[k]))
            constraint.evaluator().set_description(f"Eq(7f)[{k}]")
            self.eq7f_constraints.append(constraint)

    def check_eq7f_constraints(self, r, rd, dt):
        for k in range(1, self.N):
            constraint = self.eq7f_constraints[k-1]
            input_array = create_constraint_input_array(constraint, {
                "r": r,
                "rd": rd,
                "dt": dt
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7g_constraints(self):
        rd = self.rd
        rdd = self.rdd
        dt = self.dt
        self.eq7g_constraints = []
        for k in range(1, self.N):
            constraint = self.prog.AddConstraint(eq(rd[k] - rd[k-1], rdd[k]*dt[k]))
            constraint.evaluator().set_description(f"Eq(7g)[{k}]")
            self.eq7g_constraints.append(constraint)

    def check_eq7g_constraints(self, rd, rdd, dt):
        for k in range(1, self.N):
            constraint = self.eq7g_constraints[k-1]
            input_array = create_constraint_input_array(constraint, {
                "rd": rd,
                "rdd": rdd,
                "dt": dt
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7h_constraints(self):
        q = self.q
        v = self.v
        r = self.r
        self.eq7h_constraints = []
        for k in range(self.N):
            # COM position has dimension 3
            constraint = self.prog.AddConstraint(self.eq7h,
                    lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], r[k]]))
            constraint.evaluator().set_description(f"Eq(7h)[{k}]")
            self.eq7h_constraints.append(constraint)

    def check_eq7h_constraints(self, q, v, r):
        for k in range(self.N):
            constraint = self.eq7h_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "q": q,
                "v": v,
                "r": r
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7i_constraints(self):
        q = self.q
        v = self.v
        c = self.c
        self.eq7i_constraints = []
        for k in range(self.N):
            # np.concatenate cannot work q, cj since they have different dimensions
            constraint = self.prog.AddConstraint(self.eq7i,
                    lb=np.zeros(c[k].shape).flatten(), ub=np.zeros(c[k].shape).flatten(),
                    vars=np.concatenate([q[k], v[k], c[k]]))
            constraint.evaluator().set_description(f"Eq(7i)[{k}]")
            self.eq7i_constraints.append(constraint)

    def check_eq7i_constraints(self, q, v, c):
        for k in range(self.N):
            constraint = self.eq7i_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "q": q,
                "v": v,
                "c", c
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7j_constraints(self):
        c = self.c
        self.eq7j_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddBoundingBoxConstraint(
                    [-10, -10, -MAX_GROUND_PENETRATION]*self.num_contacts,
                    [10, 10, 10]*self.num_contacts,
                    c[k])
            constraint.evaluator().set_description(f"Eq(7j)[{k}]")
            self.eq7j_constraints.append(constraint)

    def check_eq7j_constraints(self, c):
        for k in range(self.N):
            constraint = self.eq7j_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "c": c
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7k_admissable_posture_constraints(self):
        q = self.q
        self.eq7k_admissable_posture_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddBoundingBoxConstraint(
                    self.sorted_joint_position_lower_limits, 
                    self.sorted_joint_position_upper_limits, 
                    q[k, Atlas.FLOATING_BASE_QUAT_DOF:])
            constraint.evaluator().set_description(f"Eq(7k)[{k}] joint position")
            self.eq7k_admissable_posture_constraints.append(constraint)

    def check_eq7k_admissable_posture_constraints(self, q):
        for k in range(self.N):
            constraint = self.eq7k_admissable_posture_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "q": q
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7k_joint_velocity_constraints(self):
        v = self.v
        self.eq7k_joint_velocity_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddBoundingBoxConstraint(
                    -self.sorted_joint_velocity_limits,
                    self.sorted_joint_velocity_limits,
                    v[k, Atlas.FLOATING_BASE_DOF:])
            constraint.evaluator().set_description(f"Eq(7k)[{k}] joint velocity")
            self.eq7k_velocity_constraints.append(constraint)

    def check_eq7k_joint_velocity_constraints(self, v):
        for k in range(self.N):
            constraint = self.eq7k_joint_velocity_constraints[k]
            input_array = create_constraint_input_array(constraint, {
                "v": v
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq7k_friction_cone_constraints(self):
        F = self.F
        beta = self.beta
        self.eq7k_friction_cone_constraints = []
        for k in range(self.N):
            Fj = np.reshape(F[k], (self.num_contacts, 3))
            beta_k = np.reshape(beta[k], (self.num_contacts, self.N_d))
            friction_cone_constraints = []
            for i in range(self.num_contacts):
                beta_v = beta_k[i].dot(self.friction_cone_components[:,i,:])
                constraint = self.prog.AddLinearConstraint(eq(Fj[i], beta_v))
                constraint.evaluator().set_description(f"Eq(7k)[{k}] friction cone constraint[{i}]")
                friction_cone_constraints.append(constraint)
            self.eq7k_friction_cone_constraints.append(friction_cone_constraints)

    def check_eq7k_friction_cone_constraints(self, F, beta):
        for k in range(self.N):
            for i in range(self.num_contacts)
                constraint = self.eq7k_friction_cone_constraints[k][i]
                input_array = create_constraint_input_array(constraint, {
                    "F": F,
                    "beta": beta
                })
                if not constraint.evaluator().CheckSatisfied(input_array):
                    print(f"{constraint.evaluator().get_description()} violated")
                    return False
        return True

    def add_eq7k_beta_positive_constraints(self):
        beta = self.beta
        self.eq7k_beta_positive_constraints = []
        for k in range(self.N):
            beta_positive_constraints = []
            for b in beta[k]:
                constraint = self.prog.AddLinearConstraint(b >= 0.0)
                constraint.evaluator().set_description(f"Eq(7k)[{k}] beta >= 0 constraint")
                beta_positive_constraints.append(constraint)
            self.eq7k_beta_positive_constraints.append(beta_positive_constraints)

    def check_eq7k_beta_positive_constraints(self, beta):
        for k in range(self.N):
            for i in range(self.num_contacts * self.N_d):
                constraint = self.eq7k_beta_positive_constraints[k][i]
                input_array = create_constraint_input_array(constraint, {
                    "beta": beta
                })
                if not constraint.evaluator().CheckSatisfied(input_array):
                    print(f"{constraint.evaluator().get_description()} violated")
                    return False
        return True

    def add_eq7k_torque_constraints(self):
        tau = self.tau
        beta = self.beta
        self.eq7k_torque_constraints = []
        for k in range(self.N):
            ''' Constrain torques - assume torque linear to friction cone'''
            beta_k = np.reshape(beta[k], (self.num_contacts, self.N_d))
            friction_torque_coefficient = 0.1
            friction_torque_constraints = []
            for i in range(self.num_contacts):
                max_torque = friction_torque_coefficient * np.sum(beta_k[i])
                upper_constraint = self.prog.AddLinearConstraint(le(tau[k][i], np.array([max_torque])))
                upper_constraint.evaluator().set_description(f"Eq(7k)[{k}] friction torque upper limit")
                lower_constraint = self.prog.AddLinearConstraint(ge(tau[k][i], np.array([-max_torque])))
                lower_constraint.evaluator().set_description(f"Eq(7k)[{k}] friction torque lower limit")
                friction_torque_constraints.append([upper_constraint, lower_constraint])
            self.eq7k_torque_constraints.append(friction_torque_constraints)

    def check_eq7k_torque_constraints(self, tau, beta):
        for k in range(self.N):
            for i in range(self.num_contacts):
                upper_constraint = self.eq7k_torque_constraints[k][i][0]
                input_array = create_constraint_input_array(upper_constraint, {
                    "tau": tau,
                    "beta": beta
                })
                if not upper_constraint.evaluator().CheckSatisfied(input_array):
                    print(f"{upper_constraint.evaluator().get_description()} violated")
                    return False

                lower_constraint = self.eq7k_torque_constraints[k][i][1]
                input_array = create_constraint_input_array(lower_constraint, {
                    "tau": tau,
                    "beta": beta
                })
                if not lower_constraint.evaluator().CheckSatisfied(input_array):
                    print(f"{lower_constraint.evaluator().get_description()} violated")
                    return False
        return True

    def add_eq8a_constraints(self):
        q = self.q
        v = self.v
        F = self.F
        self.eq8a_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddConstraint(
                    self.eq8a_lhs,
                    lb=[-slack],
                    ub=[slack],
                    vars=np.concatenate([q[k], v[k], F[k]]))
            constraint.evaluator().set_description(f"Eq(8a)[{k}]")
            self.eq8a_constraints.append(constraint)

    def check_eq8a_constraints(self, q, v, F):
        for k in range(self.N):
            constraint = self.eq8a_constraints[k]
            input_arra = create_constraint_input_array(constraint, {
                "q": q,
                "v": v,
                "F": F
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq8b_constraints(self):
        q = self.q
        v = self.v
        tau = self.tau
        self.eq8b_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddConstraint(
                    self.eq8b_lhs,
                    lb=[-slack]*3,
                    ub=[slack]*3,
                    vars=np.concatenate([q[k], v[k], tau[k]]))
            constraint.evaluator().set_description(f"Eq(8b)[{k}]")
            self.eq8b_constraints.append(constraint)

    def check_eq8b_constraints(self, q, v, tau):
        for k in range(self.N):
            constraint = self.eq8b_constraints[k]
            input_arra = create_constraint_input_array(constraint, {
                "q": q,
                "v": v,
                "tau": tau
            })
            if not constraint.evaluator().CheckSatisfied(input_array):
                print(f"{constraint.evaluator().get_description()} violated")
                return False
        return True

    def add_eq8c_contact_force_constraints(self):
        F = self.F
        self.eq8c_contact_force_constraints = []
        for k in range(self.N):
            Fj = np.reshape(F[k], (self.num_contacts, 3))
            constraint = self.prog.AddLinearConstraint(ge(Fj[:,2], 0.0))
            constraint.evaluator().set_description(f"Eq(8c)[{k}] contact force greater than zero")
            self.eq8c_contact_force_constraints.append(constraint)

    def add_eq8c_contact_distance_constraint(self):
        q = self.q
        v = self.v
        self.eq8c_contact_distance_constraints = []
        for k in range(self.N):
            # TODO: Why can't this be converted to a linear constraint?
            constraint = self.prog.AddConstraint(self.eq8c_2,
                    lb=[-MAX_GROUND_PENETRATION]*self.num_contacts,
                    ub=[float('inf')]*self.num_contacts,
                    vars=np.concatenate([q[k], v[k]]))
            constraint.evaluator().set_description(f"Eq(8c)[{k}] z position greater than zero")
            self.eq8c_contact_distance_constraints.append(constraint)

    def add_eq9a_constraints(self):
        F = self.F
        c = self.c
        self.eq9a_constraints = []
        for k in range(1, self.N):
            contact_constraints = []
            for i in range(self.num_contacts):
                '''
                i=i is used to capture the outer scope i variable
                https://stackoverflow.com/a/2295372/3177701
                '''
                constraint = self.prog.AddConstraint(
                        lambda F_c_cprev, i=i : self.eq9a_lhs(F_c_cprev, i),
                        ub=[slack],
                        lb=[-slack],
                        vars=np.concatenate([F[k], c[k], c[k-1]]))
                constraint.evaluator().set_description("Eq(9a)[{k}][{i}]")
                contact_constraints.append(constraint)
            self.eq9a_constraints.append(contact_constraints)

    def add_eq9b_constraints(self):
        F = self.F
        c = self.c
        self.eq9b_constraints = []
        for k in range(1, self.N):
            contact_constraints = []
            for i in range(self.num_contacts):
                '''
                i=i is used to capture the outer scope i variable
                https://stackoverflow.com/a/2295372/3177701
                '''
                constraint = self.prog.AddConstraint(
                        lambda F_c_cprev, i=i : self.eq9b_lhs(F_c_cprev, i),
                        ub=[slack],
                        lb=[-slack],
                        vars=np.concatenate([F[k], c[k], c[k-1]]))
                constraint.evaluator().set_description("Eq(9b)[{k}][{i}]")
                contact_constraints.append(constraint)
            self.eq9b_constraints.append(contact_constraints)

    def check_all_constraints(self, q, v, dt, r, rd, rdd, c, F, tau, h, hd, beta):
        return (self.check_eq7a_constraints(F, rdd)
                and self.check_eq7b_constraints(F, c, tau, hd, r)
                and self.check_eq7c_constraints(q, v, h)
                and self.check_eq7d_constraints(q, v, dt)
                and self.check_eq7h_constraints(q, v, r))

    def create_program(self, q_init, q_final, num_knot_points, max_time, pelvis_only=False):
        self.N = num_knot_points
        self.T = max_time

        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(rows=self.N, cols=self.plant_float.num_positions(), name="q")
        self.v = self.prog.NewContinuousVariables(rows=self.N, cols=self.plant_float.num_velocities(), name="v")
        self.dt = self.prog.NewContinuousVariables(self.N, name="dt")
        self.r = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="r")
        self.rd = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="rd")
        self.rdd = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="rdd")
        # The cols are ordered as
        # [contact1_x, contact1_y, contact1_z, contact2_x, contact2_y, contact2_z, ...]
        self.c = self.prog.NewContinuousVariables(rows=self.N, cols=self.contact_dim, name="c")
        self.F = self.prog.NewContinuousVariables(rows=self.N, cols=self.contact_dim, name="F")
        self.tau = self.prog.NewContinuousVariables(rows=self.N, cols=self.num_contacts, name="tau") # We assume only z torque exists
        self.h = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="h")
        self.hd = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="hd")

        ''' Additional variables not explicitly stated '''
        # Friction cone scale
        self.beta = self.prog.NewContinuousVariables(rows=self.N, cols=self.num_contacts*self.N_d, name="beta")

        ''' Rename variables for easier typing... '''
        q = self.q
        v = self.v
        dt = self.dt
        r = self.r
        rd = self.rd
        rdd = self.rdd
        c = self.c
        F = self.F
        tau = self.tau
        h = self.h
        hd = self.hd
        beta = self.beta

        self.add_eq7a_constraints()
        self.add_eq7b_constraints()
        self.add_eq7c_constraints()
        self.add_eq7d_constraints()
        self.add_eq7e_constraints()
        self.add_eq7f_constraints()
        self.add_eq7g_constraints()
        self.add_eq7h_constraints()
        if ENABLE_COMPLEMENTARITY_CONSTRAINTS:
            self.add_eq8a_constraints()
            self.add_eq8b_constraints()
            self.add_eq8c_contact_force_constraints()
            self.add_eq8c_contact_distance_constraint()
            self.add_eq9a_constraints()
            self.add_eq9b_constraints()

        ''' Eq(10) '''
        for k in range(self.N):
            Q_v = 0.5 * np.identity(self.plant_float.num_velocities())
            self.prog.AddCost(self.pose_error_cost, vars=np.concatenate([q[k], v[k], [dt[k]]])) # np.concatenate requires items to have compatible shape
            self.prog.AddCost(dt[k]*(
                    + v[k].dot(Q_v).dot(v[k])
                    + rdd[k].dot(rdd[k])))

        ''' Additional constraints not explicitly stated '''
        ''' Constrain initial pose '''
        (self.prog.AddLinearConstraint(eq(q[0], q_init))
                .evaluator().set_description("initial pose"))
        ''' Constrain initial velocity '''
        (self.prog.AddLinearConstraint(eq(v[0], 0.0))
                .evaluator().set_description("initial velocity"))
        ''' Constrain final pose '''
        if pelvis_only:
            (self.prog.AddLinearConstraint(eq(q[-1, 0:7], q_final[0:7]))
                    .evaluator().set_description("final pose"))
        else:
            (self.prog.AddLinearConstraint(eq(q[-1], q_final))
                    .evaluator().set_description("final pose"))
        ''' Constrain final velocity '''
        (self.prog.AddLinearConstraint(eq(v[-1], 0.0))
                .evaluator().set_description("final velocity"))
        ''' Constrain final COM velocity '''
        (self.prog.AddLinearConstraint(eq(rd[-1], 0.0))
                .evaluator().set_description("final COM velocity"))
        ''' Constrain final COM acceleration '''
        (self.prog.AddLinearConstraint(eq(rdd[-1], 0.0))
                .evaluator().set_description("final COM acceleration"))
        ''' Constrain time taken '''
        (self.prog.AddLinearConstraint(np.sum(dt) <= self.T)
                .evaluator().set_description("max time"))
        ''' Constrain first time step '''
        # Note that the first time step is only used in the initial cost calculation
        # and not in the backwards Euler
        (self.prog.AddLinearConstraint(dt[0] == 0)
                .evaluator().set_description("first timestep"))
        ''' Constrain remaining time step '''
        # Values taken from
        # https://colab.research.google.com/github/RussTedrake/underactuated/blob/master/exercises/simple_legs/compass_gait_limit_cycle/compass_gait_limit_cycle.ipynb
        (self.prog.AddLinearConstraint(ge(dt[1:], [0.005]*(self.N-1)))
                .evaluator().set_description("min timestep"))
        (self.prog.AddLinearConstraint(le(dt, [0.05]*self.N))
                .evaluator().set_description("max timestep"))
        ''' Constrain max joint acceleration '''
        for k in range(1, self.N):
            (self.prog.AddLinearConstraint(ge((v[k] - v[k-1]), -MAX_JOINT_ACCELERATION*dt[k]))
                    .evaluator().set_description(f"min joint acceleration[{k}]"))
            (self.prog.AddLinearConstraint(le((v[k] - v[k-1]), MAX_JOINT_ACCELERATION*dt[k]))
                    .evaluator().set_description(f"max joint acceleration[{k}]"))
        ''' Constrain unit quaternion '''
        for k in range(self.N):
            (self.prog.AddConstraint(np.linalg.norm(q[k][0:4]) == 1.)
                    .evaluator().set_description(f"unit quaternion constraint[{k}]"))
        '''
        Constrain unbounded variables to improve IPOPT performance
        because IPOPT is an interior point method which works poorly for unbounded variables
        '''
        (self.prog.AddLinearConstraint(le(F.flatten(), np.ones(F.shape).flatten()*1e3))
                .evaluator().set_description("max F"))
        (self.prog.AddBoundingBoxConstraint(-1e3, 1e3, tau)
                .evaluator().set_description("bound tau"))
        (self.prog.AddLinearConstraint(le(beta.flatten(), np.ones(beta.shape).flatten()*1e3))
                .evaluator().set_description("max beta"))

    def solve(self):
        ''' Solve '''
        initial_guess = np.empty(self.prog.num_vars())
        dt_guess = [0.0] + [self.T/(self.N-1)] * (N-1)
        self.prog.SetDecisionVariableValueInVector(dt, dt_guess, initial_guess)
        # Guess q to avoid initializing with invalid quaternion
        quat_traj_guess = PiecewiseQuaternionSlerp()
        quat_traj_guess.Append(0, Quaternion(q_init[0:4]))
        quat_traj_guess.Append(self.T, Quaternion(q_final[0:4]))
        position_traj_guess = PiecewisePolynomial.FirstOrderHold([0.0, self.T], np.vstack([q_init[4:], q_final[4:]]).T)
        q_guess = np.array([np.hstack([
            Quaternion(quat_traj_guess.value(t)).wxyz(), position_traj_guess.value(t).flatten()])
            for t in np.linspace(0, self.T, self.N)])
        self.prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)

        v_traj_guess = position_traj_guess.MakeDerivative()
        w_traj_guess = quat_traj_guess.MakeDerivative()
        v_guess = np.array([
            np.hstack([w_traj_guess.value(t).flatten(), v_traj_guess.value(t).flatten()])
            for t in np.linspace(0, self.T, self.N)])
        self.prog.SetDecisionVariableValueInVector(v, v_guess, initial_guess)

        c_guess = np.array([
            self.get_contact_positions(q_guess[i], v_guess[i]).T.flatten() for i in range(self.N)])
        for i in range(self.N):
            assert((self.eq7i(np.concatenate([q_guess[i], v_guess[i], c_guess[i]])) == 0.0).all())
        self.prog.SetDecisionVariableValueInVector(c, c_guess, initial_guess)

        r_guess = np.array([
            self.calc_r(q_guess[i], v_guess[i]) for i in range(self.N)])
        self.prog.SetDecisionVariableValueInVector(r, r_guess, initial_guess)

        h_guess = np.array([
            self.calc_h(q_guess[i], v_guess[i]) for i in range(self.N)])
        self.prog.SetDecisionVariableValueInVector(h, h_guess, initial_guess)

        solver = SnoptSolver()
        options = SolverOptions()
        # options.SetOption(solver.solver_id(), "max_iter", 50000)
        # This doesn't seem to do anything...
        # options.SetOption(CommonSolverOption.kPrintToConsole, True)
        start_solve_time = time.time()
        print(f"Start solving...")
        result = solver.Solve(self.prog, initial_guess, options) # Currently takes around 30 mins
        print(f"Solve time: {time.time() - start_solve_time}s  Cost: {result.get_optimal_cost()} Success: {result.is_success()}")
        self.q_sol = result.GetSolution(q)
        self.v_sol = result.GetSolution(v)
        self.dt_sol = result.GetSolution(dt)
        self.r_sol = result.GetSolution(r)
        self.rd_sol = result.GetSolution(rd)
        self.rdd_sol = result.GetSolution(rdd)
        self.c_sol = result.GetSolution(c)
        self.F_sol = result.GetSolution(F)
        self.tau_sol = result.GetSolution(tau)
        self.h_sol = result.GetSolution(h)
        self.hd_sol = result.GetSolution(hd)
        self.beta_sol = result.GetSolution(beta)
        if not result.is_success():
            print(result.GetInfeasibleConstraintNames(self.prog))
            pdb.set_trace()

        return (self.r_sol, self.rd_sol, self.rdd_sol,
                self.q_sol, self.v_sol, self.dt_sol)

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()

    upright_context = plant.CreateDefaultContext()
    set_atlas_initial_pose(plant, upright_context)
    q_nom = plant.GetPositions(upright_context)
    q_init = q_nom.copy()
    q_init[6] = 0.94 # Avoid initializing with ground penetration
    q_final = q_init.copy()
    q_final[4] = 0.0 # x position of pelvis
    q_final[6] = 0.9 # z position of pelvis (to make sure final pose touches ground)

    num_knot_points = 40
    max_time = 1.0
    assert(max_time / num_knot_points > 0.005)
    assert(max_time / num_knot_points < 0.05)

    export_filename = f"sample(final_x_{q_final[4]})(num_knot_points_{num_knot_points})(max_time_{max_time})"

    planner = HumanoidPlanner(plant, Atlas.CONTACTS_PER_FRAME, q_nom)
    if not PLAYBACK_ONLY:
        print(f"Starting pos: {q_init}\nFinal pos: {q_final}")
        planner.create_program(q_init, q_final, num_knot_points, max_time, pelvis_only=True)
        r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj = planner.solve()

        with open(export_filename, 'wb') as f:
            pickle.dump([r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj], f)

    with open(export_filename, 'rb') as f:
        r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj = pickle.load(f)

    controller = builder.AddSystem(HumanoidController(plant, Atlas.CONTACTS_PER_FRAME, is_wbc=True))
    controller.set_name("HumanoidController")

    ''' Connect atlas plant to controller '''
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), plant.get_actuation_input_port())

    quaternion_poly, position_poly = create_q_interpolation(plant, plant_context, q_traj, v_traj, dt_traj)

    quaternion_source = builder.AddSystem(TrajectorySource(quaternion_poly))
    position_source = builder.AddSystem(TrajectorySource(position_poly))
    q_multiplexer = builder.AddSystem(Multiplexer([4, plant.num_positions() - 4]))
    builder.Connect(quaternion_source.get_output_port(), q_multiplexer.get_input_port(0))
    builder.Connect(position_source.get_output_port(), q_multiplexer.get_input_port(1))
    builder.Connect(q_multiplexer.get_output_port(0), controller.GetInputPort("q_des"))

    positiond_source = builder.AddSystem(TrajectorySource(position_poly.MakeDerivative()))
    quaterniond_source = builder.AddSystem(TrajectorySource(quaternion_poly.MakeDerivative()))
    v_multiplexer = builder.AddSystem(Multiplexer([3, plant.num_velocities() - 3]))
    builder.Connect(quaterniond_source.get_output_port(), v_multiplexer.get_input_port(0))
    builder.Connect(positiond_source.get_output_port(), v_multiplexer.get_input_port(1))
    builder.Connect(v_multiplexer.get_output_port(0), controller.GetInputPort("v_des"))

    positiondd_source = builder.AddSystem(TrajectorySource(position_poly.MakeDerivative().MakeDerivative()))
    quaterniondd_source = builder.AddSystem(TrajectorySource(quaternion_poly.MakeDerivative().MakeDerivative()))
    vd_multiplexer = builder.AddSystem(Multiplexer([3, plant.num_velocities() - 3]))
    builder.Connect(quaterniondd_source.get_output_port(), vd_multiplexer.get_input_port(0))
    builder.Connect(positiondd_source.get_output_port(), vd_multiplexer.get_input_port(1))
    builder.Connect(vd_multiplexer.get_output_port(0), controller.GetInputPort("vd_des"))

    r_poly, rd_poly, rdd_poly = create_r_interpolation(r_traj, rd_traj, rdd_traj, dt_traj)
    r_source = builder.AddSystem(TrajectorySource(r_poly))
    rd_source = builder.AddSystem(TrajectorySource(rd_poly))
    rdd_source = builder.AddSystem(TrajectorySource(rdd_poly))
    builder.Connect(r_source.get_output_port(0), controller.GetInputPort("r_des"))
    builder.Connect(rd_source.get_output_port(0), controller.GetInputPort("rd_des"))
    builder.Connect(rdd_source.get_output_port(0), controller.GetInputPort("rdd_des"))

    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()

    frame_idx = 0
    t = 0.0
    while True:
        print(f"Frame: {frame_idx}, t: {t}, dt: {dt_traj[frame_idx]}")
        diagram_context = diagram.CreateDefaultContext()
        plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
        plant.SetPositions(plant_context, q_traj[frame_idx])

        simulator = Simulator(diagram, diagram_context)
        simulator.set_target_realtime_rate(0.0)
        simulator.AdvanceTo(0)
        pdb.set_trace()

        frame_idx = (frame_idx + 1) % q_traj.shape[0]
        t = t + dt_traj[frame_idx]

if __name__ == "__main__":
    main()