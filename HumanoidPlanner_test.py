from HumanoidPlanner import HumanoidPlanner
from HumanoidPlanner import create_q_interpolation, create_r_interpolation, apply_angular_velocity_to_quaternion
import numpy as np
from Atlas import Atlas, load_atlas, set_atlas_initial_pose
import unittest

from pydrake.all import MultibodyPlant

mbp_time_step = 1.0e-3

class TestHumanoidPlannerStandalone(unittest.TestCase):
    def test_create_q_interpolation(self):
        plant = MultibodyPlant(mbp_time_step)
        load_atlas(plant, add_ground=False)
        context = plant.CreateDefaultContext()
        pass

    def test_create_r_interpolation(self):
        pass

    def test_apply_angular_velocity_to_quaternion(self):
        pass

class TestHumanoidPlanner(unittest.TestCase):
    def setUp(self):
        self.plant = MultibodyPlant(mbp_time_step)
        load_atlas(self.plant, add_ground=False)
        self.context = self.plant.CreateDefaultContext()
        upright_context = self.plant.CreateDefaultContext()
        set_atlas_initial_pose(self.plant, upright_context)
        q_nom = self.plant.GetPositions(upright_context)
        self.planner = HumanoidPlanner(self.plant, Atlas.CONTACTS_PER_FRAME, q_nom)

    def test_getPlantAndContext(self):
        pass

    def test_toTauj(self):
        pass

    def test_get_contact_position(self):
        pass

    def test_get_contact_positions_z(self):
        pass

    def test_calc_h(self):
        pass

    def test_calc_r(self):
        pass

    def test_eq7c(self):
        pass

    def test_eq7d(self):
        pass

    def test_eq7h(self):
        pass

    def test_eq7i(self):
        pass

    def test_eq8a_lhs(self):
        pass

    def test_eq8b_lhs(self):
        pass

    def test_eq8c_2(self):
        pass

    def test_eq9a_lhs(self):
        pass

    def test_eq9b_lhs(self):
        pass

if __name__ == "__main__":
    unittest.main()
