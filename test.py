import numpy as np
from numpy.core.shape_base import hstack
from numpy.lib import index_tricks


L_FOOT_HEEL_L_IDX          = 0
L_FOOT_HEEL_R_IDX          = 1
L_FOOT_TOE_L_IDX           = 2
L_FOOT_TOE_R_IDX           = 3
R_FOOT_HEEL_L_IDX          = 4
R_FOOT_HEEL_R_IDX          = 5
R_FOOT_TOE_L_IDX           = 6
R_FOOT_TOE_R_IDX           = 7


in_stance = np.zeros((8, 31))


CONTACT_DELAY = 2
# standing
t = 0
in_stance[ R_FOOT_HEEL_L_IDX, t:] = 1
in_stance[ R_FOOT_HEEL_R_IDX, t:] = 1
in_stance[ R_FOOT_TOE_L_IDX, t:] = 1
in_stance[ R_FOOT_TOE_R_IDX, t:] = 1

in_stance[ L_FOOT_HEEL_L_IDX, t:] = 1
in_stance[ L_FOOT_HEEL_R_IDX, t:] = 1
in_stance[ L_FOOT_TOE_L_IDX, t:] = 1
in_stance[ L_FOOT_TOE_R_IDX, t:] = 1

# foot heel takeoff
# Heel should come off at the same time due to the pivot at the toe
t = 8
in_stance[ R_FOOT_HEEL_R_IDX, t:] = 0
in_stance[ R_FOOT_HEEL_L_IDX, t:] = 0

in_stance[ L_FOOT_HEEL_L_IDX, t:] = 0
in_stance[ L_FOOT_HEEL_R_IDX, t:] = 0


# foot toe takeoff
t = 12
in_stance[ R_FOOT_TOE_L_IDX, t:] = 0
in_stance[ L_FOOT_TOE_L_IDX, t:] = 0
t = t + CONTACT_DELAY
in_stance[ R_FOOT_TOE_R_IDX, t:] = 0
in_stance[ L_FOOT_TOE_R_IDX, t:] = 0


# foot toe strike
t = 16
in_stance[ R_FOOT_TOE_L_IDX, t:] = 1
in_stance[ L_FOOT_TOE_L_IDX, t:] = 1
t = t + CONTACT_DELAY
in_stance[ R_FOOT_TOE_R_IDX, t:] = 1
in_stance[ L_FOOT_TOE_R_IDX, t:] = 1

# foot heel strike
t = 22
in_stance[ R_FOOT_HEEL_R_IDX, t:] = 1
in_stance[ R_FOOT_HEEL_L_IDX, t:] = 1

in_stance[ L_FOOT_HEEL_L_IDX, t:] = 1
in_stance[ L_FOOT_HEEL_R_IDX, t:] = 1


# print(in_stance)



lb=[0]*5
print('lb 5:',lb)

lb=[0]*10
print('lb 10:',lb)






# q_cost.r_leg_aky= 0
# q_cost.l_leg_aky= 0
# q_cost.r_leg_kny= 0
# q_cost.l_leg_kny= 0
# q_cost.r_leg_hpy= 0
# q_cost.l_leg_hpy= 0

# q_cost.pelvis_x = 0
# q_cost.pelvis_y = 0
# q_cost.pelvis_qx = 0
# q_cost.pelvis_qy = 0
# q_cost.pelvis_qz = 0
# q_cost.pelvis_qw = 0
# q_cost.back_bkx = 5
# q_cost.back_bky = 0
# q_cost.back_bkz = 5


