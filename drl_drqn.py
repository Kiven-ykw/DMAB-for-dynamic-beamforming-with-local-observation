from cellular_network import CellularNetwork as CN
from DRQN import drqn
import json
import random
import numpy as np
from config import Config
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

c = Config()
random.seed(c.random_seed)
np.random.seed(c.random_seed)
cn = CN()
ee = []
r_last = 0
#cn.draw_topology()
ee_m = []
R = []
for _ in range(c.total_slots):
    print(_)
    s = cn.observe()
    actions = cn.choose_actions(s)
    cn.update(ir_change=False, actions=actions)
    ee.append(cn.get_ave_ee())
    ee_m.append(cn.get_all_ees())
    cn.update(ir_change=True)
    r = cn.give_rewards(is_cooperation=True)  # the temp reward at current time slot
    r_new = r - r_last                         # the reward at current time slot
    r_last = r                                 # the temp reward at last time slot
    s_ = cn.observe()
    R.append(np.sum(r_new)/19)
    cn.save_transitions(s, actions, r_new, s_)

    if _ > 256:
        cn.train_dqns()
#cn.save_models('DRQN_common_PL=10_dft16')

filename = 'ee/DRQN_commonR_PL=5_dft44_TS=50000.json'
with open(filename, 'w') as f:
    json.dump(ee, f)



