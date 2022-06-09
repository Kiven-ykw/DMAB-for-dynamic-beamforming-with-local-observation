""" random scheme for the formulated DDBC problem"""

from cellular_network import CellularNetwork as CN
import json
import random
import numpy as np
from config import Config
import scipy.io as sio
import os
os.environ['MKL_NUM_THREADS'] = '1'


c = Config()
np.random.seed(c.random_seed)
random.seed(c.random_seed)
cn = CN()
ee = []
cn.draw_topology()
for _ in range(c.total_slots):
    print(_)
    actions = cn.random_choose_actions()
    cn.update(ir_change=False, actions=actions)
    ee.append(cn.get_ave_ee())
    cn.update(ir_change=True)

# save data
filename = 'ee/random_performance_U=8.json'
with open(filename, 'w') as f:
    json.dump(ee, f)



