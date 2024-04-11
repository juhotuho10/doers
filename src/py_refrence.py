import pyDOE2
from pyDOE2.doe_repeat_center import repeat_center
from pyDOE2.doe_factorial import fullfact, ff2n
from pyDOE2.doe_star import star
from pyDOE2 import *
import numpy as np
from scipy.linalg import toeplitz, hankel
from math import frexp
import itertools
from scipy import spatial
from pyDOE2.doe_lhs import lhs

input = [[-1.,-2.,-3.], [1.,2.,3.], [10., -15., 32.], [-100., 340., 32.], [-342. , 421., -523.]]
Hcandidate = np.array(input).astype(np.float32)
d = np.corrcoef(Hcandidate.T).astype(np.float32)
for i in d:
    print(f"{list(i)},")

