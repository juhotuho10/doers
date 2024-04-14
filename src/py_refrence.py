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


result = fracfact("1 2 3 13").astype(int)

for i in result:
    print(f"{list(i)}")