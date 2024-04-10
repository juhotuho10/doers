import pyDOE2
from pyDOE2.doe_repeat_center import repeat_center
from pyDOE2.doe_factorial import fullfact, ff2n
from pyDOE2.doe_star import star
from pyDOE2 import *
import numpy as np
from scipy.linalg import toeplitz, hankel
from math import frexp
import itertools

from pyDOE2.doe_lhs import lhs



arr = lhs(n=10, samples=10, iterations = 100, criterion="centermaximin", random_state=10)

print(arr)

