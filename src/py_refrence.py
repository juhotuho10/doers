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



lhs(5, samples=2, criterion="lhsmu", random_state=10)

fullfact

