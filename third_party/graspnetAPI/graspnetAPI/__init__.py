__author__ = 'mhgou'
__version__ = '1.2.11'

from .graspnet import GraspNet
try:
    from .graspnet_eval import GraspNetEval
except ImportError:
    GraspNetEval = None
from .grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup
