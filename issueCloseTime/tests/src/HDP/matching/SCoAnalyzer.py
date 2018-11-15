import os
import sys

import pandas as pd

root = os.getcwd().split('src')[0] + 'src'
if root not in sys.path:
    sys.path.append(root)
from pdb import set_trace
