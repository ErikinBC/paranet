"""
Check that we can solve the solution path for all datasets in SurvSet
"""

# External modules
import numpy as np
import pandas as pd
from SurvSet.data import SurvLoader

# Internal modules
from paranet.models import parametric
from paranet.utils import dist_valid


loader = SurvLoader()
# List of available datasets and meta-info
print(loader.df_ds.head())