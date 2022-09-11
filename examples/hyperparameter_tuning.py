"""
Demonstrating how to do hyperparameter tuning to maximize out-of-sample concordance prediction
"""

# Load modules
import numpy as np
import pandas as pd
from SurvSet.data import SurvLoader
from paranet.models import parametric
from paranet.utils import dist_valid
from sksurv.metrics import concordance_index_censored as concordance
from sklearn.compose import make_column_selector
