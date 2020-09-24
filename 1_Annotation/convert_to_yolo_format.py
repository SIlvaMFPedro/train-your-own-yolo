# --------------------------------
#   USAGE
# --------------------------------
# python convert_to_yolo_format.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from PIL import Image
from os import path, makedirs
import os
import re
import pandas as pd
import sys
import argparse


# -----------------------------
#   FUNCTIONS
# -----------------------------
def get_parent_dir(n=1):
    """
        Get parent directory path function
        :param n: Directory index
        :return: Returns the n-th parent directory of the current working directory
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

# Check utils directory

