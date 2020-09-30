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
from Utils.convert_format import convert_vott_csv_to_yolo


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
sys.path.append(os.path.join(get_parent_dir(1), "Utils"))
Data_Folder = os.path.join(get_parent_dir(1), "Data")
VoTT_Folder = os.path.join(Data_Folder, "Source_Images", "Training_Images", "vott-csv-export")
VoTT_csv = os.path.join(VoTT_Folder, "Annotations-export.csv")
YOLO_filename = os.path.join(VoTT_Folder, "data_train.txt")
model_folder = os.path.join(Data_Folder, "Model_Weights")
classes_filenames = os.path.join(model_folder, "data_classes.txt")


# -----------------------------
#   MAIN
# -----------------------------
if __name__ == '__main__':
    # Construct the argument parser, parse the arguments and suppress any inherited default values
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
        Command line options
    """
    parser.add_argument("--VoTT_Folder", type=str, default=VoTT_Folder,
                        help="Absolute path to the exported files from the image tagging step with VoTT. Default is "
                             + VoTT_Folder)
    parser.add_argument("--VoTT_csv", type=str, default=VoTT_csv,
                        help="Absolute path to the *.csv file exported from VoTT. Default is " + VoTT_csv)
    parser.add_argument("--YOLO_filename", type=str, default=YOLO_filename,
                        help="Absolute path to the file where the annotations in YOLO format should be saved. "
                             "Default is " + YOLO_filename)
    FLAGS = parser.parse_args()

    # Prepare the dataset for YOLO
    multi_df = pd.read_csv(FLAGS.VoTT_csv)
    labels = multi_df["label"].unique()
    label_dict = dict(zip(labels, range(len(labels))))
    multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
    train_path = FLAGS.VoTT_Folder
    convert_vott_csv_to_yolo(multi_df, label_dict, path=train_path, target_name=FLAGS.YOLO_filename)

    # Make classes file
    file = open(classes_filenames, "w")

    # Sort dictionary by values
    sorted_label_dict = sorted(label_dict.items(), key=lambda x: x[1])
    for elem in sorted_label_dict:
        file.write(elem[0] + "\n")
    file.close()
