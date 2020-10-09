# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os
import sys
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras_yolo3.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from keras_yolo3.yolo3.utils import get_random_data
from PIL import Image


# -----------------------------
#   FUNCTIONS
# -----------------------------
def get_classes(classes_path):
    """
        Load classes function
        :param classes_path: path to classes label file
        :return: loaded classes
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """
        Load the anchors from a file
        :param anchors_path: path to anchors file
        :return: loaded anchors
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path="keras_yolo3/model_data/yolo_weights.h5"):
    """
        Create training model function
        :param input_shape:
        :param anchors:
        :param num_classes:
        :param load_pretrained:
        :param freeze_body:
        :param weights_path:
        :return:
    """
    # Get a new session
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [
        Input(
            shape=(
                h // {0: 32, 1: 16, 2: 8}[l],
                w // {0: 32, 1: 16, 2: 8}[l],
                num_anchors // 3,
                num_classes + 5,
            )
        )
        for l in range(3)
    ]
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print("Create YOLOv3 model with {} anchors and {} classes.".format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Load weights {}".format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet body or freeze all but the 3 output layers
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print("Freeze the first {} layers of total {} layers.".format(num, len(model_body.layers)))
    model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss",
                        arguments={"anchors": anchors, "num_classes": num_classes, "ignore_thresh": 0.5}
                        )([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path="keras_yolo3/model_data/tiny_yolo_weights.h5"):
    """
        Create training model for Tiny YOLOV3 function
        :param input_shape:
        :param anchors:
        :param num_classes:
        :param load_pretrained:
        :param freeze_body:
        :param weights_path:
        :return:
    """
    # Get a new session
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [
        Input(
            shape=(
                h // {0: 32, 1: 16}[l],
                w // {0: 32, 1: 16}[l],
                num_anchors // 2,
                num_classes + 5,
            )
        )
        for l in range(2)
    ]
    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print("Create Tiny YOLOv3 model with {} anchors and {} classes.".format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Load weights {}.".format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but the 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print("Freeze the first {} layers of total {} layers.".format(num, len(model_body.layers)))
    model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss",
                        arguments={"anchors": anchors, "num_classes": num_classes, "ignore_thresh": 0.7}
                        )([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """
        Data generator for fit_generator function
        :param annotation_lines:
        :param batch_size:
        :param input_shape:
        :param anchors:
        :param num_classes:
        :return:
    """
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def changeToOtherMachine(filelist, repo="train-your-own-yolo", remote_machine=""):
    """
        Takes a list of file_names located in a repo and changes it to the local machines file names.
        File must be executed from withing the repository
        Example:
        '/home/ubuntu/train-your-won-yolo/Data/Street_View_Images/vulnerable/test.jpg'
        Get's converted to
        'C:/Users/pedro/train-your-own-yolo/Data/Street_View_Images/vulnerable/test.jpg'

    """
    filelist = [x.replace("\\", "/") for x in filelist]
    if repo[-1] == "/":
        repo = repo[:-1]
    if remote_machine:
        prefix = remote_machine.replace("\\", "/")
    else:
        prefix = ((os.getcwd().split(repo))[0]).replace("\\", "/")
    new_list = []
    for file in filelist:
        suffix = (file.split(repo))[1]
        if suffix[0] == "/":
            suffix = suffix[1:]
        new_list.append(os.path.join(prefix, repo + "/", suffix).replace("\\", "/"))
    print("8888888888888888888*********************************98888888888888888888888888888888888")
    print(new_list)
    return new_list


