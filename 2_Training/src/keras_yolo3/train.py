# =============================================
#  RETRAIN THE YOLO MODEL FOR YOUR OWN DATASET
# =============================================

# -----------------------------
#   USAGE
# -----------------------------
# python train.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


# -----------------------------
#   FUNCTIONS
# -----------------------------
def get_classes(classes_path):
    """
        Load classes function
        :param classes_path: file path to classes annotation file
        :return: list with the loaded classes
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """
        Load anchors funtion
        :param anchors_path: file path to anchors annotation file
        :return: list with the loaded anchors
    """
    with open(anchors_path) as f:
        anchors_names = f.readline()
    anchors = [float(x) for x in anchors_names.split(",")]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path="model_data/yolo_weights.h5"):
    """
        Create training model function
        :param input_shape: input shape
        :param anchors: annotation anchors
        :param num_classes: number of annotation classes
        :param load_pretrained: bool pretrained flag
        :param freeze_body: number of layers to freeze
        :param weights_path: path to weights file
        :return:
    """
    # Get a new session
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print("Create YOLOv3 model with {} anchors and {} classes.".format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Load weights {}.".format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
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
                      weights_path="model_data/tiny_yolo_weights.h5"):
    """
        Create training model for Tiny YOLOv3 function
        :param input_shape: input shape
        :param anchors: annotation anchors
        :param num_classes: number of annotation classes
        :param load_pretrained: bool pretrained flag
        :param freeze_body: number of layers to freeze
        :param weights_path: path to weights file
        :return:
    """
    # Get a new session
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]
    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print("Create Tiny YOLOv3 model with {} anchors and {} classes.".format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Load weights {}.".format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
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
        Data generator function for fit_generator
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
    """
        Data generator wrapper function for fit_generator
        :param annotation_lines:
        :param batch_size:
        :param input_shape:
        :param anchors:
        :param num_classes:
        :return:
    """
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def main():
    annotation_path = "data_train.txt"
    log_dir = "logs/003/"
    classes_path = "data_classes.txt"
    anchors_path = "model_data/yolo_anchors.txt"
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)            # multiple of 32, hw
    epoch1, epoch2 = 40, 40
    tiny_version = len(anchors) == 6    # Default setting
    if tiny_version:
        model = create_tiny_model(input_shape=input_shape, anchors=anchors, num_classes=num_classes,
                                  freeze_body=2, weights_path="model_data/yolo-tiny.h5")
    else:
        model = create_model(input_shape=input_shape, anchors=anchors, num_classes=num_classes,
                             freeze_body=2, weights_path="model_data/yolo.h5")
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir="checkpoint.h5", monitor="val_loss", save_weights_only=True,
                                 save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    # Train with frozen layers first, to get a stable loss, then adjust num epochs to your dataset.
    # (This step is enough to obtain a pretty good model)
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={"yolo_loss": lambda y_true, y_pred: y_pred})
        batch_size = 32
        print("Train on {} samples, val on {} samples with batch size {}.".format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape,
                                                                   anchors, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=epoch1,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + "trained_weights_stage_1.h5")
    # Unfreeze and continue the training process to fine-tune the model.
    # (Train longer if the result is not good)
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # Recompile to apply the change
        model.compile(optimizer=Adam(lr=1e-4), loss={"yolo_loss": lambda y_true, y_pred: y_pred})
        print("Unfreeze all of the layers.")
        batch_size = 16      # Note that more GPU memory is required after unfreezing the body
        print("Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape,
                                                                   anchors, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=epoch1 + epoch2,
                            initial_epoch=epoch1,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + "trained_weights_final.h5")

    # TODO Further training if needed


# -----------------------------
#   MAIN
# -----------------------------
if __name__ == '__main__':
    main()
