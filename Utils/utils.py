# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import colorsys
import cv2
import sys
import h5py
import numpy as np
import os
from keras import Model
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

# -----------------------------
#   GLOBAL VARIABLES
# -----------------------------
min_logo_size = (10, 10)


# -----------------------------
#   FUNCTIONS
# -----------------------------
def detect_object(yolo, img_path, save_img, save_img_path="./", postfix=""):
    """
        Calls YOLO logo detector on the input image and (optionally) save resulting image
        :param yolo: keras-yolo3 initialized YOLO instance
        :param img_path: path to the image file
        :param save_img: bool to save annotated image file
        :param save_img_path: path to directory where to save resulting image
        :param postfix: string to add to filenames
        :return:
            -prediction: list of bounding boxes in format (xmin,ymin,xmax,ymax,class_id,confidence)
            -image: unaltered input image as (H,W,C) array
    """
    try:
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.array(image)
    except:
        print("File Open Error! Please try again...")
        return None, None
    prediction, new_image = yolo.detect_image(image)
    img_out = postfix.join(os.path.splitext(os.path.basename(img_path)))
    if save_img:
        new_image.save(os.path.join(save_img_path, img_out))
    return prediction, image_array


def parse_input():
    """
        Asks user input for the input images: pass path to individual images / directory
        :return:
    """
    outputs = []
    while True:
        inputs = input("Enter path (q to quit): ").strip()
        if inputs in ["q", "quit"]:
            break
        if not os.path.isdir(inputs):
            print("Error: File Not Found!")
        elif os.path.isdir(inputs):
            out = [os.path.abspath(os.path.join(inputs, f)) for f in os.listdir(inputs) if f.endswith((".jpg", ".png"))]
            break
        elif inputs.endswith((".jpg", ".png")):
            outputs.append(os.path.abspath(inputs))
        print(outputs)
    return outputs


def load_extractor_model(model_name="InceptionV3", flavor=1):
    """
        Loads variant of InceptionV3 or VGG16 model specified
        :param model_name: string, either InceptionV3 or VGG16
        :param flavor: int specifying the model variant and the input shape.
        (For InceptionV3, the map is {0: default, 1: 200*200, truncate last Inception block,
        2: 200*200, truncate last 2 blocks, 3: 200*200, truncate last 3 blocks, 4: 200*200}
        For VGG16, it only changes the input size, {0: 224 (default), 1: 128, 2: 64})
        :return:
    """
    start = timer()
    if model_name == "InceptionV3":
        from keras.applications.inception_v3 import InceptionV3
        from keras.applications.inception_v3 import preprocess_input
        model = InceptionV3(weights="imagenet", include_top=False)
        trunc_layer = [-1, 279, 248, 228, -1]
        i_layer = flavor
        model_out = Model(inputs=model.inputs, outputs=model.layers[trunc_layer[i_layer]].output)
        input_shape = (299, 299, 3) if flavor == 0 else (200, 200, 3)
    elif model_name == "VGG16":
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
        model_out = VGG16(weights="imagenet", include_top=False)
        input_length = [224, 128, 64][flavor]
        input_shape = (input_length, input_length, 3)
    end = timer()
    print("Loaded {} feature extractor in {:.2f}sec".format(model_name, end - start))
    return model_out, preprocess_input, input_shape


def chunks(l, n, preprocessing_function=None):
    """
        Yields successive n-sized chunks from l.
        General purpose function modified for Keras: made infinite loop, add preprocessing
        and returns np.array instead of list
        :param l: iterable
        :param n: number of items to take for each chunk
        :param preprocessing_function: function that processes image (3D array)
        :return: generator with n-sized np.array preprocessed chunks of the input
    """
    func = (lambda x: x) if (preprocessing_function is None) else preprocessing_function
    # In predict_generator, steps argument sets how many times looped through "while True"
    while True:
        for i in range(0, len(l), n):
            yield np.array([func(el) for el in l[i:i+n]])


def load_features(filename):
    """
        Load pre-saved HDF5 features for all logos in the LogosInTheWild database
        :param filename:
        :return:
    """
    start = timer()
    # Get database features
    with h5py.File(filename, "r") as hf:
        brand_map = list(hf.get("brand_map"))
        input_shape = list(hf.get("input_shape"))
        features = hf.get("features")
        features = np.array(features)
    end = timer()
    print("Loaded {} features from {} in {:.2f}sec".format(features.shape, filename, end - start))
    return features, brand_map, input_shape


def save_features(filename, features, brand_map, input_shape):
    """
        Save features to compressed HDF5 file for later use
        :param filename:
        :param features:
        :param brand_map:
        :param input_shape:
        :return:
    """
    print("Saving {} features into {}...".format(features.shape, filename), end="")
    # Reduce file size by saving as float16
    features = features.astype(np.float16)
    start = timer()
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("features", data=features, compression="lzf")
        hf.create_dataset("brand_map", data=brand_map)
        hf.create_dataset("input_shape", data=input_shape)
    end = timer()
    print("Done in {:2f} seconds".format(end-start))
    return None


def features_from_image(img_array, model, preprocess, batch_size=100):
    """
        Extracts features from image array given a decapitated keras model.
        Use a generator to avoid running out of memory for large inputs.
        :param img_array: (N, H, W, C) list/array of input images
        :param model: keras model, outputs
        :param preprocess: preprocessing flag
        :param batch_size: batch size
        :return:
            - features: (N, F) array of 1D features
    """
    if len(img_array) == 0:
        return np.array([])
    steps = len(img_array) // batch_size+1
    img_gen = chunks(img_array, batch_size, preprocessing_function=preprocess)
    features = model.predict_generator(img_gen, steps=steps)
    # If the generator has looped past end of array, cut it down
    features = features[: len(img_array)]
    # Reshape features: flatten last three dimensions to one
    features = features.reshape(features.shape[0], np.prod(features.shape[1:]))
    return features


def pad_image(img, shape, mode="constant_mean"):
    """
        Resizes and pad image to given size.
        :param img: (H, W, C) input numpy array
        :param shape: (H', W') destination size
        :param mode: filling mode for new padded pixels.
        (Default = 'constant_mean' returns grayscale padding with pixel intensity equal to mean of the array.
        Other options include np.pad() options, such as 'edge', 'mean' (by row/column)...
        :return: new_im: (H', W', C) padded numpy array
    """
    if mode == "constant_mean":
        mode_args = {"mode": "constant", "constant_values": np.mean(img)}
    else:
        mode_args = {"mode": mode}
    input_height, input_width = img.shape[:2]
    height, width = shape[:2]
    # First rescale image so that largest dimension matches target
    scale = min(width / input_width, height / input_height)
    new_width, new_height = int(input_width * scale), int(input_height * scale)
    img = cv2.resize(img, (new_width, new_height))
    # Center-pad rest of image: compute padding and split in two
    xpad, ypad = shape[1] - new_width, shape[0] - new_height
    xpad = (xpad // 2, xpad // 2 + xpad % 2)
    ypad = (ypad // 2, ypad // 2 + ypad % 2)
    new_img = np.pad(img, pad_width=(ypad, xpad, (0, 0)), **mode_args)
    return new_img


def bbox_colors(n):
    """
        Defines n distinct bounding box colors
        :param n: number of colors
        :return: colors: (n, 3) np.array with RGB integer values in [0-255] range
    """
    hsv_tuples = [(x / n, 1.0, 1.0) for x in range(n)]
    colors = 255 * np.array([colorsys.hsv_to_rgb(*x) for x in hsv_tuples])
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors.astype(int)


def contents_of_bbox(img, bbox_list, expand=1.0):
    """
        Extract portions of image inside bounding boxes list.
        :param img: (H, W, C) image array
        :param bbox_list: list of bounding box specifications with first 4 elements specifying box corners in
        (xmin, ymin, xmax, ymax) format.
        :param expand:
        :return:
            - candidates: list of 3D image arrays
            - i_candidates_too_small: list of indices of small candidates dropped
    """
    candidates = []
    i_candidates_too_small = []
    for i, (xmin, ymin, xmax, ymax, *_) in enumerate(bbox_list):
        # For very low confidence sometimes logos found outside of the image
        if ymin > img.shape[0] or xmin > img.shape[1]:
            continue
        xmin, ymin = int(xmin//expand), int(ymin//expand)
        xmax, ymax = int(np.round(xmax//expand)), int(np.round(ymax//expand))
        # Do not even consider tiny logos
        if xmax - xmin > min_logo_size[1] and ymax - ymin > min_logo_size[0]:
            candidates.append(img[ymin:ymax, xmin:xmax])
        else:
            i_candidates_too_small.append(i)
    return candidates, i_candidates_too_small


def draw_annotated_box(image, box_list_list, label_list, color_list):
    """
        Draw box and overhead label on the image
        :param image: PIL image object
        :param box_list_list: list of lists of bounding boxes, one for each label in which each box is
        in (xmin, ymin, xmax, ymax, [, score]) format (where score is an optional float)
        :param label_list: list of string to go above box
        :param color_list: list of RGB tuples
        :return: image: annotated PIL image object
    """
    font_path = os.path.join(os.path.dirname(__file__), "keras_yolo3/font/FiraMono-Medium.otf")
    font = ImageFont.truetype(font=font_path, size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
    thickness = (image.size[0] + image.size[1]) // 300
    draw = ImageDraw.Draw(image)
    for box_list, label, color in zip(box_list_list, label_list, color_list):
        if not isinstance(color, tuple):
            color = tuple(color)
        for box in box_list:
            # Deal with empty predictions
            if len(box) < 4:
                continue
            # If score is also passed, append to label
            the_label = "{}".format(label)
            if len(box) > 4:
                the_label += " {:.2f}".format(box[-1])
            label_size = draw.textsize(the_label, font)
            xmin, ymin, xmax, ymax = box[:4]
            ymin = max(0, np.floor(ymin + 0.5).astype("int32"))
            xmin = max(0, np.floor(xmin + 0.5).astype("int32"))
            ymax = min(image.size[1], np.floor(ymax + 0.5).astype("int32"))
            xmax = min(image.size[0], np.floor(xmax + 0.5).astype("int32"))
            if ymin - label_size[1] >= 0:
                text_origin = np.array([xmin, ymin - label_size[1]])
            else:
                text_origin = np.array([xmin, ymax])

            for i in range(thickness):
                draw.rectangle([xmin + i, ymin + i, xmax - i, ymax - i], outline=color)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)], fill=color
            )
            draw.text(text_origin, the_label, fill=(0, 0, 0), font=font)
    del draw
    return image







