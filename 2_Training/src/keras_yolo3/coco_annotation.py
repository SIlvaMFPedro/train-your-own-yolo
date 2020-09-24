# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from collections import defaultdict
import json

name_box_id = defaultdict(list)
id_name = dict()
f = open("mscoco2017/annotations/instances_train2017.json", encoding="utf-8")
data = json.load(f)
annotations = data["annotations"]
for ant in annotations:
    id = ant["image_id"]
    name = "mscoco2017/train2017/%012d.jpg" % id
    cat = ant["category_id"]
    if 1 <= cat <= 11:
        cat = cat - 1
    elif 13 <= cat <= 25:
        cat = cat - 2
    elif 27 <= cat <= 28:
        cat = cat - 3
    elif 31 <= cat <= 44:
        cat = cat - 5
    elif 46 <= cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif 72 <= cat <= 82:
        cat = cat - 10
    elif 84 <= cat <= 90:
        cat = cat - 11
    name_box_id[name].append([ant["bbox"], cat])
f = open("train.txt", "w")
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])
        box_info = " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, int(info[1]))
        f.write(box_info)
    f.write("\n")
f.close()
