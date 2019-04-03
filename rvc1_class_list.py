"""
This code is copied from the original ACRV Robotic Vision Challenge code.
Link to original code: https://github.com/jskinn/rvchallenge-evaluation/blob/master/class_list.py
Link to challenge websites:
    - CVPR 2019: https://competitions.codalab.org/competitions/20940
    - Continuous: https://competitions.codalab.org/competitions/21727
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# This is the list of valid classes for this challenge, in order
# The class id is the index in this list
CLASSES = [
    'none',
    # Clutter classes
    'bottle',
    'cup',
    'knife',
    'bowl',
    'wine glass',
    'fork',
    'spoon',
    'banana',
    'apple',
    'orange',
    'cake',
    'potted plant',
    'mouse',
    'keyboard',
    'laptop',
    'cell phone',
    'book',
    # 'vase',
    # 'pen',
    'clock',

    # furniture classes
    'chair',
    'dining table',
    'couch',
    'bed',
    'toilet',
    'television',
    'microwave',
    'toaster',
    'refrigerator',
    'oven',
    'sink',
    'person'
]
# A simple map to go back from
CLASS_IDS = {class_name: idx for idx, class_name in enumerate(CLASSES)}

# Some helper synonyms, to handle cases where multiple words mean the same class
# This list is used when loading the ground truth to map it to the list above,
# you could add what you want
SYNONYMS = {
    'tv': 'television',
    'tvmonitor': 'television',
    'computer monitor': 'television',    # They're approximately the same, right?
    # 'coffee table': 'table',
    # 'dining table': 'table',
    # 'kitchen table': 'table',
    # 'desk': 'table',
    'stool': 'chair',
    'diningtable': 'dining table',
    'pottedplant': 'potted plant',
    'cellphone': 'cell phone',
    'wineglass': 'wine glass',

    # background classes
    'background': 'none',
    'bg': 'none',
    '__background__': 'none'
}


def get_class_id(class_name):
    """
    Given a class string, find the id of that class
    This handles synonym lookup as well
    :param class_name:
    :return:
    """
    class_name = class_name.lower()
    if class_name in CLASS_IDS:
        return CLASS_IDS[class_name]
    elif class_name in SYNONYMS:
        return CLASS_IDS[SYNONYMS[class_name]]
    return None


def get_class_name(class_id):
    return CLASSES[class_id]


def get_nearest_class(potential_class_name):
    """
    Given a string that might be a class name,
    return a string that is definitely a class name.
    Again, uses synonyms to map to known class names

    :param potential_class_name:
    :return:
    """
    potential_class_name = potential_class_name.lower()
    if potential_class_name in CLASS_IDS:
        return potential_class_name
    elif potential_class_name in SYNONYMS:
        return SYNONYMS[potential_class_name]
    return None
