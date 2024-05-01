import json
import random
from tabulate import tabulate
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time

WWS_CATEGORY_LABEL = [
    {"id": 1, "name": "uav"},
    {"id": 2, "name": "airplane"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "bird"},
    {"id": 5, "name": "boat"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "car"},
    {"id": 8, "name": "cat"},
    {"id": 9, "name": "cow"},
    {"id": 10, "name": "dog"},
    {"id": 11, "name": "horse"},
    {"id": 12, "name": "motorcycle"},
    {"id": 13, "name": "person"},
    {"id": 14, "name": "traffic_light"},
    {"id": 15, "name": "train"},
    {"id": 16, "name": "truck"},
    {"id": 17, "name": "ufo"},
    {"id": 18, "name": "helicopter"}
]


COCO_CATEGORY_LABEL = [
        {"supercategory": "person","id": 1,"name": "person"},
        {"supercategory": "vehicle","id": 2,"name": "bicycle"},
        {"supercategory": "vehicle","id": 3,"name": "car"},
        {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
        {"supercategory": "vehicle","id": 5,"name": "airplane"},
        {"supercategory": "vehicle","id": 6,"name": "bus"},
        {"supercategory": "vehicle","id": 7,"name": "train"},
        {"supercategory": "vehicle","id": 8,"name": "truck"},
        {"supercategory": "vehicle","id": 9,"name": "boat"},
        {"supercategory": "outdoor","id": 10,"name": "traffic light"},
        {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
        {"supercategory": "outdoor","id": 13,"name": "stop sign"},
        {"supercategory": "outdoor","id": 14,"name": "parking meter"},
        {"supercategory": "outdoor","id": 15,"name": "bench"},
        {"supercategory": "animal","id": 16,"name": "bird"},
        {"supercategory": "animal","id": 17,"name": "cat"},
        {"supercategory": "animal","id": 18,"name": "dog"},
        {"supercategory": "animal","id": 19,"name": "horse"},
        {"supercategory": "animal","id": 20,"name": "sheep"},
        {"supercategory": "animal","id": 21,"name": "cow"},
        {"supercategory": "animal","id": 22,"name": "elephant"},
        {"supercategory": "animal","id": 23,"name": "bear"},
        {"supercategory": "animal","id": 24,"name": "zebra"},
        {"supercategory": "animal","id": 25,"name": "giraffe"},
        {"supercategory": "accessory","id": 27,"name": "backpack"},
        {"supercategory": "accessory","id": 28,"name": "umbrella"},
        {"supercategory": "accessory","id": 31,"name": "handbag"},
        {"supercategory": "accessory","id": 32,"name": "tie"},
        {"supercategory": "accessory","id": 33,"name": "suitcase"},
        {"supercategory": "sports","id": 34,"name": "frisbee"},
        {"supercategory": "sports","id": 35,"name": "skis"},
        {"supercategory": "sports","id": 36,"name": "snowboard"},
        {"supercategory": "sports","id": 37,"name": "sports ball"},
        {"supercategory": "sports","id": 38,"name": "kite"},
        {"supercategory": "sports","id": 39,"name": "baseball bat"},
        {"supercategory": "sports","id": 40,"name": "baseball glove"},
        {"supercategory": "sports","id": 41,"name": "skateboard"},
        {"supercategory": "sports","id": 42,"name": "surfboard"},
        {"supercategory": "sports","id": 43,"name": "tennis racket"},
        {"supercategory": "kitchen","id": 44,"name": "bottle"},
        {"supercategory": "kitchen","id": 46,"name": "wine glass"},
        {"supercategory": "kitchen","id": 47,"name": "cup"},
        {"supercategory": "kitchen","id": 48,"name": "fork"},
        {"supercategory": "kitchen","id": 49,"name": "knife"},
        {"supercategory": "kitchen","id": 50,"name": "spoon"},
        {"supercategory": "kitchen","id": 51,"name": "bowl"},
        {"supercategory": "food","id": 52,"name": "banana"},
        {"supercategory": "food","id": 53,"name": "apple"},
        {"supercategory": "food","id": 54,"name": "sandwich"},
        {"supercategory": "food","id": 55,"name": "orange"},
        {"supercategory": "food","id": 56,"name": "broccoli"},
        {"supercategory": "food","id": 57,"name": "carrot"},
        {"supercategory": "food","id": 58,"name": "hot dog"},
        {"supercategory": "food","id": 59,"name": "pizza"},
        {"supercategory": "food","id": 60,"name": "donut"},
        {"supercategory": "food","id": 61,"name": "cake"},
        {"supercategory": "furniture","id": 62,"name": "chair"},
        {"supercategory": "furniture","id": 63,"name": "couch"},
        {"supercategory": "furniture","id": 64,"name": "potted plant"},
        {"supercategory": "furniture","id": 65,"name": "bed"},
        {"supercategory": "furniture","id": 67,"name": "dining table"},
        {"supercategory": "furniture","id": 70,"name": "toilet"},
        {"supercategory": "electronic","id": 72,"name": "tv"},
        {"supercategory": "electronic","id": 73,"name": "laptop"},
        {"supercategory": "electronic","id": 74,"name": "mouse"},
        {"supercategory": "electronic","id": 75,"name": "remote"},
        {"supercategory": "electronic","id": 76,"name": "keyboard"},
        {"supercategory": "electronic","id": 77,"name": "cell phone"},
        {"supercategory": "appliance","id": 78,"name": "microwave"},
        {"supercategory": "appliance","id": 79,"name": "oven"},
        {"supercategory": "appliance","id": 80,"name": "toaster"},
        {"supercategory": "appliance","id": 81,"name": "sink"},
        {"supercategory": "appliance","id": 82,"name": "refrigerator"},
        {"supercategory": "indoor","id": 84,"name": "book"},
        {"supercategory": "indoor","id": 85,"name": "clock"},
        {"supercategory": "indoor","id": 86,"name": "vase"},
        {"supercategory": "indoor","id": 87,"name": "scissors"},
        {"supercategory": "indoor","id": 88,"name": "teddy bear"},
        {"supercategory": "indoor","id": 89,"name": "hair drier"},
        {"supercategory": "indoor","id": 90,"name": "toothbrush"}
    ]

COCO_CLASS_LABELS_NUM2NAME = {label["id"]: label["name"] for label in COCO_CATEGORY_LABEL}

COCO_CLASS_LABELS_NAME2NUM = {label["name"]: label["id"] for label in COCO_CATEGORY_LABEL}

WALALARIS_CLASS_LABELS_NAME2NUM = {label["name"]: label["id"] for label in WWS_CATEGORY_LABEL}

WALALARIS_CLASS_LABELS_NUM2NAME = {label["id"]: label["name"] for label in WWS_CATEGORY_LABEL}

### and *** are not contaminated categories (different categories in WWS are combined in the 
# same category in COCO)
MAP_WWS_TO_COCO_IDS = {
    1: 5,   # uav (1) -> airplance (5) ***
    2: 5,   # airplane (2) -> airplane (5) ***
    3: 2,   # bicycle (3) -> bicycle (2)
    4: 16,  # bird (4) -> bird (16) ###
    5: 9,   # boat (5) -> boat (9)
    6: 6,   # bus (6) -> bus (6)
    7: 3,   # car (7) -> car (3)
    8: 17,  # cat (8) -> cat (17)
    9: 21,  # cow (9) -> cow (21)
    10: 18, # dog (10) -> dog (18)
    11: 19, # horse (11) -> horse (19)
    12: 4,  # motorcycle (12) -> motorcycle (4)
    13: 1,  # person (13) -> person (1)
    14: 10, # traffic_light (14) -> traffic light (10)
    15: 7,  # train (15) -> train (7)
    16: 8,  # truck (16) -> truck (8)
    17: 16, # ufo (17) -> bird (16) ###
    18: 5   # helicopter (18) -> airplane (5) ***
} 

MAP_COCO_TO_WWS_IDS = {
    1: 13,  # person (1) -> person (13)
    2: 3,   # bicycle (2) -> bicycle (3)
    3: 7,   # car (3) -> car (7)
    4: 12,  # motorcycle (4) -> motorcycle (12)
    5: 2,   # airplane (5) -> uav (2)
    6: 6,   # bus (6) -> bus (6)
    7: 15,  # train (7) -> train (15)
    8: 16,  # truck (8) -> truck (16)
    9: 5,   # boat (9) -> boat (5)
    10: 14, # traffic light (10) -> traffic_light (14)
    11: 0,  # fire hydrant (11) -> None
    13: 0,  # stop sign (13) -> None
    14: 0,  # parking meter (14) -> None
    15: 0,  # bench (15) -> None
    16: 4,  # bird (16) -> bird (4)
    17: 8,  # cat (17) -> cat (8)
    18: 10, # dog (18) -> dog (10)
    19: 11, # horse (19) -> horse (11)
    20: 0,  # sheep (20) -> None
    21: 9,  # cow (21) -> cow (9)
    22: 0,  # elephant (22) -> None
    23: 0,  # bear (23) -> None
    24: 0,  # zebra (24) -> None
    25: 0,  # giraffe (25) -> None
    27: 0,  # backpack (27) -> None
    28: 0,  # umbrella (28) -> None
    31: 0,  # handbag (31) -> None
    32: 0,  # tie (32) -> None
    33: 0,  # suitcase (33) -> None
    34: 0,  # frisbee (34) -> None
    35: 0,  # skis (35) -> None
    36: 0,  # snowboard (36) -> None
    37: 0,  # sports ball (37) -> None
    38: 0,  # kite (38) -> None
    39: 0,  # baseball bat (39) -> None
    40: 0,  # baseball glove (40) -> None
    41: 0,  # skateboard (41) -> None
    42: 0,  # surfboard (42) -> None
    43: 0,  # tennis racket (43) -> None
    44: 0,  # bottle (44) -> None
    46: 0,  # wine glass (46) -> None
    47: 0,  # cup (47) -> None
    48: 0,  # fork (48) -> None
    49: 0,  # knife (49) -> None
    50: 0,  # spoon (50) -> None
    51: 0,  # bowl (51) -> None
    52: 0,  # banana (52) -> None
    53: 0,  # apple (53) -> None
    54: 0,  # sandwich (54) -> None
    55: 0,  # orange (55) -> None
    56: 0,  # broccoli (56) -> None
    57: 0,  # carrot (57) -> None
    58: 0,  # hot dog (58) -> None
    59: 0,  # pizza (59) -> None
    60: 0,  # donut (60) -> None
    61: 0,  # cake (61) -> None
    62: 0,  # chair (62) -> None
    63: 0,  # couch (63) -> None
    64: 0,  # potted plant (64) -> None
    65: 0,  # bed (65) -> None
    67: 0,  # dining table (67) -> None
    70: 0,  # toilet (70) -> None
    72: 0,  # tv (72) -> None
    73: 0,  # laptop (73) -> None
    74: 0,  # mouse (74) -> None
    75: 0,  # remote (75) -> None
    76: 0,  # keyboard (76) -> None
    77: 0,  # cell phone (77) -> None
    78: 0,  # microwave (78) -> None
    79: 0,  # oven (79) -> None
    80: 0,  # toaster (80) -> None 
    81: 0,  # sink (81) -> None
    82: 0,  # refrigerator (82) -> None
    84: 0,  # book (84) -> None
    85: 0,  # clock (85) -> None
    86: 0,  # vase (86) -> None
    87: 0,  # scissors (87) -> None
    88: 0,  # teddy bear (88) -> None
    89: 0,  # hair drier (89) -> None
    90: 0   # toothbrush (90) -> None
}

def convert_WWS_format_to_COCOformat(annotations_dataset, 
                                        annotations_dataset_COCO_format):
    """
    Convert the .json file from WWS format to COCO format. The annotations set is the only
    portion of the dataset that is converted. The rest of the dataset remains the same.
        
    Args:
        - annotations_dataset (str): Path to the annotations dataset.
        - annotations_dataset_COCO_format (str): Path to the annotations dataset

    Returns:
        - None
    """

    # Read the JSON file (1st portion dataset)
    with open(annotations_dataset, 'r') as file:
        data = json.load(file)

    # Create a dictionary for the custom dataset
    dataset_COCO_format = {}

    # Set the dataset information
    dataset_COCO_format["info"] = {
        "description": "My COCO Dataset",
        "version": "1.0",
        "year": 2023,
        "contributor": "WWS",
        "date_created": "2023-06-20"
    }

    # Set the license information
    dataset_COCO_format["licenses"] = [
        {
            "id": 1,
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        }
    ]

    # Process the images
    dataset_COCO_format["images"] = []
    for dict_img in data["images"]:
        temp = {
            "license": 1,
            "file_name": dict_img["file_name"],
            "height": dict_img["height"],
            "width": dict_img["width"],
            "id": dict_img["id"]
        }
        dataset_COCO_format["images"].append(temp)

    # Set the categories from COCO dataset
    dataset_COCO_format["categories"] = COCO_CATEGORY_LABEL


    # ------------ CUSTOM DATASET -----------                       --- COCO DATASET ---
    # [{'supercategory': 'none', 'name': 'uav', 'id': 1},           -> 5 (airplane)
    # {'supercategory': 'none', 'name': 'airplane', 'id': 2},       -> 5 (airplane)
    # {'supercategory': 'none', 'name': 'bicycle', 'id': 3},        -> 2 (bicycle)
    # {'supercategory': 'none', 'name': 'bird', 'id': 4},           -> 16 (bird)
    # {'supercategory': 'none', 'name': 'boat', 'id': 5},           -> 9 (boat)
    # {'supercategory': 'none', 'name': 'bus', 'id': 6},            -> 6 (bus)
    # {'supercategory': 'none', 'name': 'car', 'id': 7},            -> 3 (car)
    # {'supercategory': 'none', 'name': 'cat', 'id': 8},            -> 17 (cat)
    # {'supercategory': 'none', 'name': 'cow', 'id': 9},            -> 21 (cow)
    # {'supercategory': 'none', 'name': 'dog', 'id': 10},           -> 18 (dog)
    # {'supercategory': 'none', 'name': 'horse', 'id': 11},         -> 19 (horse)
    # {'supercategory': 'none', 'name': 'motorcycle', 'id': 12},    -> 4 (motorcycle)
    # {'supercategory': 'none', 'name': 'person', 'id': 13},        -> 1 (person)
    # {'supercategory': 'none', 'name': 'traffic_light', 'id': 14}, -> 10 (traffic light)
    # {'supercategory': 'none', 'name': 'train', 'id': 15},         -> 7 (train)
    # {'supercategory': 'none', 'name': 'truck', 'id': 16},         -> 8 (truck)
    # {'supercategory': 'none', 'name': 'ufo', 'id': 17},           -> 16 (bird)
    # {'supercategory': 'none', 'name': 'helicopter', 'id': 18}]    -> 5 (airplane)

    # Process the annotations
    dataset_COCO_format["annotations"] = []
    for dict_ann in data["annotations"]:
        temp = {
            "segmentation": None,
            "area": dict_ann["area"],
            "iscrowd": dict_ann["iscrowd"],
            "image_id": dict_ann["image_id"],
            "bbox": dict_ann["bbox"],
            "category_id": MAP_WWS_TO_COCO_IDS[dict_ann["category_id"]],
            "id": dict_ann["id"]
        }
        dataset_COCO_format["annotations"].append(temp)

    # Save the dictionary as a JSON file
    with open(annotations_dataset_COCO_format, 'w') as file:
        json.dump(dataset_COCO_format, file)


def get_random_sample_from_json_file(original_json_file,
                                     new_json_file,
                                     sample_size,
                                     seed=None,
                                     not_include_categories=None,
                                     not_include_annotations_from_category=None,
                                     include_remaining_categories=False):
    """
    Get a random sample from a JSON file in COCO or WWS format or any other dataset format.

    Note that if you do not want to either not include certain categories (not_include_categories), 
    not include certain annotations from certain categories (not_include_annotations_from_category) 
    or include some remaining categories in case the number of annotations is low 
    (include_remaining_categories), you need to be consistent with the category IDs and the format of 
    the JSON file (COCO, WWS or any other format).
    
    For example, if you do not want to include the category "person" you need to take into account that in COCO
    format the category ID for "person" is 1, while in WWS format the category ID for "person" is 13. 
    Therefore, if you want to exclude the category "person" you need to specify the category ID=[1] if the JSON
    file is in COCO format or the category ID=[13] if the JSON file is in WWS format. The parameter 
    type of not_include_categories, not_include_annotations_from_category and include_remaining_categories 
    must be a list with the IDs of the categories to exclude or include.
    
    Args:
        - original_json_file (str): Path to the original JSON file.
        - new_json_file (str): Path to the new JSON file.
        - sample_size (int): Number of images to select.
        - seed (int, optional): Seed for the random selection. Defaults to None.
        - not_include_categories (list, optional): List of categories to not include in the sample. 
                                                   Defaults to None.
        - not_include_annotations_from_category (list, optional): List of categories to not include 
                                                                  in the sample. Defaults to None.
        - include_remaining_categories (list, optional): List of categories to include in case their number 
                                                         of annotation is low in the sample. Defaults to False.  
        
    Returns:
        - None
    """
    
    # Load the original JSON file
    with open(original_json_file, 'r') as file:
        data = json.load(file)

    # Extract the 'images' and 'annotations' from the data
    images = data.get('images', [])
    annotations = data.get('annotations', [])

    # If not_include_categories is a list, then we want to exclude those images that contain annotations
    if isinstance(not_include_categories, list):
        not_images_ids = []
        for ann in annotations:
            image_id = ann['image_id']
            category_id = ann['category_id']
            if category_id in not_include_categories:
                not_images_ids.append(image_id)
        not_images_ids = set(not_images_ids)
        images = [img for img in images if img['id'] not in not_images_ids]


    # Select a random sample of images
    random.seed(seed) # Set the seed
    # start_time = time.time()

    # If not_include_annotations_from_category is a list, then we want to exclude those randomly 
    # selected images that only contain annotations or no annotations from not_include_annotations_from_category
    if isinstance(not_include_annotations_from_category, list):
        random_images = []  # List of images that will be selected
        not_include_set = set(not_include_annotations_from_category)  # Convert to a set for faster membership testing
        images_data = images.copy()  # Copy the list of images to avoid modifying the original list

        while len(random_images) < sample_size and images_data:
            image_selected = random.choice(images_data)  # Select a random image
            images_data.remove(image_selected)  # Remove image from images parameter

            categories_in_image = {ann['category_id'] for ann in annotations if ann['image_id'] == image_selected['id']}
            filtered_categories = categories_in_image - not_include_set  # Remove unwanted categories efficiently

            if filtered_categories:  # If the set of filtered categories is not empty
                random_images.append(image_selected)

        # If there are still more images left to select, randomly select from the remaining images.
        remaining_images = sample_size - len(random_images)
        if remaining_images > 0 and images_data:
            random_images.extend(random.sample(images_data, remaining_images))
            print("There are still {} images left to select".format(remaining_images))
            print("Included images from the remaining images")

    else: 
        # If not_include_annotations_from_category is None, then we can select the random sample directly
        random_images = random.sample(images, sample_size)

    # end_time = time.time()
    # print("Time elapsed: {}".format(end_time - start_time))

    # Extract the IDs of the selected random images. It is supposed that the IDs are unique
    random_images_ids = {image['id'] for image in random_images}
    assert len(random_images_ids) == len(random_images), "The IDs of the selected random images are not unique"

    # TODO: Include unlabeled images if necessary
    include_unlabeled_images = False
    if include_unlabeled_images:
        pass
    else:
        pass
        # # Extract the IDs of the unlabelled images
        # unlabeled_images_ids = {image['id'] for image in selected_random_images if image['id'] not in [ann['image_id'] for ann in selected_random_images]}

        # # Filter the selected random images to include only those with annotations
        # selected_random_images_ids = [image["id"] for image in selected_random_images if image['id'] not in unlabeled_images_ids]

    if isinstance(include_remaining_categories, list):
        # If include_remaining_categories is a list, then we want to include those randomly selected images 
        # that only contain annotations from include_remaining_categories
        random_images_ids_old = random_images_ids.copy()
        random_images_ids.update({ann['image_id'] for ann in annotations if ann['category_id']
                                    in include_remaining_categories and ann['image_id'] not in random_images_ids_old})
        
        # Append the images that contain annotations from include_remaining_categories
        # remaining_images_ids = random_images_ids - random_images_ids_old
        random_images.extend([img for img in images if img['id'] in random_images_ids and
                                img['id'] not in random_images_ids_old])

        if isinstance(not_include_annotations_from_category, list):
            # Update the annotations to include those from include_remaining_categories and 
            # exclude those from not_include_annotations_from_category
            included_annotations = [ann for ann in annotations if ann["image_id"] in random_images_ids
                                    and ann['category_id'] not in not_include_annotations_from_category]
        else:
            # Update the annotations to include those from include_remaining_categories
            included_annotations = [ann for ann in annotations if ann["image_id"] in random_images_ids]

    else:
        # Filter the annotations to include only those corresponding to the selected random images ID
        included_annotations = [ann for ann in annotations if ann["image_id"] in random_images_ids]


    # Update the 'images' and 'annotations' in the data with the selected random images and filtered annotations
    data['images'] = random_images
    data['annotations'] = included_annotations

    # Save the modified data to a new JSON file
    with open(new_json_file, 'w') as file:
        json.dump(data, file)


def analyse_json_file(json_file_dataset, format='COCO'):
    """
    Analyse a JSON file in COCO or WWS format or any other dataset format. The function prints the
    statistics of the dataset.

    The statistics include:
        - Number of images and unique images
        - Number of annotations and unique annotations
        - Number of annotations per category ID
        - Mean, standard deviation, median, maximum and minimum number of annotations per image

    Additionally, the function saves the statistics to an Excel file. The Excel file is saved in the same
    directory as the JSON file and has the same name as the JSON file but with the extension .xlsx. Also,
    a histogram of the number of annotations per image is saved in the same directory as the JSON file and
    has the same name as the JSON file but with the extension .png.

    This function also detects potential problems in the dataset such as:
        - Images with the same ID in the image set
        - Images with the same ID in the annotation set
        - Annotations with the same ID in the annotation set
        - Image ID and annotation ID not the same in the image set and annotation set respectively
        - Category ID not in the official dictionary of categories (COCO or WWS)
        - Category ID not in the annotation set

    Args:
        - json_file_dataset (str): Path to the JSON file in COCO format.
        - format (str, optional): Format of the categories in JSON file. Defaults to 'COCO'.

    Returns:
        - None
    """

    # Load the random sample JSON file
    with open(json_file_dataset, 'r') as file:
        data = json.load(file)

    # Extract the 'images' and 'annotations' from the data
    images = data["images"]
    categories = data["categories"]
    annotations = data["annotations"]

    # Count the number of image IDs and extract the unique image IDs in the image set
    image_ids_imgset = [img['id'] for img in images]
    unique_image_ids_imgset = set(image_ids_imgset)

    # Initialize the dictionary with the number of annotations per category
    if format == 'COCO':
        categories_count = {category['id']: 0 for category in COCO_CATEGORY_LABEL}
    elif format == 'WWS':
        categories_count = {category['id']: 0 for category in WWS_CATEGORY_LABEL}

    categories_not_included = [] # List of categories not included in the official dictionary of categories
    categories_not_included_flag = False # Flag to indicate if there are categories not included in the 
                                         # annotation set with respect to the official dictionary of categories

    # Count the number of annotations per image and extract the unique image IDs in the 
    # annotation set
    annotations_per_image = {}
    unique_image_ids_annset = set()
    unique_annotations_ids_annset = set()


    for ann in annotations:
        ann_image_id = ann['image_id'] # Get the image ID
        unique_image_ids_annset.add(ann_image_id) # Add the image ID to the set of unique image IDs
        unique_annotations_ids_annset.add(ann['id']) # Add the annotation ID to the set of unique annotation IDs
        category_id = ann['category_id'] # Get the category ID

        if ann_image_id in annotations_per_image:
            annotations_per_image[ann_image_id] += 1 # Increment the number of annotations per image
        else:
            annotations_per_image[ann_image_id] = 1 # Initialize the number of annotations per image

        if category_id in categories_count:
            # If the category ID is in the dictionary of categories, then increment the number of
            # annotations per category
            categories_count[category_id] += 1 
        else:     
            # Test a potential error in the .json file dataset, if the category ID in the annotation
            # set is not in the official dictionary of categories
            categories_not_included.append(category_id)
            categories_not_included_flag = True

    # Separate labeled and unlabeled images
    labeled_images = unique_image_ids_annset
    unlabeled_images = [img for img in images if img['id'] not in labeled_images]

    # Count the number of annotations with an image ID that is not in the image set
    not_included_image_ids = [ann['image_id'] for ann in annotations if ann['image_id'] 
                              not in unique_image_ids_imgset]

    # Prepare data for Excel file saving and histogram plotting 
    workbook = xlsxwriter.Workbook(f'{json_file_dataset}.xlsx')
    ann_sheet = workbook.add_worksheet("Annotations per category")
    ann_sheet.write_row(0, 0, ['ID',f'Category {format} format', 'Number of annotations'])

    # Get the number of annotations per category
    if format == 'COCO':
        dict_off_categories = COCO_CATEGORY_LABEL # Get the official dictionary of categories

        for idx, cat in enumerate(dict_off_categories):

            # Write the category ID, name and number of annotations to the Excel file in starting from
            # idx+1 row and 0 column     
            ann_sheet.write_row(idx + 1, 0, [str(cat["id"]), cat['name'], categories_count[cat['id']]])

    elif format == 'WWS':
        dict_off_categories =  WWS_CATEGORY_LABEL # Get the official dictionary of categories

        for idx, cat in enumerate(dict_off_categories):

            # Write the category ID, name and number of annotations to the Excel file in starting from
            # idx+1 row and 0 column
            ann_sheet.write_row(idx + 1, 0, [str(cat["id"]), cat['name'], categories_count[cat['id']]])
            
    # Calculate the mean, standard deviation, median, maximum and minimum number of annotations per image
    mean_ann_p_image = np.mean(list(annotations_per_image.values()))
    std_ann_p_image = np.std(list(annotations_per_image.values()))
    median_ann_p_image = np.median(list(annotations_per_image.values()))
    maximum_ann_image = np.max(list(annotations_per_image.values()))
    minimum_ann_image = np.min(list(annotations_per_image.values()))

    # Add a new worksheet for statistical data
    stat_sheet = workbook.add_worksheet('Statistical Data')
    stat_sheet.write_row(0, 0, ['Statistic', 'Value'])
    stat_sheet.write_row(1, 0, ['Total images', len(images)])
    stat_sheet.write_row(2, 0, ['Unique image IDs (image set)', len(unique_image_ids_imgset)])
    stat_sheet.write_row(3, 0, ['Unique image IDs (annotation set)', len(unique_image_ids_annset)])
    stat_sheet.write_row(4, 0, ['Total annotations', len(annotations)])
    stat_sheet.write_row(5, 0, ['Unique annotations IDs (annotation set)', len(unique_annotations_ids_annset)])
    stat_sheet.write_row(6, 0, ['Number of images with annotations', len(labeled_images)])
    stat_sheet.write_row(7, 0, ['Number of images without annotations', len(unlabeled_images)])
    stat_sheet.write_row(8, 0, ['Number of annotations without image ID in the image set', len(not_included_image_ids)])
    stat_sheet.write_row(9, 0, ['Mean of annotations per image', mean_ann_p_image])
    stat_sheet.write_row(10, 0, ['Standard deviation of annotations per image', std_ann_p_image])
    stat_sheet.write_row(11, 0, ['Median of annotations per image', median_ann_p_image])
    stat_sheet.write_row(12, 0, ['Maximum number of annotations in an image', maximum_ann_image])
    stat_sheet.write_row(13, 0, ['Minimum number of annotations in an image', minimum_ann_image])

    # Add a new worksheet for errors
    stat_sheet = workbook.add_worksheet('Errors') # Add a new worksheet for errors

    # If the number of unique image IDs in the image set is not equal to the number of images,
    # then it means that there are images with the same ID (id) in the image set
    if len(unique_image_ids_imgset) != len(images):
        err_msg = ("There are images with the same image ID (id) in the image set: "
                   f"Number of unique image IDs (id) in image set = {len(unique_image_ids_imgset)} != "
                   f"Number of images = {len(images)}")
        print(err_msg)
        stat_sheet.write_row(1, 0, [err_msg])
        
    # If the number of unique image IDs in the annotation set is not equal to the number of images,
    # then it means that there are images with the same ID (image_id) in the annotation set
    if len(unique_image_ids_annset) != len(images):
        err_msg = ("There are images with the same image ID (image_id) in the annotation set: "
                   f"Number of unique image IDs (image_id) in annotation set = {len(unique_image_ids_annset)} "
                   f"!= Number of images = {len(images)}")
        print(err_msg)
        stat_sheet.write_row(2, 0, [err_msg])
        
    # If the number of unique annotation IDs in the annotation set is not equal to the number of
    # annotations, then it means that there are annotations with the same ID (id) in the annotation set
    if len(unique_annotations_ids_annset) != len(annotations):
        err_msg = ("There are annotations with the same annotation ID (id) in the annotation set: "
                   f"Number of unique annotation IDs (id) in annotation set = {len(unique_annotations_ids_annset)} "
                   f"!= Number of annotations = {len(annotations)}")
        print(err_msg)
        stat_sheet.write_row(3, 0, [err_msg])

    # If the set of unique image IDs (id) in the image set is not equal to the set of unique image IDs
    # in the annotation set (image_id), then it means that there are images and annotations whose IDs do not 
    # match in the image set and annotation set respectively
    if unique_image_ids_imgset != unique_image_ids_annset:
        err_msg = ("The unique image IDs from the image set (id) and unique annotation IDS (image_id) from "
                   "the annotation set are not the same")
        print(err_msg)
        stat_sheet.write_row(4, 0, [err_msg])
        
    # If the number of categories in the category set is not equal to the number of categories that
    # appear in the official dictionary of categories, then it means that there are categories in the
    # annotation set that do not appear in the official dictionary of categories
    if len(categories) != len(dict_off_categories):
        err_msg = (f"The number of categories ({len(categories)}) in the category set does not "
                   "match the number of categories that appear in the official dictionary of categories "
                   f"({len(dict_off_categories)}) of {format} format")
        print(err_msg)
        stat_sheet.write_row(5, 0, [err_msg])
    
    # If there are categories in the annotation set that are not included in the official dictionary of
    # categories, then it means that there are categories in the annotation set that do not appear in the
    # official dictionary of categories
    if categories_not_included_flag:
        err_msg = ("There are categories in the annotation set (category_id) that are not included in the "
                   f"official dictionary of categories of {format} format: "
                   f"{sorted(set(categories_not_included))} ")
        print(err_msg)                                      
        stat_sheet.write_row(6, 0, [err_msg])

    # Close the workbook
    workbook.close()

    # Print the results
    print("\n")
    print("Total images: {}".format(len(images)))
    print("Unique image IDs (image set): {}".format(len(unique_image_ids_imgset)))
    print("Unique image IDs (annotation set): {}".format(len(unique_image_ids_annset)))
    print("Total annotations: {}".format(len(annotations)))
    print("Unique annotations IDs (annotation set): {}".format(len(unique_annotations_ids_annset)))
    print("Number of images with annotations: {}".format(len(labeled_images)))
    print("Number of images without annotations: {}".format(len(unlabeled_images)))
    print("Number of annotations without image ID in the image set: {}".format(len(not_included_image_ids)))
    print("Mean of annotations per image: {}".format(mean_ann_p_image))
    print("Standard deviation of annotations per image: {}".format(std_ann_p_image))
    print("Median of annotations per image: {}".format(median_ann_p_image))
    print("Maximum number of annotations in an image: {}".format(maximum_ann_image))
    print("Minimum number of annotations in an image: {}".format(minimum_ann_image))
    print("\n")    
    
    # Plot histogram of annotations per image
    plt.hist(list(annotations_per_image.values()), bins=100)

    # Add labels and title
    plt.xlabel('Number of Annotations')
    plt.ylabel('Frequency')
    plt.title('Histogram of Annotations per Image')
    plt.savefig(f'{json_file_dataset}.png')
    plt.close()
    # plt.show()

    # Print file names that are saved
    print("The statistical data is saved in {}".format(f'{json_file_dataset}.xlsx'))
    print("The histogram of annotations per image is saved in {}".format(f'{json_file_dataset}.png'))

    # Return the results
    return unlabeled_images, labeled_images, categories_count


def visualize_image_and_annotations_bbox(images_dataset, 
                                         annotations_dataset_json,
                                         format='COCO'):
    """
    Visualize an image and its annotations with a bounding box and the category name. The image and
    annotations are randomly selected from the dataset if the user does not specify a category. Otherwise,
    the user can select a category and the image and annotations will be randomly selected from the dataset
    corresponding to the selected category.

    Args:
        - images_dataset (str): Path to the images dataset.
        - annotations_dataset_json (str): Path to the annotations dataset in JSON format.
        - format (str, optional): Format of the JSON file. Defaults to 'COCO'.

    Returns:
        - None
    """

    # Load the dataset
    with open(annotations_dataset_json, 'r') as file:
        data = json.load(file)

    # Extract the 'images' and 'annotations' from the data
    images = data.get('images', [])
    annotations = data.get('annotations', [])

    # Ask the user which category to visualize
    if format == 'COCO':
        valid_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 21]
        dict_categories = COCO_CLASS_LABELS_NUM2NAME

    elif format == 'WWS':
        valid_categories = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
        dict_categories = WALALARIS_CLASS_LABELS_NUM2NAME

    image_category = -1
    while image_category not in valid_categories:
        if format == 'COCO':
            print("\nType a valid category (COCO format) are:")
            print("1. Person / 2. Bicycle / 3. Car / 4. Motorcycle / 5. Airplane / "
                    "6. Bus / 7. Train / 8. Truck / 9. Boat / 10. Traffic Light / 16. Bird /"
                    "/ 17. Cat / 18. Dog / 19. Horse / 21. Cow")
            print("Type 0 if you want to visualize random categories")

        elif format == 'WWS':
            print("\nType a valid category (WWS format) are:")
            print("1. UAV / 2. Airplane / 3. Bicycle / 4. Bird / 5. Boat / 6. Bus / 7. Car / 8. Cat / " 
                  "9. Cow / 10. Dog / 11. Horse / 12. Motorcycle / 13. Person / 14. Traffic Light / "
                  "15. Train / 16. Truck / 17. UFO / 18. Helicopter")
            print("Type 0 if you want to visualize random categories")

        try:
            image_category = int(input("\nEnter a category number: "))

        except ValueError: {  
            print("\n\nThe input was not a valid integer.\n\n")       
        }
            
        for ann in annotations:
            if ann["category_id"] == image_category:
                break
            else:
                image_category = -1
                print("The category does not have any annotations. Please, type another category.\n")
                break
    
    # Ask the user how many images to visualize
    num_images = 0
    while num_images <= 0:
        try:
            num_images = int(input("Number of images to visualize: "))
        except ValueError: {  
            print("The input was not a valid integer.")        
        }



    # Select a random image
    if image_category == 0:
        random_images = random.sample(images, num_images)
    else:
        filtered_images = []
        for ann in annotations:
            if ann["category_id"] == image_category:
                image_id = ann["image_id"]
                filtered_images.extend([img for img in images if img["id"] == image_id])
        random_images = random.sample(filtered_images, min(num_images, len(filtered_images)))

    dpi_monitor = 76.979166666 # Monitor resolution in dpi


    for image in random_images:
        # Extract the annotations corresponding to the selected random image
        annotations_image = [ann for ann in annotations if ann["image_id"] == image["id"]]

        # Visualize the random image with its annotations
        image_load = Image.open(images_dataset + image["file_name"])

        # Visualize the image
        fig = plt.figure(figsize=(image_load.size[0]/dpi_monitor, image_load.size[1]/dpi_monitor))
        ax = plt.Axes(fig, [0., 0., 1., 1.]) # Set the axis
        ax.set_axis_off() # Remove the axis
        fig.add_axes(ax) # Add the axis to the figure

        for ann in annotations_image:
            bbox = ann["bbox"]
        
            category = dict_categories[ann["category_id"]]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, 
                                     edgecolor="r", facecolor='none',alpha=0.7)
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], category, fontsize=5, color='black', 
                    bbox=dict(facecolor="r", alpha=0.7, boxstyle='round,pad=0.2'))
        plt.imshow(image_load)
        plt.show()
        

        

if __name__=='__main__':
    # -------------------------------------- PARAMETERS ------------------------------------------------------
    
    # WWS images dataset
    WWS_img_dataset = '/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/'

    # Number of images to select from the dataset
    sample_size_number = 150000 

    # Random sample .json file from the WWS .json file converted to COCO format
    original_dataset = '/home/tarsier/Documents/WWS/Tasks/Task3_Training/data_preprocessing/data/merged_train.json'
    # original_dataset = '../data/NoPlaymentLabels_DINO_4scale_SwinL_add_annotations/day_noplayment_11092023_train.json'
    # random_sampling_dataset = f'../data/NoPlaymentLabels_DINO_4scale_SwinL_add_annotations/day_noplayment_11092023_train_{sample_size_number}images.json'

    # format_cat_json = 'COCO'
    format_cat_json = 'WWS'


    # ----------------------------- CONVERT FROM WWS FORMAT TO COCO FORMAT -------------------------------
    
    # WWS .json file converted to COCO format
    # dataset_WWS_json = './../data/NoPlaymentLabels/day_noplayment_11092023_train.json'    
    # dataset_COCO_json = './../data/NoPlaymentLabels/day_noplayment_11092023_train_coco_format.json'

    # Conversion from WWS dataset to COCO format dataset
    # convert_WWS_format_to_COCOformat(dataset_WWS_json,
    #                                      dataset_COCO_json)

    # -------------------------------------- RANDOM SAMPLE FROM DATASET --------------------------------------

    # Get a random sample from a json file in either COCO or WWS format
    # get_random_sample_from_json_file(original_dataset,
    #                                  random_sampling_dataset,
    #                                  sample_size=sample_size_number,
    #                                  not_include_categories=None,
    #                                  not_include_annotations_from_category=None,
    #                                  include_remaining_categories=None)
    
    # get_random_sample_from_json_file(original_dataset,
    #                                        random_sampling_dataset,
    #                                        sample_size=sample_size_number,
    #                                        not_include_categories=[4, 16, 17],
    #                                        not_include_annotations_from_category=[4, 5, 16, 17],
    #                                        include_remaining_categories=[2, 7, 9, 18])
    
    # Not include categories: [4, 5, 16, 17] -> Motorcyle, Airplane, Bird, Cat (COCO format)
    # Not include annotations from category: [4, 5, 16, 17] -> Motorcyle, Airplane, Bird, Cat
    # We include airplane images but not its annotations because airplane category is in
    # almost every image, otherwise the dataset would be too small (only 137 images)
    # We can include all annotations from a category if its number is relatively small, in such case
    # we can include category 2 (bicycle), 7 (train), 9 (boat) and 18 (dog) because they are not
    # in many images [2, 7, 9, 18]

    # -------------------------------------- ANALYSE DATASET JSON FILE --------------------------------------

    # Analyse the random sample from json file
    analyse_json_file(json_file_dataset=original_dataset,
                        format=format_cat_json)
    # analyse_json_file(json_file_dataset=random_sampling_dataset,
    #                   format=format_cat_json)

    # ------------------------------------------- VISUALIZE IMAGES -------------------------------------------

    # Visualize a random image from the .json file in COCO format
    # visualize_image_and_annotations_bbox(images_dataset=WWS_img_dataset, 
    #                                      annotations_dataset_json=original_dataset,
    #                                      format=format_cat_json)
    # visualize_image_and_annotations_bbox(images_dataset=WWS_img_dataset, 
    #                                      annotations_dataset_json=random_sampling_dataset,
    #                                      format=format_cat_json)
    

    

# https://stackoverflow.com/questions/10799417/performance-and-memory-allocation-comparison-between-list-and-set
# If you don't care about the ordering, and don't delete elements, then it really boils down 
# to whether you need to find elements in this data structure, and how fast you need those 
# lookups to be.

# Finding an element by value in a HashSet is O(1). In an ArrayList, it's O(n).

# If you are only using the container to store a bunch of unique objects, and iterate over them 
# at the end (in any order), then arguably ArrayList is a better choice since it's simpler and 
# more economical.