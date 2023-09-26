import random
from PIL import Image
import json
import matplotlib.pyplot as plt
import os
import datetime

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

WALARIS_CATEGORY_LABEL = [
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

COCO_CLASS_LABELS_NUM2NAME = {label["id"]: label["name"] for label in COCO_CATEGORY_LABEL}

WALALARIS_CLASS_LABELS_NUM2NAME = {label["id"]: label["name"] for label in WALARIS_CATEGORY_LABEL}



def visualize_img_json_file(json_file, 
                            dir_save, 
                            num_images=10,
                            format_json='COCO'):
    """
    Visualize the images from a json file in COCO or WALARIS format

    :param json_file: path to the json file
    :param dir_save: directory to save the images
    :param num_images: number of images to visualize
    :param format_json: format of the json file (COCO or WALARIS)

    :return: None
    """

    # Read the json file
    with open(json_file, 'r') as json_file:
        data = json.load(json_file)
        images = data['images']
        annotations = data['annotations']

    # Get the image path
    dataset_images_path = "/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/"

    # Get the random images from the dataset
    seed = 42
    random.seed(seed)
    random.shuffle(images)
    images_random = images[:num_images]

    # Visualize the images
    for idx, image in enumerate(images_random):
        image_name = image['file_name']
        image_path = dataset_images_path + image_name
        img_id = image['id']

        # Get annotations from the image
        annotations_img = [ann for ann in annotations if ann['image_id'] == img_id]
        
        
        # print(img.size)

        # Visualize the image
        visualise_img(image_path,
                      annotations_img,
                      dir_save,
                      file_name=image_name, 
                      format_json=format_json, 
                      idx=idx,
                      image_id=img_id)


def visualize_img_ID(input_json, 
                     dir_save,
                     img_ids, 
                     format_json='COCO'):
    """
    Visualize the images from a json file in COCO or WALARIS format with the image ID

    :param input_json: path to the json file
    :param dir_save: directory to save the images
    :param img_id: image ID
    :param format_json: format of the json file (COCO or WALARIS)

    :return: None
    """

    # Read the json file
    with open(input_json, 'r') as json_file:
        data = json.load(json_file)
        images = data['images']
        annotations = data['annotations']

    # Get the image path
    dataset_images_path = "/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/"
    
    for idx_img, img_id in enumerate(img_ids):
        # Get the images from the dataset
        for img in images:
            if img["id"] == img_id: # Image ID is unique
                # Read the image from the dataset
                image_name = img['file_name']
                image_path = dataset_images_path + image_name
                break
    
        # Get annotations from the image
        annotations_img = [ann for ann in annotations if ann['image_id'] == img_id]

        # Visualize the image
        visualise_img(image_path, annotations_img, dir_save, file_name=image_name,  format_json=format_json, 
                  idx=idx_img, image_id=img_id)

def visualise_img(image_path,
                  annotations_img,
                  dir_save,
                  file_name, 
                  format_json='COCO', 
                  idx=0,
                  image_id=0):
    
    # Read the image
    img = Image.open(image_path)
        
    dpi_monitor = 76.979166666 # dpi of the monitor

    if format_json == 'COCO':
        dict_categories = COCO_CLASS_LABELS_NUM2NAME
    elif format_json == 'WALARIS':
        dict_categories = WALALARIS_CLASS_LABELS_NUM2NAME

    # Visualize the image
    plt.figure(figsize=(img.size[0]/dpi_monitor, img.size[1]/dpi_monitor))
    ax = plt.gca()
    ax.imshow(img)
    plt.axis('off')

    # Add the annotations to the image
    for ann in annotations_img:
        bbox = ann['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], dict_categories[ann['category_id']], fontsize=8, color='black',
                bbox=dict(facecolor='red', alpha=0.5))
    
    # Save the image
    file_name = f"{idx}_ImageID:{image_id}_{str(datetime.datetime.now()).replace(' ', '-')}.png"
    plt.savefig(os.path.join(dir_save, file_name), pad_inches=0, bbox_inches='tight')
    print(f"Image {file_name} saved!")
    plt.close() 

if __name__ == "__main__":
    
    # visualize_img_json_file

    # The json file path
    json_file = "../data/NoPlaymentLabels/annotations_added/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3).json"

    # The directory to save the images
    dir_save = "../data/NoPlaymentLabels/annotations_added/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3)"

    # Number of images to visualize
    num_images = 1000

    # The format of the json file
    format_json = 'WALARIS'

    visualize_img_json_file(json_file=json_file,
                            dir_save=dir_save,
                            num_images=num_images,
                            format_json=format_json)
    
    # visualize_img_ID
    # input_json = "../data/NoPlaymentLabels/annotations_added_and_json_cleaned/day_noplayment_11092023_train_ADDED_ground-based_objects(threshold=0.3)_CLEANED.json"
    # dir_save = "../data/NoPlaymentLabels/annotations_added_and_json_cleaned/test_clean_dataset"  
    # img_ids = [11575452484848485253, 11575452484848485454, 11575452484848485254, 11575452484848485254, 11575452484848485355, 11575452484848484949, 11575452484848484950, 11575452484848485449, 11575452484848485357, 11575452484848485255, 11575452484848484951, 11575452484848485448, 11575452484848485348, 11575452484848485348, 11575452484848485349, 11575452484848485356, 11575452484848485356, 11575452484848485055, 11575452484848484948, 11575452484848485056, 11575452484848484955, 11575452484848485257, 11575452484848485257, 11575452484848485257, 11575452484848484952, 11575452484848484857, 11575452484848485256, 1149545551484848484853, 1149545551484848485351, 1149545551484848485148, 1149545551484848485354, 1149545551484848485353, 1149545551484848485353, 1149545551484848485352, 11554949484848484951, 11554949484848485157, 11554949484848485351, 11554949484848485052, 11504956484848485057, 11504956484848485357, 11504956484848485055, 11504956484848485053, 11504956484848485150, 11504956484848485355, 11504956484848485148, 11504956484848485152, 11504956484848485452, 11504956484848485151, 11504956484848485448, 11504956484848485451, 11504956484848485248, 11504956484848485149, 11504956484848485054, 11504956484848485453, 11504956484848485056, 11535148484848545657, 11535148484848545657, 11535148484848545657, 11535148484848525456, 11535148484848525456, 11535148484848515054, 11535148484848515054, 11535148484848524854, 11535148484848535757, 11535148484848535757, 11535148484848534957, 11535148484848534957, 11535148484848534957, 11535148484848534957, 11535148484848515156, 11535148484848515156, 11535148484848535156, 11535148484848555352, 11535148484848555352, 11535148484848555352, 11535148484848555350, 11535148484848555350, 11535148484848555350, 11535148484848505149, 11535148484848525552, 11535148484848525552, 11535148484848555453, 11535148484848525054, 11535148484848515055, 11535148484848545656, 11535148484848554948, 11535148484848554948, 11535148484848525551, 11535148484848525551, 11535148484848534950, 11535148484848534950, 11535148484848534950, 11535148484848534950, 11535148484848534950, 11535148484848555149, 11535148484848545153, 11535148484848555349, 11535148484848515257, 11535148484848515257, 11535148484848535152, 11535148484848544850, 11535148484848535049, 11535148484848535049, 11535148484848535049, 11535148484848535049, 11535148484848554853, 11535148484848554853, 11535148484848555052, 11535148484848554954, 11535148484848554954, 11535148484848554850, 11535148484848555448, 11535148484848515155, 11535148484848554950, 11535148484848545348, 11535148484848545348, 11535148484848555556, 11535148484848554857, 11535148484848535052, 11535148484848535756, 11535148484848535149, 11535148484848535050, 11535148484848535050, 11535148484848535050, 11535148484848515255, 11535148484848525149, 11535148484848525149, 11535148484848525457, 11535148484848525457, 11535148484848515254, 11535148484848515254, 11535148484848545152, 11535148484848554855, 11535148484848554855, 11535148484848545253, 11535148484848545253, 11535148484848545253, 11535148484848545449, 11535148484848515157, 11535148484848515157, 11535148484848555050, 11535148484848515252, 11535148484848545251, 11535148484848545251]
    # format_json = 'WALARIS'

    # visualize_img_ID(input_json=input_json,
    #                  dir_save=dir_save,
    #                  img_ids=img_ids,
    #                  format_json=format_json)
    
    # Visualize the images from a json file selected randomly where image ID are normalized (dataset size = 176963)
    # input_json = "../data/NoPlaymentLabels/annotations_added_and_json_cleaned_imgID_normalized/day_noplayment_11092023_train_ADDED_ground-based_objects(threshold=0.3)_CLEANED_imgIDnormalized.json"
    # dir_save = "../data/NoPlaymentLabels/annotations_added_and_json_cleaned_imgID_normalized/test_clean_dataset"  
    # img_ids = random.sample(range(0, 176963), 1000) # 1000 random numbers from 0 to 176963
    # format_json = 'WALARIS'

    # visualize_img_ID(input_json=input_json,
    #                  dir_save=dir_save,
    #                  img_ids=img_ids,
    #                  format_json=format_json)
    




