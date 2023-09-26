import sys, os
import torch, json
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
import numpy as np
from pathlib import Path 


sys.path.append(str(Path(__file__).resolve().parent.parent)) # add path to main folder

from main import build_model_main
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from datasets import build_dataset
from util import box_ops

# COCO categories
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

# Map from COCO category ID to COCO category name
COCO_CLASS_LABELS_NAME2NUM = {label["name"]: label["id"] for label in COCO_CATEGORY_LABEL}

# Map from COCO category name to COCO category ID
COCO_CLASS_LABELS_NUM2NAME = {label["id"]: label["name"] for label in COCO_CATEGORY_LABEL}

# Map from COCO to Walaris
MAP_COCO_TO_WALARIS = {
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


def inference_and_add_categories_json(dir_save=None, 
                                      show_in_console=False, 
                                      confidence_threshold=0.5,
                                      remove_labels=None,
                                      valid_categories_COCO_labels=None,
                                      img_dataset=None,
                                      input_json_file=False, 
                                      output_json_file=None):
    
    """
    Add annotations of some categories which the user can select in valid_categories_COCO_labels.
    The model used is DINO_4scale_swin pre-trained in COCO dataset. The predictions are made in
    COCO format. 

    The .json file with the name input_json_file in Walaris format. Therefore, a conversion 
    of the predictions from COCO to Walaris format is applied. The annotations are added to the .json 
    file with the name output_json_file. The annotations are added in the last position of the .json 
    file.

    I commented the code to visualize the predictions because to predict in a huge .json dataset file 
    takes a lot of time. If you want to visualize the predictions, uncomment the code.

    Parameters:
    ----------
    dir_save: str
        Path to save the images with the predictions. If None, the images are not saved.
    show_in_console: bool
        If True, the images with the predictions are shown in the console. If False, the images
        with the predictions are not shown in the console.
    confidence_threshold: float
        Minimum score for a prediction to be considered.
    remove_labels: list
        List of labels to remove from the predictions.
    valid_categories_COCO_labels: list
        List of valid categories in COCO format.
    img_dataset: str
        Name of the dataset.
    input_json_file: str
        Path to the json file to add categories.
    output_json_file: str
        Path to save the json file with the added categories.

    Returns:
    -------
    None
    """
    
    # Initialize and Load Pre-trained Models
    model_config_path = "../config/DINO/DINO_4scale_swin.py" # change the path of the model config file
    model_checkpoint_path = "../checkpoints/checkpoint0029_4scale_swin.pth" # change the path of the model checkpoint

    args = SLConfig.fromfile(model_config_path)  # load model config
    args.device = 'cuda'  # force to use cuda (GPU)
    model, criterion, postprocessors = build_model_main(args) # build model architecture
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu') # load model checkpoint
    model.load_state_dict(checkpoint['model']) # load model weights
    _ = model.eval() # set model to eval mode

    # Load datasets
    args.dataset_file = 'coco' # change the dataset name
    args.fix_size = False # set fix_size to False
    args.coco_path = '/home/zhengkai/Datasets/coco' # Just add random string here

    # Change inside the function the path for the dataset and annotation file
    dataset_val = build_dataset(image_set='val', 
                                args=args, 
                                img_dataset=img_dataset, 
                                ann_json_file=input_json_file) 

    if isinstance(input_json_file, str): # if input_json_file is a string

        with open(input_json_file, 'r') as json_file: # Read annotation file
            data = json.load(json_file)

        annotations = data['annotations'] # get annotations
        id_annotations = [annotation['id'] for annotation in annotations] # get annotation IDs
        id_annotations.sort() # sort annotation IDs
        last_value = id_annotations[-1] # get last annotation ID
        del id_annotations # delete id_annotations
    else:
        print("Input json file is not a string") # print error message
        quit() # exit program

    full_hd_resolution = (1080, 1920) # set full hd resolution
    added_annotations = [] # initialize added annotations

    # Iterate over the dataset and get predictions
    for num_idx, (image, targets) in enumerate(dataset_val): # iterate over the dataset  

        image_4K_or_higher = False # initialize image_4K to False
        if image.size()[2] >= 3840 and image.size()[1] >= 2160: # if image size is 4K
            original_image_resolution = (image.size()[2], image.size()[1]) # get original image resolution
            image = transforms.Resize(full_hd_resolution)(image) # resize image to full hd
            targets['size'] = torch.tensor([full_hd_resolution[0], full_hd_resolution[1]]) # set new image size
            image_4K_or_higher = True # set image_4K to True
    
        # Build a dictionary of ground truths for visualization
        # box_label_gt = [COCO_CLASS_LABELS_NUM2NAME[int(item)] for item in targets['labels']] # get box labels from IDs in COCO format

        # gt_dict = {
        #     'boxes': box_ops.box_cxcywh_to_xyxy(targets['boxes'].to("cpu")), 
        #     'image_id': targets['image_id'],
        #     'size': targets['size'],
        #     'box_label': box_label_gt,
        # }

        # Get model predictions
        with torch.no_grad():
            output = model.cuda()(image[None].cuda()) # get model output
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0] # get predictions
        threshold = confidence_threshold # set a thershold (minimum score for a prediction to be considered)
        scores = output['scores'] # get scores

        select_mask = scores > threshold # get mask for scores > thershold
        boxes = output['boxes'][select_mask] # get boxes in cxcywh format
        labels = output['labels']# get labels
        box_label_pred = [COCO_CLASS_LABELS_NUM2NAME[int(item)] for item in labels[select_mask]] # get box labels from IDs in COCO format

        # if remove_labels is not None: # if there are labels to remove
        #     valid_labels = [label not in remove_labels for label in box_label] # get mask for labels to remove
        #     boxes = boxes[valid_labels] # get boxes for valid labels
        #     box_label = np.array(box_label)[valid_labels] # get box labels for valid labels

        # Build a dictionary of predictions for visualization
        pred_dict = {
            'boxes': boxes.to("cpu"),
            'image_id': targets['image_id'],
            'size': targets['size'],
            'box_label': box_label_pred
        }

        # Compare predictions with ground truth (Hungarian Algorithm)
        # cost_matrix = torch.zeros(len(gt_dict['boxes']), len(pred_dict['boxes'])) # initialize cost matrix
        # iou_matrix, _ = box_ops.box_iou(gt_dict['boxes'], pred_dict['boxes']) # calculate iou
        # cost_matrix = 1 - iou_matrix # calculate cost
        # row_idx, col_idx = linear_sum_assignment(cost_matrix) # solve the assignment problem (Hungarian Algorithm)

        # Detect if the format is xyxy 
        # if (gt_dict['boxes'][:,2] > gt_dict['boxes'][:,0]).all(): # if format is xyxy
        #     gt_dict['boxes'] = box_ops.box_xyxy_to_cxcywh(gt_dict['boxes']) # convert to format cxcywh

        if (pred_dict['boxes'][:,2] > pred_dict['boxes'][:,0]).all(): # if format is xyxy
            pred_dict['boxes'] = box_ops.box_xyxy_to_cxcywh(pred_dict['boxes']) # convert to format cxcywh     


        # Get indexes of valid labels
        valid_labels_idx = [idx for idx, label in enumerate(pred_dict['box_label'])
                            if label in valid_categories_COCO_labels]

        if len(valid_labels_idx) > 0:
            # Visualize and save predictions (comment this line if you don't want to visualize the predictions)
            # visualize_save_predictions(image, gt_dict, pred_dict, iou_matrix, row_idx, col_idx,
            #                         dir_save, show_in_console, num_idx)

            print("Adding annotations in .json file. Image index:", num_idx)
            # added_img_ann_ID.append(num_idx) # append image ID

            for val_idx in valid_labels_idx:
                bbox_pred = pred_dict['boxes'][val_idx] # get predicted box
                cat_pred = pred_dict['box_label'][val_idx] # get predicted category

                # Unnormalize prediction with the original image size
                if not image_4K_or_higher:
                    bbox_pred[0] = bbox_pred[0] * image.size()[2]
                    bbox_pred[1] = bbox_pred[1] * image.size()[1]
                    bbox_pred[2] = bbox_pred[2] * image.size()[2]
                    bbox_pred[3] = bbox_pred[3] * image.size()[1]

                else:
                    bbox_pred[0] = bbox_pred[0] * original_image_resolution[0]
                    bbox_pred[1] = bbox_pred[1] * original_image_resolution[1]
                    bbox_pred[2] = bbox_pred[2] * original_image_resolution[0]
                    bbox_pred[3] = bbox_pred[3] * original_image_resolution[1]

                bbox_pred[:2] -= bbox_pred[2:]/2 # unnormbbox[:2] -= unnormbbox[2:] / 2

                last_value += 1 # increase last value

                # Add category to json file
                added_annotations.append({
                    'segmentation': None,
                    'area': (bbox_pred[2] * bbox_pred[3]).item(),
                    'iscrowd': 0,
                    'ignore': 0,
                    'image_id': pred_dict['image_id'],
                    'bbox': bbox_pred.tolist(),
                    'category_id': MAP_COCO_TO_WALARIS[COCO_CLASS_LABELS_NAME2NUM[cat_pred]],
                    'id': last_value 
                })
                
            # visualize_save_predictions(image, gt_dict, pred_dict, iou_matrix, row_idx, col_idx,
            #         dir_save, show_in_console, num_idx)
        

        torch.cuda.empty_cache() # empty cuda cache

    annotations.extend(added_annotations) # extend annotations
    data['annotations'] = annotations # set annotations in json file
    with open(output_json_file, 'w') as json_file: # save json file
        json.dump(data, json_file) # save json file

def visualize_save_predictions(image,
                               gt_dict,
                               pred_dict,
                               iou_matrix,
                               row_idx,
                               col_idx,
                               dir_save,
                               show_in_console,
                               num_idx): 
    
    dpi_monitor = 76.979166666 # set dpi of the monitor
    fig_size = (gt_dict['size'][1]/dpi_monitor, gt_dict['size'][0]/dpi_monitor) # get image size in inches

    # Visualize all examples 
    iou_all_ex = torch.tensor([])

    # Build a dictionary of ground truths and predictions with all examples for visualization
    temp_gt_pred_all_ex = {'boxes': {"gt": torch.tensor([]), "pred": torch.tensor([])}, 
        'box_label': {"gt": [], "pred": []},
        'image_id': gt_dict['image_id'], 
        'size': gt_dict['size'],
        'iou': 0}
    
    iou_values = torch.tensor([x for x in iou_matrix[row_idx, col_idx]]) # get iou values for the assignment

    # # Remove iou_idx and col_idx if the iou value is 0
    # iou_idx = [idx for idx, iou in enumerate(iou_values) if iou > 0]
    # col_idx = [col_idx[idx] for idx in iou_idx]
    # row_idx = [row_idx[idx] for idx in iou_idx]
    # iou_values = [iou_values[idx] for idx in iou_idx]

    for idx, (row_i, col_i) in enumerate(zip(row_idx, col_idx)):

        iou_value = iou_values[idx] # get iou value

        gt_box = gt_dict['boxes'][row_i] # get ground truth box
        pred_box = pred_dict['boxes'][col_i] # get predicted box
        temp_gt_pred_all_ex['boxes']['gt'] = torch.cat((temp_gt_pred_all_ex['boxes']['gt'], 
                                                        gt_box[None]), dim=0) # concatenate ground truth boxes
        temp_gt_pred_all_ex['boxes']['pred'] = torch.cat((temp_gt_pred_all_ex['boxes']['pred'], 
                                                            pred_box[None]), dim=0) # concatenate predicted boxes

        box_label_gt = gt_dict['box_label'][row_i] # get ground truth box label
        box_label_pred = pred_dict['box_label'][col_i] # get predicted box label
        temp_gt_pred_all_ex['box_label']['gt'].append(box_label_gt) # append ground truth box label
        temp_gt_pred_all_ex['box_label']['pred'].append(box_label_pred) # append predicted box label

        iou_all_ex = torch.cat((iou_all_ex, iou_value[None]), dim=0) # concatenate iou values

    temp_gt_pred_all_ex['iou'] = iou_all_ex # set iou values

    if iou_matrix.size()[0] > iou_matrix.size()[1]: # if there are more ground truth boxes than predictions

        # Get the indexes of the ground truth boxes that were not assigned
        bbox_no_assign = [x for x in range(iou_matrix.size()[0]) if x not in row_idx] 
        
        # Concatenate ground truth boxes and labels for all examples and negative 
        temp_gt_pred_all_ex['boxes']['gt'] = torch.cat((temp_gt_pred_all_ex['boxes']['gt'],
                                                gt_dict['boxes'][bbox_no_assign]), dim=0)

        for i in bbox_no_assign: # loop over all ground truth boxes that were not assigned
            temp_gt_pred_all_ex["box_label"]["gt"].append(gt_dict['box_label'][i]) # append ground truth labels


    elif iou_matrix.size()[0] < iou_matrix.size()[1]: # if there are more predictions than ground truth boxes

        # Get the indexes of the predictions boxes that were not assigned
        bbox_no_assign = [x for x in range(iou_matrix.size()[1]) if x not in col_idx]

        # Concatenate ground truth boxes and labels for all examples and negative examples
        temp_gt_pred_all_ex['boxes']['pred'] = torch.cat((temp_gt_pred_all_ex['boxes']['pred'],
                                                    pred_dict['boxes'][bbox_no_assign]), dim=0)

        for i in bbox_no_assign: # loop over all ground truth boxes that were not assigned
            temp_gt_pred_all_ex["box_label"]["pred"].append(pred_dict['box_label'][i]) # append ground truth labels
            
    vslzr = COCOVisualizer() # initialize visualizer         

    vslzr.visualize(image, temp_gt_pred_all_ex, iou=None, caption="ALL_EXAMPLES_NEG_IMG", dpi=None, 
        savedir=dir_save, show_in_console=show_in_console, num=num_idx, 
        size_image=fig_size, view_all=True) 



if __name__ == '__main__':
    
    # The directory where the images with the added annotations will be saved
    # dir_save = "./results/NoPlaymentLabels/added_annotations_images_dataset(1000_images)_threshold=0.3_with_resizing_(only_4Kimages)"
    dir_save = "./results/NoPlaymentLabels/test/"
    # dir_save = None

    # If you want to show the images in the console
    show_in_console = False

    # If you want to remove some labels
    confidence_threshold = 0.3

    # If you want to remove some labels (must to be a list)
    remove_labels = None

    # Change the categories if you want to add other categories in the .json file (with the exact 
    # name in COCO format) -> See COCO_CATEGORY_LABEL
    valid_categories_COCO_labels = ['bicycle', 'boat', 'bus', 'car', 'cat', 'cow', 'dog', 'horse',
                            'motorcycle', 'person', 'traffic light', 'train', 'truck']
    valid_categories_COCO_id = [COCO_CLASS_LABELS_NAME2NUM[label] for label in valid_categories_COCO_labels]
    print("Valid labels:", valid_categories_COCO_labels)
    print("Valid categories:", valid_categories_COCO_id)

    # Evaluate if the number of labels and categories are the same
    assert len(valid_categories_COCO_labels) == len(valid_categories_COCO_id), "The number of labels and categories must be the same"
    
    # Path to the dataset
    img_dataset = '/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/' # change the path of the dataset

    # Path to the json file
    # input_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_train_random_sampling_1000images.json"
    input_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_train.json"
    # input_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_val.json"

    # Path to the output json file
    # output_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_train_random_sampling_1000images_ADDED_ground-based_objects.json"
    output_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_train_ADDED_ground-based_objects.json"
    # output_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_val_ADDED_ground-based_objects.json"

    # Run inference and add categories in json file
    inference_and_add_categories_json(dir_save=dir_save, 
                                      show_in_console=show_in_console, 
                                      confidence_threshold=confidence_threshold, 
                                      remove_labels=remove_labels,
                                      valid_categories_COCO_labels=valid_categories_COCO_labels, 
                                      img_dataset=img_dataset,
                                      input_json_file=input_json_file,
                                      output_json_file=output_json_file)
    
    input_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_val.json"
    output_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_val_ADDED_ground-based_objects.json"

    # Run inference and add categories in json file
    inference_and_add_categories_json(dir_save=dir_save, 
                                      show_in_console=show_in_console, 
                                      confidence_threshold=confidence_threshold, 
                                      remove_labels=remove_labels,
                                      valid_categories_COCO_labels=valid_categories_COCO_labels, 
                                      img_dataset=img_dataset,
                                      input_json_file=input_json_file,
                                      output_json_file=output_json_file)