import sys, os
import torch, json
import torchvision.transforms as transforms
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent)) # add path to main folder

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

def visualize_save_neg_img(dir_save, 
                           show_in_console=False, 
                           confidence_threshold=0.5,
                           iou_threshold_neg_img=0.5,
                           remove_labels=None,
                           img_dataset=None, 
                           ann_json_file=None):
    """
    Function for visualizing and saving the results of the negative images using a pre-trained model 
    (DINO-4scale-SwinL) in the COCO dataset. The function uses the Hungarian Algorithm to compare the
    ground truth with the predictions. 
    
    A positive image is considered when the IoU between the ground truth and the prediction is higher 
    than the threshold (iou_threshold_neg_img). A negative image is considered when the IoU between the 
    ground truth and the prediction is lower than the threshold (iou_threshold_neg_img). 

    Additionally, it is possible to remove labels from the results. For example, if the user is only 
    interested in the results for the labels "person" and "car", the user can remove the rest of the 
    labels from the results. Just add the labels (name in COCO format) to remove in the list 
    "remove_labels". If the user is ot interested in removing labels, just set the variable 
    "remove_labels" to None.

    This function constantly checks if the image resolution is 4K, 2K, Full HD or HD. If the image is 4K
    or higher, the function downsamples the image to Full HD (DINO pre-trained model only accepts images
    with a maximum resolution of 1920x1080). If the image is 2K, Full HD, HD or any other type of resolution,
    the function does not change the image resolution.

    Also, this function calculates statistics for the results. The statistics are:
        - pos_img: Number of positive images (images with at least one positive example).
        - neg_img: Number of negative images (images with at least one negative example).
        - tp_bbox: Number of true positives for bounding boxes.
        - neg_assign_bbox: Number of negative assignments for bounding boxes.
        - fp_bbox: Number of false positives for bounding boxes.
        - fn_bbox: Number of false negatives for bounding boxes.
        - tp_bbox_label: Number of true positives for labels.
        - fp_bbox_label: Number of false positives for labels.
        - fn_bbox_label: Number of false negatives for labels.

    This function saves two types of images:
        - ALL_EXAMPLES_NEG_IMG: All the examples (ground truth and predictions) for the negative images.
        - NEGATIVE_EXAMPLES_NEG_IMG: Only the negative examples (ground truth and predictions) for the
            negative images.

    At the end, the function saves a .txt file with the results and the statistics. The images are saved
    in the directory "dir_save".

    Parameters
    ----------
    dir_save : str
        Path for saving the negative images.
    show_in_console : bool
        Flag for displaying the images.
    confidence_threshold : float
        Minimum score for a prediction to be considered.
    iou_threshold_neg_img : float
        Threshold for negative examples.
    remove_labels : list
        List of labels to remove from the results.
    img_dataset : str
        Path to the image dataset.
    ann_json_file : str
        Path to the annotation json file.

    Returns
    -------
    dict_stats : dict
        Dictionary with the statistics of the results.
    dict_resolutions : dict
        Dictionary with the resolutions of the images.
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

    # load coco names
    with open('../util/coco_id2name.json') as f: 
        id2name = json.load(f) 
        id2name = {int(k):v for k,v in id2name.items()} # dict from ID to name in COCO format: {id: name}

    # Load datasets
    args.dataset_file = 'coco' # change the dataset name
    args.fix_size = False # set fix_size to False
    args.coco_path = '/home/zhengkai/Datasets/coco' # Just add random string here

    # Change inside the function the path for the dataset and annotation file
    dataset_val = build_dataset(image_set='val', 
                                args=args, 
                                img_dataset=img_dataset, 
                                ann_json_file=ann_json_file) 

    # Initialize a dictionary to store the results
    dict_stats = {"pos_img": 0, "neg_img": 0, "tp_bbox": 0, "neg_assign_bbox": 0, "fp_bbox": 0,
                    "fn_bbox": 0, "tp_bbox_label":0, "fp_bbox_label": 0, "fn_bbox_label": 0}
    
    dict_resolutions = {"4K (3840x2160)": [], "2K (2560x1440)":[], "Full HD (1920x1080)": [], 
                        "HD (1280x720)": []}
    
    full_hd_resolution = (1080, 1920) # set full hd resolution


    # Iterate over the dataset and get predictions
    for num_idx, (image, targets) in enumerate(dataset_val): # iterate over the dataset
        # print(image.size())

        # initialize flags for image resolution 
        image_4K, image_2K, image_full_hd, image_hd, image_other = False, False, False, False, False 

        if image.size()[2] >= 3840 and image.size()[1] >= 2160: # if image size is 4K
            image_4K = True # set flag for 4K image
            image = transforms.Resize(full_hd_resolution)(image) # resize image to full hd
            targets['size'] = torch.tensor([full_hd_resolution[0], full_hd_resolution[1]]) # set new image size
        
        elif image.size()[2] == 2560 and image.size()[1] == 1440: # if image size is 2K
            image_2K = True # set flag for 2K image

        elif image.size()[2] == 1920 and image.size()[1] == 1080: # if image size is full hd
            image_full_hd = True # set flag for full hd image

        elif image.size()[2] == 1280 and image.size()[1] == 720: # if image size is hd
            image_hd = True # set flag for hd image

        else:
            resolution_image = (image.size()[2], image.size()[1]) # get image resolution
            image_other = True
            # If resolution is not in the dictionary, add it
            if f"{str(resolution_image[0])}x{str(resolution_image[1])}" not in dict_resolutions.keys():
                dict_resolutions[f"{str(resolution_image[0])}x{str(resolution_image[1])}"] = [] # add resolution to dictionary

        # build gt_dict for visualization
        box_label = [id2name[int(item)] for item in targets['labels']] # get box labels from IDs in COCO format

        gt_dict = {
            'boxes': box_ops.box_cxcywh_to_xyxy(targets['boxes'].to("cpu")), 
            'image_id': targets['image_id'],
            'size': targets['size'],
            'box_label': box_label,
        }

        # Get model predictions
        with torch.no_grad():
            output = model.cuda()(image[None].cuda()) # get model output
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0] # get predictions
        threshold = confidence_threshold # set a thershold (minimum score for a prediction to be considered)
        scores = output['scores'] # get scores

        select_mask = scores > threshold # get mask for scores > thershold
        boxes = output['boxes'][select_mask] # get boxes in cxcywh format
        labels = output['labels']# get labels
        box_label = [id2name[int(item)] for item in labels[select_mask]] # get box labels from IDs in COCO format

        if remove_labels is not None: # if there are labels to remove
            valid_labels = [label not in remove_labels for label in box_label] # get mask for labels to remove
            boxes = boxes[valid_labels] # get boxes for valid labels
            box_label = np.array(box_label)[valid_labels] # get box labels for valid labels

        # Build a dictionary of predictions for visualization
        pred_dict = {
            'boxes': boxes.to("cpu"),
            'image_id': targets['image_id'],
            'size': targets['size'],
            'box_label': box_label
        }
        
        # Compare predictions with ground truth (Hungarian Algorithm)
        cost_matrix = torch.zeros(len(gt_dict['boxes']), len(pred_dict['boxes'])) # initialize cost matrix
        iou_matrix, _ = box_ops.box_iou(gt_dict['boxes'], pred_dict['boxes']) # calculate iou
        cost_matrix = 1 - iou_matrix # calculate cost
        row_idx, col_idx = linear_sum_assignment(cost_matrix) # solve the assignment problem (Hungarian Algorithm)

        # Detect if the format is xyxy 
        if (gt_dict['boxes'][:,2] > gt_dict['boxes'][:,0]).all(): # if format is xyxy
            gt_dict['boxes'] = box_ops.box_xyxy_to_cxcywh(gt_dict['boxes']) # convert to format cxcywh

        if (pred_dict['boxes'][:,2] > pred_dict['boxes'][:,0]).all(): # if format is xyxy
            pred_dict['boxes'] = box_ops.box_xyxy_to_cxcywh(pred_dict['boxes']) # convert to format cxcywh 
        
        iou_values = torch.tensor([x for x in iou_matrix[row_idx, col_idx]]) # get iou values for the assignment
        dpi_monitor = 76.979166666 # set dpi of the monitor
        fig_size = (gt_dict['size'][1]/dpi_monitor, gt_dict['size'][0]/dpi_monitor) # get image size in inches


        # Visualize the results
        vslzr = COCOVisualizer()        
        
        if (iou_values < iou_threshold_neg_img).any(): # if there is at least one negative example
            dict_stats['neg_img'] += 1 # count the number of images with at least one negative example 
                                         # for sorting them
            if image_4K: # if image is 4K
                dict_resolutions["4K (3840x2160)"].append(num_idx)
            elif image_2K: # if image is 2K
                dict_resolutions["2K (2560x1440)"].append(num_idx)
            elif image_full_hd: # if image is full hd
                dict_resolutions["Full HD (1920x1080)"].append(num_idx)
            elif image_hd: # if image is hd
                dict_resolutions["HD (1280x720)"].append(num_idx)
            elif image_other: # if image is other
                dict_resolutions[f"{str(resolution_image[0])}x{str(resolution_image[1])}"].append(num_idx)
               
     
        elif (iou_values > iou_threshold_neg_img).all() and (iou_matrix.size()[0] > iou_matrix.size()[1] or
                                        iou_matrix.size()[0] < iou_matrix.size()[1]): # if there are no negative examples
            dict_stats['neg_img'] += 1 # count the number of images with at least one negative example 
                                         # for sorting them
            if image_4K: # if image is 4K
                dict_resolutions["4K (3840x2160)"].append(num_idx)
            elif image_2K: # if image is 2K
                dict_resolutions["2K (2560x1440)"].append(num_idx)
            elif image_full_hd: # if image is full hd
                dict_resolutions["Full HD (1920x1080)"].append(num_idx)
            elif image_hd: # if image is hd
                dict_resolutions["HD (1280x720)"].append(num_idx)
            elif image_other: # if image is other
                dict_resolutions[f"{str(resolution_image[0])}x{str(resolution_image[1])}"].append(num_idx)

        elif (iou_values > iou_threshold_neg_img).all() and (iou_matrix.size()[0] == iou_matrix.size()[1]): 
            dict_stats['pos_img'] += 1 # increase counter for positive examples

        # Build a dictionary of ground truths and predictions with all examples for visualization
        temp_gt_pred_all_ex = {'boxes': {"gt": torch.tensor([]), "pred": torch.tensor([])}, 
            'box_label': {"gt": [], "pred": []},
            'image_id': gt_dict['image_id'], 
            'size': gt_dict['size'],
            'iou': 0}
        
         # Build a dictionary of predictions for visualization with the negative examples for visualization
        temp_gt_pred_neg_ex = {'boxes': {"gt": torch.tensor([]), "pred": torch.tensor([])}, 
            'box_label': {"gt": [], "pred": []},
            'image_id': gt_dict['image_id'], 
            'size': gt_dict['size'],
            'iou': 0} 
        
        at_least_neg_ex = False # flag for images with at least one negative example
        iou_all_ex = torch.tensor([]) # initialize tensor for iou values for all examples
        iou_neg_ex = torch.tensor([]) # initialize tensor for iou values for negative examples
        
        # Loop over all examples that had a match between ground truth and prediction
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

            if iou_value < iou_threshold_neg_img: # if iou is lower than the threshold

                at_least_neg_ex = True # set flag for at least one negative example

                dict_stats['neg_assign_bbox'] += 1 # increase counter for negative assignments
                dict_stats["fp_bbox"] += 1 # increase counter for false positives for bounding boxes
 
                temp_gt_pred_neg_ex['boxes']['gt'] = torch.cat((temp_gt_pred_neg_ex['boxes']['gt'],
                                                                gt_box[None]), dim=0) # concatenate ground truth boxes
                temp_gt_pred_neg_ex['boxes']['pred'] = torch.cat((temp_gt_pred_neg_ex['boxes']['pred'], 
                                                                  pred_box[None]), dim=0) # concatenate predicted boxes

                temp_gt_pred_neg_ex["box_label"]["gt"].append(box_label_gt) # append ground truth labels
                temp_gt_pred_neg_ex["box_label"]["pred"].append(box_label_pred) # append predicted labels

                iou_neg_ex = torch.cat((iou_neg_ex, iou_value[None]), dim=0) # concatenate iou values
                
                
            else: # if iou is higher than iou_threshold_neg_img
                dict_stats['tp_bbox'] += 1 # increase counter for true positives

            # count true positives and false positives for labels
            if gt_dict['box_label'][row_i] == pred_dict['box_label'][col_i]:
                dict_stats['tp_bbox_label'] += 1 # increase counter for true positives for labels
            else:
                dict_stats['fp_bbox_label'] += 1 # increase counter for false positives for labels#


        if iou_matrix.size()[0] > iou_matrix.size()[1]: # if there are more ground truth boxes than predictions

            at_least_neg_ex = True # set flag for at least one negative example

            # Get the indexes of the ground truth boxes that were not assigned
            bbox_no_assign = [x for x in range(iou_matrix.size()[0]) if x not in row_idx] 
            dict_stats["neg_assign_bbox"] += len(bbox_no_assign) # increase counter for negative assignments
            dict_stats['fn_bbox'] += len(bbox_no_assign) # increase counter for false negatives for bounding boxes 
            dict_stats['fn_bbox_label'] += len(bbox_no_assign) # increase counter for false negatives for labels

            # Concatenate ground truth boxes and labels for all examples and negative 
            temp_gt_pred_all_ex['boxes']['gt'] = torch.cat((temp_gt_pred_all_ex['boxes']['gt'],
                                                    gt_dict['boxes'][bbox_no_assign]), dim=0)
            temp_gt_pred_neg_ex['boxes']['gt'] = torch.cat((temp_gt_pred_neg_ex['boxes']['gt'], 
                                                    gt_dict['boxes'][bbox_no_assign]), dim=0) 
            for i in bbox_no_assign: # loop over all ground truth boxes that were not assigned
                temp_gt_pred_all_ex["box_label"]["gt"].append(gt_dict['box_label'][i]) # append ground truth labels
                temp_gt_pred_neg_ex["box_label"]["gt"].append(gt_dict['box_label'][i]) # append ground truth labels


        elif cost_matrix.size()[0] < cost_matrix.size()[1]: # if there are more predictions than ground truth boxes
            at_least_neg_ex = True # set flag for at least one negative example

            # Get the indexes of the predictions boxes that were not assigned
            bbox_no_assign = [x for x in range(cost_matrix.size()[1]) if x not in col_idx]
            dict_stats["neg_assign_bbox"] += len(bbox_no_assign) # increase counter for negative assignments
            dict_stats['fp_bbox'] += len(bbox_no_assign) # increase counter for false positives for bounding boxes
            dict_stats['fp_bbox_label'] += len(bbox_no_assign) # increase counter for false positives for labels

            # Concatenate ground truth boxes and labels for all examples and negative examples
            temp_gt_pred_all_ex['boxes']['pred'] = torch.cat((temp_gt_pred_all_ex['boxes']['pred'],
                                                        pred_dict['boxes'][bbox_no_assign]), dim=0)
            temp_gt_pred_neg_ex['boxes']['pred'] = torch.cat((temp_gt_pred_neg_ex['boxes']['pred'], 
                                                     pred_dict['boxes'][bbox_no_assign]), dim=0) 

            for i in bbox_no_assign: # loop over all ground truth boxes that were not assigned
                temp_gt_pred_all_ex["box_label"]["pred"].append(pred_dict['box_label'][i]) # append ground truth labels
                temp_gt_pred_neg_ex["box_label"]["pred"].append(pred_dict['box_label'][i]) # append ground truth labels

        # Visualize the results
        if at_least_neg_ex:
            temp_gt_pred_all_ex['iou'] = iou_all_ex # add iou values to dictionary
            temp_gt_pred_neg_ex['iou'] = iou_neg_ex # add iou values to dictionary

            # Visualize the results for all examples
            vslzr.visualize(image, temp_gt_pred_all_ex, iou=None, caption="ALL_EXAMPLES_NEG_IMG", dpi=None, 
                savedir=dir_save, show_in_console=show_in_console, num=num_idx, 
                size_image=fig_size, view_all=True)  

            # Visualize the results for negative examples
            vslzr.visualize(image, temp_gt_pred_neg_ex, iou=None, caption="NEGATIVE_EXAMPLES_NEG_IMG", dpi=None, 
                savedir=dir_save, show_in_console=show_in_console, num=num_idx, 
                size_image=fig_size, view_all=True)                      

        # Delete variables to free memory
        # del output, image, targets, temp_gt_pred_all_ex, temp_gt_pred_neg_ex, pred_dict
        torch.cuda.empty_cache()   

    file_path = os.path.join(dir_save, "_00_results.txt") # path for saving the results

    with open(file_path, "w") as f: # open file for writing
        # write additional information
        f.write("ADDITIONAL INFORMATION:\n")
        # f.write("- Results for the DINO model: No resizing by DINO or the user applied to any image before getting the predictions\n")
        # f.write("- Results for the DINO model - Resizing applied to 4K images (downsampled to Full HD) by the user before getting the predictions\n")
        # f.write("- Results for the DINO model - Resizing applied to all images by DINO model before getting the predictions\n")
        f.write(f"- The confidence threshold is: {confidence_threshold}\n")
        f.write(f"- The IoU threshold for negative examples is: {iou_threshold_neg_img}\n")
        f.write(f"- The labels and bboxes were removed (not interesting for the results), categories: {remove_labels}\n")
        f.write("- Green: ground truth bounding boxes\n")
        f.write("- Red: predicted bounding boxes\n\n") 

        # Write the dictionary results to the file
        f.write("RESULTS:\n")
        for key, value in dict_stats.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        # Write the number of 4K images
        f.write("IMAGE RESOLUTIONS:\n")
        for key, value in dict_resolutions.items():
            f.write(f"- {key} -> Negative images: {len(value)} // Indices: {value}\n\n")

    return dict_stats, dict_resolutions


if __name__ == '__main__':

    # The directory where the results will be saved
    # directory_save = "./results/negative_examples_subsampling1173_samples_included_remaining_categories_(all_examples_same_image)_threshold=0.3_with_resizing_(only_4Kimages)"
    dir_save = "./results/NoPlaymentLabels/negative_images_results_dataset(1000_images)_threshold=0.3_with_resizing_(only_4Kimages)"
    # directory_save = "./results/test/"

    # Flag for displaying the images in console
    show_in_console = False

    # Confidence threshold for the predictions
    confidence_threshold = 0.3
    # confidence_threshold = 0.5 

    # IoU threshold for negative examples
    iou_threshold_neg_img = 0.5 

    # Path to image dataset
    img_dataset = '/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/' # change the path of the dataset

    # Path to annotation file
    ann_json_file = "./results/NoPlaymentLabels/day_noplayment_11092023_train_random_sampling_1000images.json"

    # 
    stats, idx_4K_images = visualize_save_neg_img(dir_save=dir_save, 
                                                     show_in_console=show_in_console,
                                                     confidence_threshold=confidence_threshold,
                                                     iou_threshold_neg_img=iou_threshold_neg_img,
                                                     img_dataset=img_dataset,
                                                     ann_json_file=ann_json_file)

