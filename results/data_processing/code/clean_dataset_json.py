import json
from visualize_images_json_file import visualise_img
from collections import defaultdict


def clean_dataset_json(input_json, 
                       output_json,
                       dir_save,
                       not_remove_categories_ID=None,
                       overlap_threshold=0.9,
                       visualize=False):

    """
    Takes a .json file and cleans it. The criteria for cleaning is based on if there is
    a bounding box "uav" (class_id = 1 in WWS format) that overlaps with another 
    bounding box (class_id != 1) but the overlap is not considered if the class_id is
    in the list of categories to not remove (not_remove_categories_ID). The overlap is
    calculated using the IoU (Intersection over Union) formula. The output is a .json file
    with the same format as the input file.

    Parameters
    ----------
    input_json : str
        Path to the input .json file.
    output_json : str
        Path to the output .json file.
    dir_save : str
        Path to the directory where the images are going to be saved.
    not_remove_categories_ID : list, optional
        List of categories ID to not remove. The default is None.
    overlap_threshold : float, optional
        Threshold to consider that there is an overlap between two bounding boxes.
        The default is 0.9.
    visualize : bool, optional
        Flag to visualize and save the images. The default is False.

    Returns
    -------
    None.
    """

    # Open the input json file
    with open(input_json) as f:
        data = json.load(f)

    # Get the images and annotations
    images = data["images"]
    annotations = data["annotations"]

    # Get the image path
    if visualize:
        dataset_images_path = "/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/"

    # Image ID which annotations are going to be removed
    images_id_ann_removed = []

    # Global index of the annotations to remove
    global_index_ann_to_remove = [] 
    
    # Copy the annotations
    clean_annotations = annotations.copy() 

    # Create a dictionary to group annotations by image_id
    ann_idx_by_img_id = defaultdict(list)
    ann_val_by_img_id = defaultdict(list)

    # Group annotations by image_id (key) and index or value 
    for idx_ann, ann in enumerate(annotations):
        ann_idx_by_img_id[ann["image_id"]].append(idx_ann)
        ann_val_by_img_id[ann["image_id"]].append(ann)

    # Iterate over the images
    for idx_img, img in enumerate(images):

        ann_index_to_remove = [] # Index of the annotations to remove
        highest_IoU = 0.0 # Highest IoU in an image
        remove_annotations = False # Flag to remove annotations

        img_id = img["id"] # Image ID
        ann_id_list = ann_idx_by_img_id[img_id] # Annotations ID of the image
        ann_value_list = ann_val_by_img_id[img_id] # Annotations value of the image

        if len(ann_id_list) <= 1: # If there is only one annotation in the image,
            continue # continue with the next image
        
        # Iterate over the annotations of the image img
        for i, ann_i in enumerate(ann_value_list): 
                
                if ann_i["category_id"] == 1: # If the category ID is 1 (uav)
                    bbox_i = ann_i["bbox"] # Bounding box i

                    # Iterate over the annotations of the image img
                    for j,  ann_j in enumerate(ann_value_list): 

                        # Check if the category ID of the annotation i is 1 (uav) and the 
                        # category ID of the annotation j is not 1 (uav) and is not in 
                        # the list of categories to  not remove (not_remove_categories_ID)
                        if i != j and ann_j["category_id"] not in not_remove_categories_ID:
                            bbox_j = ann_j["bbox"] # Bounding box j
                            # Calculate the overlap value
                            overlap_value = calculate_overlap_value(bbox_i, bbox_j) 

                            # Check if the overlap value is greater than the threshold
                            if overlap_value > overlap_threshold: 
                                remove_annotations = True # Set the flag to remove annotations
                                images_id_ann_removed.append(img_id) # Append the image ID to remove
                                ann_index_to_remove.append(j) # Append the index of the annotation
                                # Append the global index
                                global_index_ann_to_remove.append(ann_id_list[j]) 
                                print("Overlap: ", overlap_value) # Print the overlap value

                                if highest_IoU < overlap_value: # Update the highest IoU
                                    highest_IoU = overlap_value 

        if remove_annotations: # If there is an overlap between the annotations of the image

            if visualize: # If the flag visualize is True
                highest_IoU = round(highest_IoU, 2) # Round the highest IoU
                image_name = img["file_name"] # Image name
                image_path = dataset_images_path + image_name # Image path
                visualise_img(image_path, ann_value_list, dir_save, image_name,
                            format_json="WWS",
                            idx=f"{idx_img}_ID:{img_id}_"
                            f"Overlap_considered:{overlap_threshold}_"
                            f"MaxIoU:{highest_IoU}_before_cleaning") # Visualize the image

            # Remove the annotations in the list ann_index_to_remove
            ann_index_to_remove = sorted(ann_index_to_remove, reverse=True)
            for index in ann_index_to_remove: # Iterate over the indexes to remove
                del ann_value_list[index] # Remove the annotation

            if visualize:
                visualise_img(image_path, ann_value_list, dir_save, image_name,
                            format_json="WWS",
                            idx=f"{idx_img}_ID:{img_id}_"
                            f"Overlap_considered:{overlap_threshold}_"
                            f"MaxIoU:{highest_IoU}_after_cleaning") # Visualize the image
                
            print("index: ", idx_img) # Print the index of the image
            print("-------------------------------------------------")
    
    # Remove from the input .json file (sorted in reverse order to avoid index problems)
    for index in sorted(global_index_ann_to_remove, reverse=True):
        del clean_annotations[index]

    # Save the new json file
    data["annotations"] = clean_annotations
    with open(output_json, 'w') as outfile:
        json.dump(data, outfile)
    
    # Save in .txt file the images ID where the annotations were removed
    with open(f"{dir_save}/00_images_id_ann_removed.txt", "w") as f:
        f.write("Number of images which were cleaned from noisy annotations (uav + non-uav)"
                f": {len(images_id_ann_removed)}\n\n")
        f.write(f"Images ID with annotations removed:\n")
        f.write(f"{images_id_ann_removed}")

def calculate_overlap_value(bbox1, bbox2):
    """
    Calculate the overlap value between two bounding boxes using the IoU (Intersection over Union)
    formula. The overlap value is the area of the intersection divided by the area of the union.
    
    Note that the coordinates of the bounding boxes are in the format [x, y, width, height], where 
    x and y are the coordinates of the lower-left corner of the bounding box.

    Parameters
    ----------
    bbox1 : list
        Bounding box 1.
    bbox2 : list
        Bounding box 2.

    Returns
    -------
    overlap_value : float
    """
    
    # Extract coordinates and dimensions from the input bboxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the coordinates of the upper-right corners
    x1_right = x1 + w1
    y1_top = y1 + h1
    x2_right = x2 + w2
    y2_top = y2 + h2

    # Calculate the coordinates of the intersection rectangle
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    x_inter_right = min(x1_right, x2_right)
    y_inter_top = min(y1_top, y2_top)

    # Check if there's no overlap (width or height of the intersection is <= 0)
    if x_inter >= x_inter_right or y_inter >= y_inter_top:
        return 0.0

    # Calculate the area of intersection
    area_inter = (x_inter_right - x_inter) * (y_inter_top - y_inter)

    # Calculate the areas of the two bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate the area of the union
    area_union = area1 + area2 - area_inter

    # Calculate the IoU (Intersection over Union)
    overlap_value = area_inter / area_union

    return overlap_value


if __name__ == "__main__":

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

    input_json = "../data/NoPlaymentLabels/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3).json"

    output_json = "../data/NoPlaymentLabels/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3)_CLEANED.json"

    # Dont add "/" at the end of the path   
    dir_save = "../data/NoPlaymentLabels/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3)_CLEANED"

    overlap_threshold = 0.0

    # Do not remove the categories 1, 2, 4, 17 and 18 (uav, airplane, bird, ufo and helicopter)
    not_remove_categories_ID = [1, 2, 4, 17, 18] 

    visualize = False

    clean_dataset_json(input_json, 
                       output_json, 
                       dir_save, 
                       not_remove_categories_ID, 
                       overlap_threshold,  
                       visualize)
    
    # # ---------------------------------------------------------------------------------------------
    
    input_json = "../data/NoPlaymentLabels/day_noplayment_11092023_train_ADDED_ground-based_objects(threshold=0.3).json"

    output_json = "../data/NoPlaymentLabels/day_noplayment_11092023_train_ADDED_ground-based_objects(threshold=0.3)_CLEANED.json"

    # Dont add "/" at the end of the path
    dir_save = "../data/NoPlaymentLabels/day_noplayment_11092023_train_ADDED_ground-based_objects(threshold=0.3)_CLEANED"

    overlap_threshold = 0.0

    # Do not remove the categories 1, 2, 4, 17 and 18 (uav, airplane, bird, ufo and helicopter)
    not_remove_categories_ID = [1, 2, 4, 17, 18] 

    visualize = False

    clean_dataset_json(input_json, 
                        output_json, 
                        dir_save,
                        not_remove_categories_ID, 
                        overlap_threshold,  
                        visualize=visualize)

    # ---------------------------------------------------------------------------------------------

    # input_json = "../data/NoPlaymentLabels/day_noplayment_11092023_train_random_sampling_1000images_ADDED_ground-based_objects(threshold=0.3).json"
    
    # output_json = "../data/NoPlaymentLabels/day_noplayment_11092023_train_random_sampling_1000images_ADDED_ground-based_objects(threshold=0.3)_CLEANED.json"

    # # Dont add "/" at the end of the path
    # dir_save = "../data/NoPlaymentLabels/day_noplayment_11092023_train_random_sampling_1000images_ADDED_ground-based_objects(threshold=0.3)_CLEANED"

    # overlap_threshold = 0.0

    # # # Do not remove the categories 1, 2, 4, 17 and 18 (uav, airplane, bird, ufo and helicopter)
    # not_remove_categories_ID = [1, 2, 4, 17, 18]

    # visualize = True

    # clean_dataset_json(input_json, 
    #                    output_json, 
    #                    dir_save,
    #                    not_remove_categories_ID, 
    #                    overlap_threshold,  
    #                    visualize)

    # # ---------------------------------------------------------------------------------------------