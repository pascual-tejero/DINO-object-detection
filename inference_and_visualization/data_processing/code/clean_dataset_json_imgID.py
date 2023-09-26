import json
from collections import defaultdict


def clean_dataset_json_imgID(input_json, 
                             output_json):
    
    """
    This function takes in a json file and outputs a new json file with the image IDs normalized.

    Parameters
    ----------
    input_json : str
        The path to the input json file.
    output_json : str
        The path to the output json file.

    Returns
    -------
    None.
    """
    
    # Open the input json file
    with open(input_json) as f:
        data = json.load(f)

    # Get the images and annotations
    images = data['images']
    annotations = data['annotations']

    # Make a copy of the images and annotations
    images_copy = images.copy()
    annotations_copy = annotations.copy()

    # Create a new dictionary
    new_imgID_dict = defaultdict()

    # Put the images into the new dictionary
    for idx_img, img in enumerate(images_copy):
        new_imgID_dict[img['id']] = idx_img
        images_copy[idx_img]['id'] = idx_img

    # Put the annotations into the new dictionary
    for ann in annotations_copy:
        ann['image_id'] = new_imgID_dict[ann['image_id']]

    # Save the new images and annotations
    data['images'] = images_copy
    data['annotations'] = annotations_copy

    # Save the new json file
    with open(output_json, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':

    # input_json = '../data/NoPlaymentLabels/annotations_added_and_json_cleaned/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3)_CLEANED.json'
    input_json = '../data/NoPlaymentLabels/annotations_added_and_json_cleaned/day_noplayment_11092023_train_ADDED_ground-based_objects(threshold=0.3)_CLEANED.json'
    
    # output_json = '../data/NoPlaymentLabels/annotations_added_and_json_cleaned_imgID_normalized/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3)_CLEANED_imgIDnormalized.json'
    output_json = '../data/NoPlaymentLabels/annotations_added_and_json_cleaned_imgID_normalized/day_noplayment_11092023_train_ADDED_ground-based_objects(threshold=0.3)_CLEANED_imgIDnormalized.json'
    

    clean_dataset_json_imgID(input_json, output_json)
