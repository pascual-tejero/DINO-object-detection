import json
import os
from PIL import Image

data_root_annotations = '/home/tarsier/Documents/Walaris/Tasks/Task2_Ground_based_Object_Detection/Code/data_processing/annotations_Walaris_dataset/data/' # Walaris dataset annotations .json file
json_file_path = data_root_annotations + 'Walaris_dataset(1000samples_included_remaining_categories)_coco_format_random_subsampling_from10000_samples_dataset.json'
# json_file_path = data_root_annotations + 'Walaris_dataset_coco_format_random_sampling_10000samples.json'
image_folder_path = '/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/'

with open(json_file_path, 'r') as f:
    coco_data = json.load(f)

cont = 0
image_ids = []
for image_info in coco_data['images']:
    image_id = image_info['id']
    image_name = image_info['file_name']
    image_path = os.path.join(image_folder_path, image_name)

    # Read the image using PIL:
    image = Image.open(image_path)
    image_ids.append(image_id)
    dpi_x, dpi_y = image.info.get("dpi", (None, None))
    print(dpi_x, dpi_y)
    # Do whatever you want with the image here, e.g., display, process, etc.
    continue

set_image_ids = set(image_ids)
print(len(set_image_ids))
