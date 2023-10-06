from PIL import Image
import os
import json
# import matplotlib.pyplot as plt

def get_img_resolution_folder_images():

    """
    Analyze the resolution of the images in a folder of images. It prints the unique resolutions 
    found in the folder.

    :return: None
    """

    # Define the folder path containing the images
    # folder_path = './results/negative_examples_subsampling1173_samples_included_remaining_categories_(all_examples_same_image)_threshold=0.5_with_NOresizing'
    folder_path = "./results/test"

    # Create a set to store unique image resolutions
    unique_resolutions = set()

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Open the image file using Pillow
            with Image.open(os.path.join(folder_path, filename)) as img:
                # Get the image resolution
                resolution = img.size
                # Add the resolution to the set
                unique_resolutions.add(resolution)

    print(f"Unique resolutions: {unique_resolutions}")

def get_img_resolution_json_file(json_file_path,
                                 txt_name_save):
    """
    Analyze the resolution of the images in a json file. It saves in a .txt file the 
    following information:
        - Number of images
        - Number of unique resolutions
        - Unique resolutions
        - Number of images per resolution
        - Indices of the images per resolution

    :param json_file: The json file path
    :param txt_name_save: The name of the json file to save the resolution image analysis

    :return: None
    """
    # Read the json file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    images = data['images'] # list of dictionaries

    dataset_images_path = "/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/" # path to the images
    unique_resolutions = set() # set to store unique image resolutions
    dict_resolutions = {} # dictionary to store the number of images per resolution

    for idx, image in enumerate(images):
        print(idx) 
        image_name = image['file_name'] # get the image name
        image_path = dataset_images_path + image_name # get the image path
        with Image.open(image_path) as img:
            resolution = img.size # get the image resolution
            # If the resolution is not in the dictionary, add it
            if f"{resolution[0]}x{resolution[1]}" not in dict_resolutions.keys(): 
                dict_resolutions[f"{resolution[0]}x{resolution[1]}"] = []
            # Add the image index to the dictionary
            dict_resolutions[f"{resolution[0]}x{resolution[1]}"].append(idx) 
            unique_resolutions.add(resolution) # add the resolution to the set

    # Save in .txt file
    with open(f'{txt_name_save}_analysis_image_resolutions.txt', 'w') as f:
        f.write(f"Number of images: {len(images)}\n") 
        f.write(f"Number of unique resolutions: {len(unique_resolutions)}\n")
        f.write(f"Unique resolutions: {unique_resolutions}\n")
        f.write("\n")
        for key, value in dict_resolutions.items():
            f.write(f"- {key} -> Images: {len(value)} // Indices: {value}\n\n")

    print(f"Image resolution analysis saved in {txt_name_save}_analysis_image_resolutions.txt")

            
if __name__ == '__main__':
    #----------------------------------- PARAMETERS -----------------------------------#
    json_file_path = "./results/NoPlaymentLabels/day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3).json"
    txt_name_save = "day_noplayment_11092023_val_ADDED_ground-based_objects(threshold=0.3)_ANALYSIS_image_resolutions"
    #------------------------------------- MAIN ---------------------------------------#
    # get_img_resolution_folder_images()
    get_img_resolution_json_file(json_file_path,
                                 txt_name_save)
    


    


















    # ---------------------------------- TEST ----------------------------------------#

    # walaris_dataset = '/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/'
    # image_1 = walaris_dataset + 'uav/uav_286/000000117684.png'
    # image_2 = walaris_dataset + 'uav/uav_15/000000064363.png'

    # # Read the images
    # img_1 = Image.open(image_1)
    # img_2 = Image.open(image_2)


    # # Save the images with matplotlib
    # ax = plt.gca() # get current axis
    # plt.axis('off')

    # ax.imshow(img_1)
    # plt.savefig('./results/test/image_1.png', dpi=132)
    # plt.axis('off')

    # ax.imshow(img_2)
    # plt.savefig('./results/test/image_2.png', dpi=132)


    # Save the images with exact resolution
    # w = 195
    # h = 841

    # im_np = numpy.random.rand(h, w)

    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(w,h)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(im_np)
    # fig.savefig('figure.png', dpi=1)