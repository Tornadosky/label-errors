
import random
import json


def get_all_label_ids(dataset):
    label_ids = []
    for entry in dataset:
        for label in entry['labels']:
            label_ids.append(label['id'])
    return label_ids


def get_label_ids_by_min_area(dataset, min_area=200):
    """
    Filters and returns label IDs based on a minimum bounding box area.
    
    Parameters:
    - dataset: A list of dictionaries, each containing 'labels' with bounding box data.
    - min_area (int): The minimum area threshold for the bounding boxes.
    
    Returns:
    - list: A list of label IDs that have a bounding box area greater than or equal to the minimum area.
    """
    eligible_label_ids = []

    # Iterate over the dataset to check each label
    for image_info in dataset:
        for label in image_info['labels']:
            x1, y1, x2, y2 = label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            if area >= min_area:
                eligible_label_ids.append(label['id'])

    return eligible_label_ids


def clamp(value, min_value, max_value):
    '''
    Ensures bounding box is in image frame.
    '''
    return max(min_value, min(value, max_value))

categories = {'bicycle',
       'bus',
       'car',
       'motorcycle',
       'other vehicle',
       'pedestrian',
       'rider',
       'traffic light',
       'traffic sign',
       'truck'
    }


def misclassificate_labels_in_dataset(dataset, categories, fraction=0.1):
    """
    Randomly applies shifting errors to a fraction of the labels in the dataset.

    :param labels: The entire dataset of labels.
    :param fraction: Fraction of labels to modify.
    :return: The dataset with updated labels.
    """
    print("Starting to Misclassify labels...")
    
    all_label_ids = get_all_label_ids(dataset)
    num_to_modify = int(len(all_label_ids) * fraction)
    selected_label_ids = random.sample(all_label_ids, num_to_modify)
    selected_label_ids.sort(key=float)

    total_labels = len(selected_label_ids)
    report_interval = total_labels // 10  # Determine when to report progress

    # Initialize a counter for reporting progress
    counter = 0

    # Iterate over the dataset and modify only selected labels
    for image in dataset:
        labels = image['labels']
        for label in labels:
            if label['id'] in selected_label_ids:
                original_category = label['category']
                possible_categories = categories - {original_category}  # Remove the original category from the possible choices  # Remove the original category from the possible choices
                label['category'] = random.choice(list(possible_categories))
                
                # Increment and report progress at intervals
                counter += 1
                if counter % report_interval == 0:
                    progress_percentage = (counter / total_labels) * 100
                    print(f"Progress: {progress_percentage:.1f}%")

    return selected_label_ids

def alter_bounding_box(label_info, min_factor=0.2, max_factor=0.3):
    """
    Modifies one of the four sides of the bounding box in place, either increasing or decreasing it
    by a random factor between the given minimum and maximum percentages.

    Parameters:
    - label_info (dict): A label dictionary with bounding box data.
    - min_factor (float): Minimum percentage change factor.
    - max_factor (float): Maximum percentage change factor.
    """
    img_width, img_height = 1280, 720  # Image dimensions, specific to BDD100K

    # Extract coordinates
    x1, y1, x2, y2 = label_info['box2d']['x1'], label_info['box2d']['y1'], label_info['box2d']['x2'], label_info['box2d']['y2']

    # Decide randomly whether to change width or height
    if random.choice(['width', 'height']) == 'width':
        # Calculate current width and select a random change factor
        width = x2 - x1
        width_change = random.uniform(min_factor, max_factor) * width
        width_change *= -1 if random.choice([True, False]) else 1

        # Randomly apply width change to either the left or right side
        if random.choice(['left', 'right']) == 'left':
            label_info['box2d']['x1'] = clamp(x1 + width_change, 0, x2)
        else:
            label_info['box2d']['x2'] = clamp(x2 - width_change, x1, img_width)
    else:
        # Calculate current height and select a random change factor
        height = y2 - y1
        height_change = random.uniform(min_factor, max_factor) * height
        height_change *= -1 if random.choice([True, False]) else 1

        # Randomly apply height change to either the top or bottom side
        if random.choice(['top', 'bottom']) == 'top':
            label_info['box2d']['y1'] = clamp(y1 + height_change, 0, y2)
        else:
            label_info['box2d']['y2'] = clamp(y2 - height_change, y1, img_height)


def alternate_size_in_dataset(dataset, fraction=0.1, min_area=400,
                              min_factor=0.2, max_factor=0.3):
    """
    Randomly alters the size of a fraction of the bounding boxes in the dataset that meet a minimum area requirement.
    This alteration involves randomly increasing or decreasing the dimensions of the bounding boxes.

    :param dataset: List of dictionaries, each representing an image with labels.
    :param fraction: Fraction of eligible labels to modify.
    :param min_area: Minimum area of the bounding box to include a label for potential modification.
    :param min_factor: Minimum percentage by which the bounding box dimensions will be changed.
    :param max_factor: Maximum percentage by which the bounding box dimensions will be changed.
    :return: None; modifies the dataset in place.
    """

    print("Starting to Alternate labels sizes...")
    
    satisfactory_label_ids = get_label_ids_by_min_area(dataset, min_area)
    num_to_modify = int(len(satisfactory_label_ids) * fraction)
    selected_label_ids = random.sample(satisfactory_label_ids, num_to_modify)
    selected_label_ids.sort(key=float)

    # Total labels in the dataset for progress calculation
    total_labels = len(selected_label_ids)
    labels_processed = 0

    # Iterate through each image in the dataset
    for image in dataset:
        labels = image['labels']
        # Iterate through each label in the current image
        for label in labels:
            if label['id'] in selected_label_ids:
                # Modify the label in place
                alter_bounding_box(label, min_factor, max_factor)
            
                # Report progress every 10% (adjustable)
                labels_processed += 1
                if labels_processed % (total_labels // 10) == 0:
                    progress_percentage = (labels_processed / total_labels) * 100
                    print(f"Progress: {progress_percentage:.1f}%")
                
    return selected_label_ids


def remove_random_labels_in_dataset(dataset, fraction=0.1):
    """
    Randomly removes a fraction of the labels from the dataset.

    :param dataset: The dataset containing all label information.
    :param fraction: Fraction of labels to remove.
    :return: The dataset with labels removed.
    """
    print("Starting to Remove labels...")
    
    # Gather all label IDs in the dataset
    all_label_ids = get_all_label_ids(dataset)
    num_to_modify = int(len(all_label_ids) * fraction)
    selected_label_ids = random.sample(all_label_ids, num_to_modify)
    selected_label_ids.sort(key=float)

    total_entries = len(dataset)
    progress_checkpoint = total_entries // 10

    # Remove selected labels from each entry in the dataset directly
    for index, entry in enumerate(dataset):
        entry["labels"] = [label for label in entry["labels"] if label["id"] not in selected_label_ids]

        # Print progress every 10%
        if (index + 1) % progress_checkpoint == 0 or index + 1 == total_entries:
            progress = (index + 1) / total_entries * 100
            print(f"Progress: {progress:.1f}%")

    return selected_label_ids


def shift_bounding_box(label_info, min_factor=0.15, max_factor=0.2):
    """
    Shifts the entire bounding box diagonally based on a percentage of its own dimensions,
    ensuring that each shift is at least a minimum percentage and no more than a maximum percentage of its size.

    Parameters:
    - label_info (dict): A label dictionary with bounding box data.
    - min_factor (float): Minimum percentage of the bounding box dimensions to shift.
    - max_factor (float): Maximum percentage of the bounding box dimensions to shift.
    """
    img_width, img_height = 1280, 720  # Image dimensions, specific to BDD100K

    
    # Extract bounding box coordinates
    x1 = label_info['box2d']['x1']
    y1 = label_info['box2d']['y1']
    x2 = label_info['box2d']['x2']
    y2 = label_info['box2d']['y2']
    
    # Calculate width and height of the bounding box
    width = x2 - x1
    height = y2 - y1

    # Calculate random shift factors for x and y based on the bounding box dimensions
    x_shift = random.uniform(min_factor, max_factor) * width * random.choice([-1, 1])
    y_shift = random.uniform(min_factor, max_factor) * height * random.choice([-1, 1])

    # Apply the shifts directly to the bounding box coordinates
    label_info['box2d']['x1'] += x_shift
    label_info['box2d']['y1'] += y_shift
    label_info['box2d']['x2'] += x_shift
    label_info['box2d']['y2'] += y_shift

    # Apply the shifts and use clamp to ensure coordinates stay within the image frame
    label_info['box2d']['x1'] = clamp(label_info['box2d']['x1'], 0, img_width)
    label_info['box2d']['y1'] = clamp(label_info['box2d']['y1'], 0, img_height)
    label_info['box2d']['x2'] = clamp(label_info['box2d']['x2'], 0, img_width)
    label_info['box2d']['y2'] = clamp(label_info['box2d']['y2'], 0, img_height)


def shift_bounding_boxes_in_dataset(dataset, fraction=0.1, min_area=400,
                                    min_factor=0.15, max_factor=0.2):
    """
    Modifies the bounding box by shifting its coordinates diagonally.

    :param label_info: A dictionary containing the bounding box details.
    :param min_factor: Minimum percentage of the bounding box dimensions to shift.
    :param max_factor: Maximum percentage of the bounding box dimensions to shift.
    :return: None; modifies the label_info dictionary in place.
    """
    print("Starting to Shift labels...")
    
    satisfactory_label_ids = get_label_ids_by_min_area(dataset, min_area=min_area)
    num_to_modify = int(len(satisfactory_label_ids) * fraction)
    selected_label_ids = random.sample(satisfactory_label_ids, num_to_modify)
    selected_label_ids.sort(key=float)

    # Total labels in the dataset for progress calculation
    total_labels = len(selected_label_ids)
    labels_processed = 0

    # Iterate through each image in the dataset
    for image in dataset:
        labels = image['labels']
        # Iterate through each label in the current image
        for label in labels:
            if label['id'] in selected_label_ids:
                # Modify the label in place
                shift_bounding_box(label, min_factor, max_factor)
            
                # Report progress every 10% (adjustable)
                labels_processed += 1
                if labels_processed % (total_labels // 10) == 0:
                    progress_percentage = (labels_processed / total_labels) * 100
                    print(f"Progress: {progress_percentage:.1f}%")
    
    return selected_label_ids


if __name__ == "__main__":
    json_labels = "bdd100k_labels_images_val.json"

    # Define the dataset
    with open(json_labels, 'r') as file:
        error_dataset = json.load(file)

    # Clean the dataset
    categories_to_del = ["train", "trailer", "other person"] # too few instances
    for image_data in error_dataset:
        image_data['labels'] = [label for label in image_data['labels'] if label['category'] not in categories_to_del]

    # Labels that have been affected
    error_ids_dict = {
        "misclassification_err": [],
        "size_alternation_err": [],
        "remove_label_err": [],
        "shifting_err": [],
    }

    # Fraction means the fraction of the dataset that will be affected by the error 
    error_ids_dict["remove_label_err"] = remove_random_labels_in_dataset(
        dataset=error_dataset, fraction=0.1
        )
    error_ids_dict["size_alternation_err"] = alternate_size_in_dataset(
        dataset=error_dataset,
        fraction=0.15,
        min_area=400,
        min_factor=0.2,
        max_factor=0.3
        )
    error_ids_dict["shifting_err"] = shift_bounding_boxes_in_dataset(
        dataset=error_dataset,
        fraction=0.07,
        min_area=400, # minimum area of the bounding box to include a label for potential modification
        min_factor=0.15, # percentage of the bounding box dimensions to shift 
        max_factor=0.2
        )
    error_ids_dict["misclassification_err"] = misclassificate_labels_in_dataset(
        dataset=error_dataset, categories=categories, fraction=0.2,
    )

    # Save the modified label id's and dataset
    with open("error_ids_dict.json", 'w') as file:
        json.dump(error_ids_dict, file)
    with open("bdd100k_labels_w_errors.json", 'w') as file:
        json.dump(error_dataset, file)