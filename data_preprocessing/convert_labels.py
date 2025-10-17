import json
import os
import sys
from datetime import datetime
import logging

def create_txt_files(input_json_path, output_directory, logger=None):
    """
    Create YOLO-format .txt files from a custom JSON format containing image annotations.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_directory (str): Directory to save the .txt label files.
    """
    logger = logging.getLogger(__name__) if logger is None else logger
    os.makedirs(output_directory, exist_ok=True)

    # Load the JSON
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)

    # Get list of images
    image_entries = input_data.get("images", [])
    if not isinstance(image_entries, list):
        raise ValueError("'images' should be a list in the JSON file.")

    # Gather unique class names
    categories = set()
    for item in image_entries:
        for annotation in item.get("annotations", []):
            if isinstance(annotation, dict) and "class" in annotation:
                categories.add(annotation["class"])

    # Create a class-to-ID mapping
    category_to_id = {category: idx for idx, category in enumerate(sorted(categories))}
    logger.info(f"Category mapping: {category_to_id}")

    # Process each image
    for item in image_entries:
        base_filename = os.path.splitext(item["file_name"])[0]
        txt_filepath = os.path.join(output_directory, f"{base_filename}.txt")

        width = item.get("width")
        height = item.get("height")
        annotations = item.get("annotations", [])

        # If annotations is empty, create an empty file
        if not annotations:
            with open(txt_filepath, "w") as f:
                f.write("")
            logger.info(f"Created empty file for {item['file_name']}")
            continue

        lines = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            if ("bbox" not in annotation and "segmentation" not in annotation) or "class" not in annotation:
                logger.warning(f"Skipping annotation (missing 'bbox', 'segmentation' or 'class') in {item['file_name']}")
                continue
            if height is None or width is None:
                logger.warning(f"Skipping {item['file_name']} (missing 'width' or 'height')")
                continue

            class_id = category_to_id[annotation["class"]]
            
            if "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                center_x = (x + w / 2) / width
                center_y = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height

                lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

            elif "segmentation" in annotation:
                points = annotation["segmentation"]
                norm_points = []
                for i in range(0, len(points), 2):
                    x = points[i] / width
                    y = points[i + 1] / height
                    norm_points.append(f"{x:.6f} {y:.6f}")
                
                lines.extend(f"{class_id} {point}" for point in norm_points)

        # Save YOLO txt file
        with open(txt_filepath, "w") as f:
            f.write("\n".join(lines))

    logger.info(f"Created {len(image_entries)} .txt files in: {output_directory}")
    return category_to_id


def create_txt_files_coco_format(input_json_path, output_directory, logger=None):
    """
    Create individual .txt files for each image entry in COCO bbox format
    
    Args:
        input_json_path (str): Path to input JSON file
        output_directory (str): Directory to save .txt files
    """
    logger = logging.getLogger(__name__) if logger is None else logger
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Load the input JSON
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    
    # Get unique categories and create category to ID mapping
    categories = set()
    # for item in input_data:
    for item in input_data['images']:
        for annotation in item.get('annotations', []):
            categories.add(annotation['class'])
    
    category_to_id = {}
    for idx, category in enumerate(sorted(categories)):
        category_to_id[category] = idx
    
    logger.info(f"Category mapping: {category_to_id}")
    
    # Process each image
    # for item in input_data:
    for item in input_data['images']:
        # Get filename without extension
        base_filename = os.path.splitext(item["file_name"])[0]
        txt_filename = f"{base_filename}.txt"
        txt_filepath = os.path.join(output_directory, txt_filename)
        
        # Create content for this image's txt file
        lines = []
        for annotation in item.get('annotations', []):
            # bbox = annotation['bbox']
            # x, y, width, height = bbox
            class_id = category_to_id[annotation['class']]
            points = annotation["segmentation"]

            # Normalize coordinates
            norm_points = []
            for i in range(0, len(points), 2):
                x = points[i] / item["width"]
                y = points[i + 1] / item["height"]
                norm_points.extend([f"{x:.6f}", f"{y:.6f}"])
            
            # Format: class_id x y width height (COCO format - absolute coordinates)
            # line = f"{class_id} {x:.3f} {y:.3f} {width:.3f} {height:.3f}"
            line = f"{class_id} " + " ".join(norm_points)
            lines.append(line)
        
        # Write to txt file
        with open(txt_filepath, 'w') as f:
            f.write('\n'.join(lines))
    
    logger.info(f"Created {len(input_data)} .txt files in directory: {output_directory}")
    logger.debug("Format: COCO format (class_id x y width height) - absolute coordinates")
    
    return category_to_id


def convert_to_coco_format(input_json_path, output_json_path, logger=None):

    logger = logging.getLogger(__name__) if logger is None else logger
    
    # Load the input JSON
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    
    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": "Vacant lot detection dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get unique categories from the data
    categories = set()
    for item in input_data:
        for annotation in item.get('annotations', []):
            categories.add(annotation['class'])
    
    # Create categories list with IDs
    category_to_id = {}
    for idx, category in enumerate(sorted(categories), 1):
        category_to_id[category] = idx
        coco_data["categories"].append({
            "id": idx,
            "name": category,
            "supercategory": ""
        })
    
    # Convert images and annotations
    annotation_id = 1
    
    for image_id, item in enumerate(input_data, 1):
        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "width": item["width"],
            "height": item["height"],
            "file_name": item["file_name"],
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        
        # Add annotations for this image
        for annotation in item.get('annotations', []):
            bbox = annotation['bbox']
            x, y, width, height = bbox
            
            # Calculate area
            area = width * height
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_to_id[annotation['class']],
                "segmentation": [],  # Empty for bounding box only
                "area": area,
                "bbox": [x, y, width, height],
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    # Save to output file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print("Conversion completed!")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total categories: {len(coco_data['categories'])}")
    print(f"Categories: {[cat['name'] for cat in coco_data['categories']]}")
    print(f"Output saved to: {output_json_path}")


def convert_from_list(input_data_list, output_json_path):
    """
    Convert from a list of dictionaries (if data is already loaded) to COCO format
    
    Args:
        input_data_list (list): List of dictionaries with image data
        output_json_path (str): Path to output COCO format JSON file
    """
    
    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": "Vacant lot detection dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get unique categories from the data
    categories = set()
    for item in input_data_list:
        for annotation in item.get('annotations', []):
            categories.add(annotation['class'])
    
    # Create categories list with IDs
    category_to_id = {}
    for idx, category in enumerate(sorted(categories), 1):
        category_to_id[category] = idx
        coco_data["categories"].append({
            "id": idx,
            "name": category,
            "supercategory": ""
        })
    
    # Convert images and annotations
    annotation_id = 1
    
    for image_id, item in enumerate(input_data_list, 1):
        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "width": item["width"],
            "height": item["height"],
            "file_name": item["file_name"],
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        
        # Add annotations for this image
        for annotation in item.get('annotations', []):
            bbox = annotation['bbox']
            x, y, width, height = bbox
            
            # Calculate area
            area = width * height
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_to_id[annotation['class']],
                "segmentation": [],  # Empty for bounding box only
                "area": area,
                "bbox": [x, y, width, height],
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    # Save to output file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print("Conversion completed!")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total categories: {len(coco_data['categories'])}")
    print(f"Categories: {[cat['name'] for cat in coco_data['categories']]}") #noqa
    print(f"Output saved to: {output_json_path}")
