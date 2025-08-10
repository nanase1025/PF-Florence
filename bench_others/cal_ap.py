import json
import numpy as np
import supervision as sv
from tqdm import tqdm
import torch
from PIL import Image
import os
from transformers import AutoProcessor
import warnings
warnings.filterwarnings("ignore", module="supervision")
def calculate_ap_from_json(json_file_path, image_dir=None):
    """
    Calculate average precision (AP) from processed JSON file
    
    Parameters:
        json_file_path: Path to JSON file containing bounding box data
        image_dir: Directory containing image files (if available)
    """
    
    processor = AutoProcessor.from_pretrained(
    "/home/shr/wzccode/Florence-2-base-ft",
    trust_remote_code=True,
    revision='refs/pr/6'
    )
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    targets = []
    predictions = []
    
    print("Calculating AP values...")
    for key, item in tqdm(data.items(), desc="Processing images"):
        if 'bboxes' not in item:
            continue
        
        # Determine image size
        if image_dir and os.path.exists(os.path.join(image_dir, f"{key}.jpg")):
            # If actual image exists, get its dimensions
            img = Image.open(os.path.join(image_dir, f"{key}.jpg"))
            image_size = img.size
        
        for bbox in item['bboxes']:
            # Get class ID
            class_id = bbox.get('class_id', 0)
            
            # Process target bounding boxes (from target field)
            if 'target' in bbox:
                bbox['target'] = "class" + bbox['target']
                target = processor.post_process_generation(bbox['target'], task="<OPEN_VOCABULARY_DETECTION>", image_size=image_size)
                target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image_size)
            # Process predicted bounding boxes (from response field)
            if 'dino2florence_format' in bbox:
                bbox['dino2florence_format'] = "class" + bbox['dino2florence_format']
                response = processor.post_process_generation(bbox['dino2florence_format'], task="<OPEN_VOCABULARY_DETECTION>", image_size=image_size)
                prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image_size)
            # Method 1: Use unified class ID (all set to 0)
            prediction.class_id = np.zeros(len(prediction), dtype=np.int32)
            target.class_id = np.zeros(len(target), dtype=np.int32)

            # Ensure confidence scores exist
            prediction.confidence = np.ones(len(prediction))
            
            targets.append(target)
            predictions.append(prediction)
    mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
    )
    metrics = {
        "map50_95": mean_average_precision.map50_95,
        "map50": mean_average_precision.map50,
        "map75": mean_average_precision.map75
    }
    print(f"map50_95: {metrics['map50_95']:.4f}")
    print(f"map50: {metrics['map50']:.4f}")
    print(f"map75: {metrics['map75']:.4f}")
    return metrics
def main():
    # File paths
    json_file_path = '/home/shr/wzccode/icra2025/code/test_results/qwenvl25_test_result.json'
    image_dir = '/home/shr/wzccode/icra2025/data/sunrgbd_jpgs'  # If image folder exists
    
    # Call function to calculate AP
    metrics = calculate_ap_from_json(json_file_path, image_dir)
    
    # # Print results
    # if metrics:
    #     print("\nEvaluation results:")
    #     for metric_name, value in metrics.items():
    #         print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()