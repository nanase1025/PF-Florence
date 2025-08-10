# Testing Florence-2 for Object Detection

import os
import json
import torch
import numpy as np
import supervision as sv
from modeling_florence2 import Florence2ForConditionalGeneration
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import warnings
from safetensors.torch import load_file

warnings.filterwarnings("ignore", module="supervision")
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model configuration
CHECKPOINT = "microsoft/Florence-2-base-ft" 
WEIGHTS_PATH = "path/to/your/weights"  
REVISION = 'refs/pr/6'

# Dataset classes
class CustomDataset:
    """Base class for reading custom JSON annotation files."""
    def __init__(self, image_directory_path: str, annotations: Dict[str, Dict]):
        self.image_directory_path = image_directory_path
        self.annotations = annotations
        self.image_ids = list(self.annotations.keys())
        
    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.image_ids):
            raise IndexError("Index out of range")

        image_id = self.image_ids[idx]
        annotation = self.annotations[image_id]
        
        if image_id.endswith('.jpg'):
            image_filename = image_id
        else:
            image_filename = f"{image_id}.jpg"
            
        image_path = os.path.join(self.image_directory_path, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
            return (image, annotation, image_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

class SunRGBDAffordanceDataset(Dataset):
    """Dataset for SUNRGBD affordance-based object detection."""
    def __init__(self, image_directory_path: str, annotations: Dict[str, Dict]):
        self.dataset = CustomDataset(image_directory_path, annotations)
        self.indices = list(range(len(self.dataset)))
        
        # Create expanded index mapping to handle multiple bboxes per image
        self.expanded_indices = []
        for idx in self.indices:
            image, annotation, _ = self.dataset[idx]
            bboxes = annotation.get("bboxes", [])
            if bboxes:
                for bbox_idx in range(len(bboxes)):
                    self.expanded_indices.append((idx, bbox_idx))
        
        print(f"Test dataset: {len(self.indices)} images, {len(self.expanded_indices)} total samples")

    def __len__(self):
        return len(self.expanded_indices)

    def __getitem__(self, idx):
        actual_idx, bbox_idx = self.expanded_indices[idx]
        image, annotation, image_id = self.dataset[actual_idx]
        
        bboxes = annotation.get("bboxes", [])
        selected_bbox = bboxes[bbox_idx]
        
        affordance = selected_bbox.get("affordance", "")
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        text_input = affordance
        target = selected_bbox.get("target", "")
        
        return task_prompt, text_input, target, image

def load_custom_dataset(image_dir, annotation_file):
    """Load a custom dataset from local files."""
    print(f"Loading custom dataset from {image_dir}")
    print(f"Using annotation file: {annotation_file}")
    
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    if not os.path.isfile(annotation_file):
        raise ValueError(f"Annotation file not found: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Loaded annotations for {len(annotations)} images")
    
    return {
        "location": image_dir,
        "annotations": annotations
    }

def evaluate_model(model, processor, test_dataset):
    """Evaluate model on test dataset."""
    model.eval()
    targets = []
    predictions = []
    
    print("Running evaluation...")
    for batch in tqdm(test_dataset, desc="Testing"):
        task_prompt, text_input, target, image = batch
        inputs = processor(text=task_prompt+text_input, images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        prediction = processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
        
        target = processor.post_process_generation(target, task=task_prompt, image_size=image.size)
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
        
        prediction.class_id = np.zeros(len(prediction), dtype=np.int32)
        target.class_id = np.zeros(len(target), dtype=np.int32)
        prediction.confidence = np.ones(len(prediction))
        
        targets.append(target)
        predictions.append(prediction)
    
    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )
    
    print(f"Testing Results:")
    print(f"mAP50-95: {mean_average_precision.map50_95:.4f}")
    print(f"mAP50: {mean_average_precision.map50:.4f}")
    print(f"mAP75: {mean_average_precision.map75:.4f}")
    
    return mean_average_precision

def main():
    """Main testing function."""
    # Load model and processor
    print(f"Loading base model from {CHECKPOINT}")
    model = Florence2ForConditionalGeneration.from_pretrained(
        CHECKPOINT, 
        trust_remote_code=True, 
        revision=REVISION
    ).eval().cuda()
    
    print(f"Loading trained weights from {WEIGHTS_PATH}")
    state_dict = load_file(os.path.join(WEIGHTS_PATH, "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT,
        trust_remote_code=True,
        revision=REVISION
    )

    # Load test dataset 
    IMAGE_DIR = "./data/sunrgbd_jpgs"
    TEST_ANNOTATION_FILE = "./data/test.json"
    
    test_dataset_info = load_custom_dataset(IMAGE_DIR, TEST_ANNOTATION_FILE)
    test_dataset = SunRGBDAffordanceDataset(
        image_directory_path=test_dataset_info["location"],
        annotations=test_dataset_info["annotations"]
    )
    
    # Run evaluation
    evaluate_model(model, processor, test_dataset)
    
    print("Testing complete!")

if __name__ == "__main__":
    main()