# Fine-tuning Florence-2 for Object Detection

import os
import re
import json
import torch
import numpy as np
import supervision as sv
import wandb  # Add wandb import
from modeling_florence2 import Florence2ForConditionalGeneration, DaViTFiLMLayer, PromptGenBlock
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    # AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from peft import LoraConfig, get_peft_model
import torch.multiprocessing as mp
from timm.models.layers import trunc_normal_
from functools import partial
mp.set_start_method('spawn', force=True)
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model configuration
CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'

# Training hyperparameters
BATCH_SIZE = 12
NUM_WORKERS = 2
EPOCHS = 20
LEARNING_RATE = 5e-6

# wandb configuration
WANDB_PROJECT = "your_project_name"  # wandb project name
WANDB_ENTITY = None  # Your wandb username or organization name
WANDB_NAME = "your_experiment_name"  # Experiment name

# Pattern for extracting object detection labels
def init_prompt_gen_blocks(model, std=0.02):
    """
    Initialize the parameters of all PromptGenBlock layers in the model
    
    Args:
        model: The model to initialize the PromptGenBlock layers for
        std: The standard deviation for weight initialization
    """
    # Find and initialize all PromptGenBlock layer parameters
    for name, module in model.named_modules():
        if isinstance(module, PromptGenBlock):
            # Initialize prompt_param parameters
            if hasattr(module, 'prompt_param'):
                # Normal distribution initialization for weights
                nn.init.normal_(module.prompt_param, mean=0.0, std=std)
            
            # Initialize linear_layer
            if hasattr(module, 'linear_layer'):
                if hasattr(module.linear_layer, 'weight'):
                    # Use truncated normal distribution initialization for weights
                    trunc_normal_(module.linear_layer.weight, std=std)
                if hasattr(module.linear_layer, 'bias') and module.linear_layer.bias is not None:
                    # Zero initialization for bias
                    nn.init.constant_(module.linear_layer.bias, 0)
            
            # Initialize conv3x3 layer
            if hasattr(module, 'conv3x3'):
                if hasattr(module.conv3x3, 'weight'):
                    # Normal distribution initialization for weights
                    nn.init.normal_(module.conv3x3.weight, mean=0.0, std=std)
                if hasattr(module.conv3x3, 'bias') and module.conv3x3.bias is not None:
                    # Zero initialization for bias
                    nn.init.constant_(module.conv3x3.bias, 0)
    
    return model
def init_film_layers(model, std=0.02):
    """
    Initialize the parameters of all DaViTFiLMLayer layers in the model
    
    Args:
        model: The model to initialize the FiLM layers for
        std: The standard deviation for weight initialization
    """
    # Find and initialize all FiLM layer parameters
    for name, module in model.named_modules():
        if isinstance(module, DaViTFiLMLayer):
            # Initialize scale linear layer
            if hasattr(module, 'scale'):
                if hasattr(module.scale, 'weight'):
                    # Use truncated normal distribution initialization for weights
                    trunc_normal_(module.scale.weight, std=std)
                if hasattr(module.scale, 'bias') and module.scale.bias is not None:
                    # Zero initialization for bias
                    nn.init.constant_(module.scale.bias, 0)
            
            # Initialize shift linear layer
            if hasattr(module, 'shift'):
                if hasattr(module.shift, 'weight'):
                    # Use truncated normal distribution initialization for weights
                    trunc_normal_(module.shift.weight, std=std)
                if hasattr(module.shift, 'bias') and module.shift.bias is not None:
                    # Zero initialization for bias
                    nn.init.constant_(module.shift.bias, 0)
    
    return model

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
        
        # Construct image path (adding jpg extension if needed)
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
    def __init__(self, image_directory_path: str, annotations: Dict[str, Dict], split_ratio=0.8, train=True):
        self.dataset = CustomDataset(image_directory_path, annotations)
        # Split dataset into train/val
        all_indices = list(range(len(self.dataset)))
        split_idx = int(len(all_indices) * split_ratio)
        
        # Use deterministic random to ensure consistent splits
        np.random.seed(42)
        np.random.shuffle(all_indices)
        
        if train:
            self.indices = all_indices[:split_idx]
        else:
            self.indices = all_indices[split_idx:]
        
         # ===== New: reduce dataset size --experiment only =====
        # reduce_ratio=0.1
        # if reduce_ratio < 1.0:
        #     reduced_size = max(1, int(len(self.indices) * reduce_ratio))
        #     self.indices = self.indices[:reduced_size]
        # ===========================

        # Create expanded index mapping to handle multiple bboxes per image
        self.expanded_indices = []
        for idx in self.indices:
            image, annotation, _ = self.dataset[idx]
            bboxes = annotation.get("bboxes", [])
            if bboxes:
                for bbox_idx in range(len(bboxes)):
                    self.expanded_indices.append((idx, bbox_idx))
        
        print(f"{'Train' if train else 'Validation'} dataset: {len(self.indices)} images, {len(self.expanded_indices)} total samples")
        self.train = train

    def __len__(self):
        return len(self.expanded_indices) if self.train else len(self.indices)

    def __getitem__(self, idx):
        # Use expanded indices to get both image index and bbox index
        actual_idx, bbox_idx = self.expanded_indices[idx]
        image, annotation, image_id = self.dataset[actual_idx]
        
        bboxes = annotation.get("bboxes", [])
        selected_bbox = bboxes[bbox_idx]
        
        affordance = selected_bbox.get("affordance", "")
                
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        text_input = affordance
        
        target = selected_bbox.get("target", "")
        
        return task_prompt, text_input, target, image

def collate_fn(batch, processor=None):
    """Collate function for DataLoader in affordance-based detection mode.
    
    For training mode, processes task_prompt, text_input, target, image tuples.
    For evaluation mode, passes through the data.
    """
    # Check if we're in training mode (expecting 4-element tuples)
    if len(batch[0]) == 4:
        task_prompts, text_inputs, targets, images = zip(*batch)
        
        # Combine task prompt and text input
        combined_text = [f"{prompt+text}" for prompt, text in zip(task_prompts, text_inputs)]
        
        # Process inputs
        inputs = processor(text=combined_text, images=list(images), return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        return inputs, targets, images
    else:
        # In evaluation mode, just pass through the data
        return batch


def load_custom_dataset(image_dir, annotation_file):
    """Load a custom dataset from local files.
    
    Args:
        image_dir: Directory containing image files
        annotation_file: JSON file with annotations
        
    Returns:
        Dictionary with dataset information
    """
    print(f"Loading custom dataset from {image_dir}")
    print(f"Using annotation file: {annotation_file}")
    
    # Check if directory and file exist
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    if not os.path.isfile(annotation_file):
        raise ValueError(f"Annotation file not found: {annotation_file}")
    
    # Load annotation data
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Loaded annotations for {len(annotations)} images")
    
    # Return dataset info
    return {
        "location": image_dir,
        "annotations": annotations
    }


def evaluate_model(model, processor, val_dataset):
    model.eval()
    targets = []
    predictions = []
    
    print("Running evaluation...")
    for batch in tqdm(val_dataset, desc="Evaluating"):
        task_prompt, text_input, target, image = batch
        inputs = processor(text=task_prompt+text_input, images=image, return_tensors="pt").to(DEVICE)
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3
            )
        
        # Decode generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        prediction = processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
        
        # Process target
        target = processor.post_process_generation(target, task=task_prompt, image_size=image.size)
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
        
        prediction.class_id = np.zeros(len(prediction), dtype=np.int32)
        target.class_id = np.zeros(len(target), dtype=np.int32)

        # Ensure confidence
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
    print(f"map50_95: {metrics['map50_95']:.2f}")
    print(f"map50: {metrics['map50']:.2f}")
    print(f"map75: {metrics['map75']:.2f}")
    
    wandb.log(metrics)
    
    return mean_average_precision
    
def train_model(train_loader, val_dataset, model, processor, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the model for affordance-based object detection."""
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(epochs):
        print(f"\n{'='*20} Epoch {epoch + 1}/{epochs} {'='*20}")
        # Training phase
        model.train()
        train_loss = 0
        
        for step, (inputs, targets, _) in enumerate(tqdm(train_loader, desc=f"Training")):
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            
            # Prepare labels for training
            labels = processor.tokenizer(
                text=targets,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)
            
            # Forward pass
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
            # 每100步记录一次训练损失
            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}, Learning Rate: {lr_scheduler.get_last_lr()[0]}")
                wandb.log({
                    "train_loss": loss.item(), 
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch + step/len(train_loader)
                })
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch+1})
        
        # Evaluation phase
        if epoch % 1 == 0:
            metrics = evaluate_model(model, processor, val_dataset)
            # Evaluation metrics are recorded in evaluate_model

        # Save checkpoint
        output_dir = f"./final/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
          
        print(f"Model checkpoint saved to {output_dir}")
        
        # Save model files using wandb
        wandb.save(f"{output_dir}/*")

def main():
    """Main function."""
    global processor, model
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_checkpoint": CHECKPOINT,
            "device": str(DEVICE),
        }
    )
    
    # Load model and processor
    print(f"Loading Florence-2 from {CHECKPOINT}")
    # model = AutoModelForCausalLM.from_pretrained(
    #     CHECKPOINT,
    #     trust_remote_code=True,
    #     revision=REVISION
    # ).to(DEVICE)
    model = Florence2ForConditionalGeneration.from_pretrained(
        CHECKPOINT, 
        trust_remote_code=True, 
        revision=REVISION).eval().cuda()
    model = init_film_layers(model)
    model = init_prompt_gen_blocks(model)
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT,
        trust_remote_code=True,
        revision=REVISION
    )

    # Load custom dataset 
    IMAGE_DIR = "./data/sunrgbd_jpgs"
    ANNOTATION_FILE = "./data/train.json"
    dataset = load_custom_dataset(IMAGE_DIR, ANNOTATION_FILE)
    # Create datasets and dataloaders
    train_dataset = SunRGBDAffordanceDataset(
        image_directory_path=dataset["location"],
        annotations=dataset["annotations"],
        split_ratio=0.8,
        train=True
    )
    val_dataset = SunRGBDAffordanceDataset(
        image_directory_path=dataset["location"],
        annotations=dataset["annotations"],
        split_ratio=0.8,
        train=False
    )
    
    # Record dataset size to wandb
    wandb.config.update({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset)
    })
    
    # Create dataloaders
    train_collate_fn = partial(collate_fn, processor=processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=train_collate_fn,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Train the model
    print(f"Starting training for {EPOCHS} epochs...")
    # train_model(train_loader, val_dataset, peft_model, processor, classes, epochs=EPOCHS, lr=LEARNING_RATE)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {all_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")
    # Train the model
    print(f"Starting training for {EPOCHS} epochs...")
    train_model(train_loader, val_dataset, model, processor, epochs=EPOCHS, lr=LEARNING_RATE)    
    # Save final model
    output_dir = f"./final/final_model"
    os.makedirs(output_dir, exist_ok=True)
    # peft_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Final model saved to {output_dir}")
    
    # Finish wandb recording
    wandb.finish()
    
    print("Training and evaluation complete!")
    # return results

if __name__ == "__main__":
    main()