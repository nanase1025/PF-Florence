import os
import json
import torch
import torchvision.transforms as T
import re
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 8  # Adjust based on GPU memory

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, image_size=448, max_num=12):
    """Simplified preprocessing for a single image"""
    width, height = image.size
    aspect_ratio = width / height
    
    # For simplicity, just resize to square
    resized_img = image.resize((image_size, image_size))
    processed_images = [resized_img]
    
    # Add thumbnail
    thumbnail_img = image.resize((image_size, image_size))
    processed_images.append(thumbnail_img)
    
    return processed_images

def load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def extract_number(filename):
    """Extract the numeric part from the filename for proper sorting"""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def main():
    # Path to the directory containing JPG images
    base_dir = "/home/featurize/work/internvl8b/sunrgbd_jpgs"
    
    # Output JSON file path
    output_path = "/home/featurize/work/internvl8b/image_captions_batch.json"
    
    # Load existing captions if file exists
    captions_dict = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as json_file:
                captions_dict = json.load(json_file)
            print(f"Loaded {len(captions_dict)} existing captions")
        except Exception as e:
            print(f"Error loading existing captions: {str(e)}")
    
    # Load model and tokenizer
    print("Loading InternVL3-8B model...")
    path = "OpenGVLab/InternVL3-8B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    # Set generation config
    generation_config = dict(max_new_tokens=512, do_sample=True)
    
    # Get all jpg files in the directory
    jpg_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    # Sort files by their numeric part
    jpg_files = sorted(jpg_files, key=extract_number)
    
    # Filter out already processed files
    jpg_files = [f for f in jpg_files if os.path.splitext(f)[0] not in captions_dict]
    print(f"Found {len(jpg_files)} unprocessed images")
    
    # Save interval
    save_interval = 10  # Save results every 10 images
    
    # Process in batches with overall progress bar
    overall_progress = tqdm(total=len(jpg_files), desc="Overall Progress")
    processed_count = 0
    
    # Process in batches
    for i in range(0, len(jpg_files), BATCH_SIZE):
        batch_files = jpg_files[i:i+BATCH_SIZE]
        batch_size = len(batch_files)
        
        if batch_size == 0:
            continue
        
        try:
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(jpg_files)-1)//BATCH_SIZE + 1}")
            
            # Prepare batch data
            pixel_values_list = []
            num_patches_list = []
            file_names = []
            
            for jpg_file in batch_files:
                file_path = os.path.join(base_dir, jpg_file)
                pixel_values = load_image(file_path).to(torch.bfloat16).cuda()
                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.size(0))
                file_names.append(os.path.splitext(jpg_file)[0])
            
            # Concatenate all pixel values
            batch_pixel_values = torch.cat(pixel_values_list, dim=0)
            
            # Prepare questions
            questions = ["<image>\nProvide a natural, flowing description of this indoor space as part of a lived-in home. Focus on how furniture and objects relate to each other, how people might interact with them, and how the environment contributes to everyday use, without listing items separately. No more than 300 words."] * batch_size
            
            # Generate captions using batch inference
            responses = model.batch_chat(
                tokenizer, 
                batch_pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config
            )
            
            # Store results
            for file_name, response in zip(file_names, responses):
                captions_dict[file_name] = response
                print(f"Processed {file_name}")
                processed_count += 1
                overall_progress.update(1)  # Update the overall progress bar
            
            # Save periodically
            if processed_count % save_interval == 0 or (i + BATCH_SIZE) >= len(jpg_files):
                with open(output_path, 'w', encoding='utf-8') as json_file:
                    json.dump(captions_dict, json_file, ensure_ascii=False, indent=4)
                print(f"Saved progress after processing {processed_count} images")
                
        except Exception as e:
            print(f"Error processing batch starting at {i}: {str(e)}")
            # Save current progress in case of error
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(captions_dict, json_file, ensure_ascii=False, indent=4)
    
    overall_progress.close()
    
    # Save final results
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(captions_dict, json_file, ensure_ascii=False, indent=4)
    
    print(f"All captions saved to {output_path}")

if __name__ == "__main__":
    main()