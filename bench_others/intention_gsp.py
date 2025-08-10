import torch
import json
import os
import copy
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def create_affordance_prompt(affordance):
    """Create a prompt to identify the single most likely object for an affordance"""
    
    prompt = f"""In this image, identify the SINGLE most likely object I would use when: '{affordance}'

Please respond with ONLY ONE object name - the most suitable object for this purpose.
Do not include any explanations, lists, or additional text.

Important: Your entire response must be a single word or short phrase naming only the most suitable object."""
    
    return prompt

def run_inference(model, processor, image_path, question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open(image_path),
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return response

def main():
    model_path = "/home/shr/wzccode/Qwen2.5-VL-7B-Instruct"
    print("Loading model and tokenizer...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded successfully.")
    
    json_file_path = "/home/shr/wzccode/icra2025/data/merged_affordances_flo_v1.json"
    print(f"Loading JSON data from {json_file_path}...")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    print(f"JSON data loaded with {len(data)} images.")
    
    new_data = copy.deepcopy(data)
    
    image_dir = "/home/shr/wzccode/icra2025/data/sunrgbd_jpgs"
    
    for img_id, img_data in tqdm(data.items(), desc="Processing images", ncols=100):
        image_path = os.path.join(image_dir, f"{img_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist. Skipping.")
            continue
        
        bboxes = img_data.get('bboxes', [])
        
        for i, bbox in enumerate(tqdm(bboxes, desc=f"Processing bboxes for image {img_id}", leave=False, ncols=100)):
            affordance = bbox.get('affordance')
            
            if not affordance:
                continue
                
            prompt = create_affordance_prompt(affordance)
            
            response = run_inference(model, processor, image_path, prompt)
            
            clean_response = response.strip()
            
            if '\n' in clean_response:
                clean_response = clean_response.split('\n')[0].strip()
            
            clean_response = clean_response.lstrip('- •').strip()
            
            if ',' in clean_response:
                clean_response = clean_response.split(',')[0].strip()
            if '.' in clean_response:
                clean_response = clean_response.split('.')[0].strip()
                
            new_data[img_id]['bboxes'][i]['object_name'] = clean_response
        
    output_path = "/home/shr/wzccode/icra2025/code/qwen/merged_affordances_flo_v1_sampled_qwen25vcls.json"
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print("Processing completed.")

if __name__ == "__main__":
    main()