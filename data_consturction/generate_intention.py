import json
import os
from openai import OpenAI
from typing import List, Dict, Any
import time
from tqdm import tqdm
# Function to extract unique class names from bboxes
def extract_unique_classes(bboxes: List[Dict]) -> List[str]:
    unique_classes = set()
    for bbox in bboxes:
        unique_classes.add(bbox["class_name"])
    return list(unique_classes)

# Function to create prompt for GPT-4o
def create_prompt(caption: str, object_classes: List[str]) -> str:

    objects_str = str(object_classes)
    
    prompt = """Given the following real home environment scene caption and a list of detected objects from this environment, describe how each object might fulfill human intentions and needs in everyday life. Focus on why someone would want to use these objects (such as tired legs desperately needing a comfortable place to rest), what purpose they serve for the person (like I want to keep track of my appointments), and how they meet specific needs in this home context (for reading comfort my eyes require additional brightness in the room). For each object, provide one sentence that captures the human intention behind using it. Do not use punctuation in the middle, and avoid starting every sentence the same way. Make the phrasing natural, varied and written from a first-person perspective expressing a need or want.

Output format: A flat JSON object where each key is the object name (string), and each value is a single sentence describing the human intention for using it, written without punctuation.

Example: { "sofa": "I want to sit down and relax after a long day", "coffee table": "I need somewhere to place my drinks and books within reach", "lamp": "I need some light to read my favorite novel before bed", ... }

Caption: """ + caption + """
Detected objects: """ + objects_str
    
    return prompt

# Function to get affordances from GPT-4o using the specified API endpoint with retry logic
def get_affordances_from_gpt4o(prompt: str, max_retries: int = 3) -> Dict[str, str]:
    # Initialize the client with the provided endpoint and API key
    client = OpenAI(
        base_url="your_base_url",
        api_key="your_api_key"
    )
    
    # Initialize retry counter
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"  Attempt {retry_count + 1} of {max_retries} to get affordances from GPT-4o...")
            
            # Call the API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Try to extract and parse the JSON object from the response
            try:
                # Check if the response contains code blocks
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                else:
                    # If no code blocks, try to find JSON object directly
                    # Look for content between { and }
                    if '{' in content and '}' in content:
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        json_content = content[start_idx:end_idx].strip()
                    else:
                        json_content = content.strip()
                
                # Try to parse JSON
                affordances = json.loads(json_content)
                
                # If we successfully parse the JSON, return it
                if affordances and isinstance(affordances, dict):
                    print(f"  Successfully parsed JSON response on attempt {retry_count + 1}")
                    return affordances
                else:
                    raise json.JSONDecodeError("Empty or invalid JSON object", json_content, 0)
            
            except (json.JSONDecodeError, IndexError) as e:
                print(f"  Error parsing GPT-4o response on attempt {retry_count + 1}: {e}")
                print(f"  Raw response: {content}")
                
                # Increment retry counter and try again
                retry_count += 1
                
                # If we've reached max retries, give up
                if retry_count >= max_retries:
                    print(f"  Failed to parse JSON after {max_retries} attempts, giving up.")
                    return {}
                
                # Wait briefly before retrying to avoid rate limits
                time.sleep(2)
                
                # Add explicit instructions for JSON format in the retry
                prompt += "\n\nIMPORTANT: Please respond ONLY with a valid JSON object. No additional text before or after the JSON. Ensure that all object keys and values are properly quoted strings."
        
        except Exception as e:
            print(f"  Error calling GPT-4o API on attempt {retry_count + 1}: {e}")
            
            # Increment retry counter and try again
            retry_count += 1
            
            # If we've reached max retries, give up
            if retry_count >= max_retries:
                print(f"  Failed to call GPT-4o API after {max_retries} attempts, giving up.")
                return {}
            
            # Wait before retrying
            time.sleep(3)
    
    # If we get here, all retries failed
    return {}

# Function to check if an item is fully processed
def is_item_processed(bboxes):
    # Check if all bboxes have an "affordance" field
    for bbox in bboxes:
        if "affordance" not in bbox:
            return False
    return True

# Main function to process the JSON file with resume capability
def process_json_file(file_path: str, output_path: str):

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = {}
    
    # Count total items
    items_to_process = list(data.items())[:]
    total_items = len(items_to_process)
    
    print(f"Found {total_items} total items to process in the JSON file.")
    
    # Process items
    current_batch = 0
    newly_processed = 0
    skipped_items = 0
    
    # Create progress bar using tqdm
    pbar = tqdm(items_to_process, desc="Processing...", unit="items")
    
    for item_id, item in pbar:
        # Update progress bar description
        pbar.set_description(f"Processing {item_id}")
        
        # Extract bboxes
        bboxes = item.get("bboxes", [])
        
        # Skip if already processed
        if is_item_processed(bboxes):
            pbar.set_postfix(status="processed", skip=True)
            continue
        
        # Extract caption and unique object classes
        caption = item.get("caption", "")
        
        unique_classes = extract_unique_classes(bboxes)
        
        if not unique_classes:
            pbar.set_postfix(status="no objects", skip=True)
            skipped_items += 1
            continue
            
        pbar.set_postfix(objects=len(unique_classes))
        
        # Create prompt and get affordances with retry logic
        prompt = create_prompt(caption, unique_classes)
        affordances = get_affordances_from_gpt4o(prompt, max_retries=3)
        
        if not affordances:
            pbar.set_postfix(status="failed", skip=True)
            skipped_items += 1
            continue
            
        # Update bboxes with affordances
        updated_count = 0
        for bbox in bboxes:
            class_name = bbox["class_name"]
            if class_name in affordances:
                bbox["affordance"] = affordances[class_name]
                updated_count += 1
        
        pbar.set_postfix(updated=f"{updated_count}/{len(bboxes)}")
        
        # Add this item to processed_data after successful processing
        processed_data[item_id] = item
        
        newly_processed += 1
        current_batch += 1
        
        if current_batch % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            pbar.set_postfix(saved=True, processed=newly_processed)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processing completed: {newly_processed} items processed, {skipped_items} items skipped")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Configuration
    json_file_path = "/home/featurize/work/icra2025/anno_withcaption_merged.json"
    output_file_path = "/home/featurize/work/icra2025/anno_withcaption_affordances.json"
    
    # Process the JSON file with resume capability
    process_json_file(json_file_path, output_file_path)