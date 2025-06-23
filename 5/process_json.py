#!/usr/bin/env python3
import json
import sys

def clean_user_message(user_message):
    """Extract text from user_message, handling different content types."""
    if isinstance(user_message, str):
        return user_message
    elif isinstance(user_message, dict):
        # Handle audio transcription
        if user_message.get("content_type") == "audio_transcription":
            return user_message.get("text", "")
        # Handle image asset pointer - no text content, skip
        elif user_message.get("content_type") == "image_asset_pointer":
            return ""
        # Handle real_time_user_audio_video_asset_pointer - no text content, skip
        elif user_message.get("content_type") == "real_time_user_audio_video_asset_pointer":
            return ""
        # For any other dict structure, try to find text field
        else:
            return user_message.get("text", "")
    else:
        return str(user_message)

def process_json_file(input_file, output_file):
    """Process the JSON file to clean up entries."""
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    
    cleaned_data = []
    skipped_count = 0
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
            
        conversation_id = entry.get("conversation_id")
        user_message = entry.get("user_message")
        
        if conversation_id is None or user_message is None:
            continue
        
        # Clean the user_message
        cleaned_message = clean_user_message(user_message)
        
        # Skip entries where we couldn't extract meaningful text
        if not cleaned_message.strip():
            skipped_count += 1
            continue
        
        # Create cleaned entry
        cleaned_entry = {
            "conversation_id": conversation_id,
            "user_message": cleaned_message
        }
        
        cleaned_data.append(cleaned_entry)
    
    print(f"Writing {len(cleaned_data)} cleaned entries to {output_file}...")
    print(f"Skipped {skipped_count} entries with no text content")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    input_file = "Capstone-UCSD/5/green/GRE_Human_5685.json"
    output_file = "Capstone-UCSD/5/green/GRE_Human_5685_cleaned.json"
    
    process_json_file(input_file, output_file) 