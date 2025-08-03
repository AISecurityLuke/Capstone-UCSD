import os
import json
import glob
import random
import re

# Directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Normalization mappings
html_fingerprint_entities = [
    {'char': "'", 'named': '&apos;', 'decimal': '&#39;', 'hex': '&#x27;'},
    {'char': '"', 'named': '&quot;', 'decimal': '&#34;', 'hex': '&#x22;'},
    {'char': '<', 'named': '&lt;', 'decimal': '&#60;', 'hex': '&#x3C;'},
    {'char': '>', 'named': '&gt;', 'decimal': '&#62;', 'hex': '&#x3E;'},
    {'char': '&', 'named': '&amp;', 'decimal': '&#38;', 'hex': '&#x26;'},
    {'char': '/', 'named': None, 'decimal': '&#47;', 'hex': '&#x2F;'},
    {'char': '(', 'named': None, 'decimal': '&#40;', 'hex': '&#x28;'},
    {'char': ')', 'named': None, 'decimal': '&#41;', 'hex': '&#x29;'},
]
unicode_escape_entities = [
    {'char': "'", 'unicode': '\\u0027'},
    {'char': '"', 'unicode': '\\u0022'},
    {'char': '<', 'unicode': '\\u003C'},
    {'char': '>', 'unicode': '\\u003E'},
    {'char': '&', 'unicode': '\\u0026'},
    {'char': '/', 'unicode': '\\u002F'},
    {'char': '(', 'unicode': '\\u0028'},
    {'char': ')', 'unicode': '\\u0029'},
]

# Build replacement map
replace_map = {}
for ent in html_fingerprint_entities:
    for key in ['named', 'decimal', 'hex']:
        if ent.get(key):
            replace_map[ent[key]] = ent['char']
for ent in unicode_escape_entities:
    replace_map[ent['unicode']] = ent['char']

def normalize_message(msg):
    if not isinstance(msg, str):
        return msg
    for k, v in replace_map.items():
        msg = msg.replace(k, v)
    return msg

# Collect all .json files in current directory and immediate subdirectories only
json_files = []
for subdir in ['green', 'yellow', 'red']:
    subdir_path = os.path.join(BASE_DIR, subdir)
    if os.path.exists(subdir_path):
        subdir_files = glob.glob(os.path.join(subdir_path, '*.json'))
        json_files.extend(subdir_files)
        print(f"Found {len(subdir_files)} files in {subdir}: {[os.path.basename(f) for f in subdir_files]}")

# Filter out unwanted files
json_files = [
    f for f in json_files
    if os.path.basename(f) not in {'combined.json', '.DS_Store'}
]

combined = []

for file_path in json_files:
    parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue
        if not isinstance(data, list):
            print(f"Skipping {file_path}: not a list of dicts")
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            user_message = entry.get('user_message')
            user_message = normalize_message(user_message)
            # Always set classification based on parent_dir if in red/yellow/green
            if parent_dir == 'green':
                norm_class = '0'
            elif parent_dir == 'yellow':
                norm_class = '1'
            elif parent_dir == 'red':
                norm_class = '2'
            else:
                classification = entry.get('classification', '')
                cls = str(classification).lower()
                if 'green' in cls:
                    norm_class = '0'
                elif 'yellow' in cls:
                    norm_class = '1'
                else:
                    norm_class = '2'
            combined.append({
                'user_message': user_message,
                'classification': norm_class
            })

# De-duplication methodology:
# To ensure the dataset is free from repeated entries, we remove duplicates after combining all data.
# Duplicates are defined as entries with the same 'user_message' and 'classification'.
# This step is important for data quality, preventing model bias and overfitting to repeated prompts.

exact_seen = set()
finger_seen = set()
deduped = []

# Helper for fuzzy fingerprint (letters + numbers only, lower-cased)
def fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())

for entry in combined:
    # First-pass: exact user_message+class duplication (existing logic)
    exact_key = (entry['user_message'], entry['classification'])
    if exact_key in exact_seen:
        continue

    # Second-pass: similarity fingerprint – collapses minor punctuation / spacing differences
    fp = fingerprint(entry['user_message'])
    fp_key = (fp, entry['classification'])
    if fp_key in finger_seen:
        continue

    # Record keys
    exact_seen.add(exact_key)
    finger_seen.add(fp_key)
    deduped.append(entry)

# Random sampling methodology:
# To create a balanced dataset for training, we randomly sample specific quantities from each classification:
# - Classification "0" (green): 5,000 entries
# - Classification "1" (yellow): 5,000 entries  
# - Classification "2" (red): 10,000 entries
# This ensures proper class balance while maintaining randomization for unbiased model training.

# Separate entries by classification
class_0 = [entry for entry in deduped if entry['classification'] == '0']
class_1 = [entry for entry in deduped if entry['classification'] == '1']
class_2 = [entry for entry in deduped if entry['classification'] == '2']

print(f"Available entries: Class 0: {len(class_0)}, Class 1: {len(class_1)}, Class 2: {len(class_2)}")

# Set random seed for reproducibility
random.seed(42)

# Randomly sample from each classification
sampled_0 = random.sample(class_0, min(10010, len(class_0)))
sampled_1 = random.sample(class_1, min(10010, len(class_1)))
sampled_2 = random.sample(class_2, min(10010, len(class_2)))

# Combine sampled data
final_dataset = sampled_0 + sampled_1 + sampled_2

# Shuffle the final dataset to mix classifications
random.shuffle(final_dataset)

print(f"Final dataset: {len(sampled_0)} class 0, {len(sampled_1)} class 1, {len(sampled_2)} class 2 = {len(final_dataset)} total entries")

# Write deduplicated output
output_path = os.path.join(BASE_DIR, 'combined_dataset.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

print(f"Combined {len(combined)} entries from {len(json_files)} files. {len(deduped)} unique entries after deduplication. {len(final_dataset)} entries in final sampled dataset written to {output_path}.") 