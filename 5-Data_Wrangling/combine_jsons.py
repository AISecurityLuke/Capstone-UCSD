import os
import json
import glob

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

# Recursively find all .json files (excluding combined.json and .DS_Store)
json_files = [
    f for f in glob.glob(os.path.join(BASE_DIR, '**', '*.json'), recursive=True)
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
            classification = entry.get('classification', '')
            # If classification is missing, infer from parent directory
            if not classification:
                if parent_dir in {'green', 'yellow', 'red'}:
                    norm_class = parent_dir
                else:
                    norm_class = 'red'  # default fallback
            else:
                cls = str(classification).lower()
                if 'green' in cls:
                    norm_class = 'green'
                elif 'yellow' in cls:
                    norm_class = 'yellow'
                else:
                    norm_class = 'red'
            combined.append({
                'user_message': user_message,
                'classification': norm_class
            })

# Write combined output
output_path = os.path.join(BASE_DIR, 'combined.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f"Combined {len(combined)} entries from {len(json_files)} files into {output_path}") 