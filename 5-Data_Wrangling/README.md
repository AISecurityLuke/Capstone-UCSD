# Dataset Overview

This directory contains the primary datasets used for model training and evaluation.

**Data Sources:**
- Data was generated or curated using large language models (LLMs), with contributions from multiple user accounts. The file "GRE_Human_5387.json" was compiled from exported chat sessions across three distinct users.
- **Morally Gray Dataset**: 50 files (`yel_copy1.json` through `yel_copy50.json`) containing 4,869 morally ambiguous prompts for AI safety research.

**Data Processing:**
- Initial cleaning and normalization were performed using ChatGPT (o3-Pro), primarily through prompt engineering and chat exports, with minimal manual coding required.
- Datasets were further processed and combined into a single, deduplicated file, `combined.json`, using the provided scripts:
  - `process_json.py`: Normalizes and cleans user prompt data.
  - `combine_jsons.py`: Merges datasets, standardizes Unicode and HTML-escaped characters, removes duplicate entries (same `user_message` and `classification`), and applies **random sampling** for balanced training.

**Random Sampling Methodology:**
The `combine_jsons.py` script implements balanced random sampling to create optimal training datasets:
- **Classification "0" (Green/Safe)**: 5,000 randomly selected entries
- **Classification "1" (Yellow/Morally Gray)**: 5,000 randomly selected entries  
- **Classification "2" (Red/Harmful)**: 10,000 randomly selected entries
- **Total**: 20,000 entries with reproducible randomization (seed=42)

**Current Dataset Statistics:**
- **Combined Dataset**: 20,000 entries in `combined.json`
- **Morally Gray Prompts**: 4,869 entries across 50 files
- **Word Length Range**: 8-346 words per prompt
- **Length Distribution**: 25.8% short (8-25 words), 52.7% medium (26-80 words), 21.5% long (81+ words)
- **Target Compliance**: 90.3% of prompts within 8-322 word target range

**File Structure:**
- `combined.json`: Final balanced dataset with random sampling
- `FINAL_DATA_combined.json`: Alternative combined dataset version
- `yellow/yel_copy1.json` through `yellow/yel_copy50.json`: Individual morally gray prompt files
- `yellow/YELclaude_set1.json`, `yellow/YELclaude_set2.json`: Dataset organization files