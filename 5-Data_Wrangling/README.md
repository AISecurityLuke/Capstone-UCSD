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
- **Classification "0" (Green/Safe)**: 4,000 randomly selected entries
- **Classification "1" (Yellow/Morally Gray)**: 4,000 randomly selected entries  
- **Classification "2" (Red/Harmful)**: 8,000 randomly selected entries
- **Total**: 16,000 entries with reproducible randomization (seed=42)

**Current Dataset:**
- **Combined Dataset**: 16,000 entries in `combined.json`

**File Structure:**
- `combined.json`: Final balanced dataset with random sampling (16,000 entries)
- `combine_jsons.py`: Script for combining and sampling datasets
- `process_json.py`: Script for normalizing and cleaning data
- `yellow/yel_copy1.json` through `yellow/yel_copy50.json`: Individual morally gray prompt files
- `yellow/YELclaude_set1.json`, `yellow/YELclaude_set2.json`: Dataset organization files
- `green/GRE_Human_5387.json`, `green/GREchatgpt.json`: Cleaned green/safe prompts
- `red/hackaprompt.json`, `red/inject.json`, `red/red.json`: Red/harmful prompts (cleaned of repetitive patterns)