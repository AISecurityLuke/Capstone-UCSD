# Dataset Overview

This directory contains the primary datasets used for model training and evaluation.

**Data Sources:**
- Data was generated or curated using large language models (LLMs), with contributions from multiple user accounts. The file "GRE_Human_5387.json" was compiled from exported chat sessions across three distinct users.

**Data Processing:**
- Initial cleaning and normalization were performed using ChatGPT (o3-Pro), primarily through prompt engineering and chat exports, with minimal manual coding required.
- Datasets were further processed and combined into a single, deduplicated file, `combined.json`, using the provided scripts:
  - `process_json.py`: Normalizes and cleans user prompt data.
  - `combine_jsons.py`: Merges datasets, standardizes Unicode and HTML-escaped characters, and removes duplicate entries (same `user_message` and `classification`).