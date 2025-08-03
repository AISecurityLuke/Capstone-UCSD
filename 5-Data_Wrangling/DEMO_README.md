# Data Processing Pipeline Demo

This directory contains a comprehensive demonstration of the data processing pipeline used to clean and prepare the AI safety dataset. The demo shows all the steps that were originally implemented in `combine_jsons.py` and `process_json.py`.

## Files Created

- **`data_processing_demo.ipynb`**: Interactive Jupyter notebook with step-by-step visualization
- **`combined_demo.json`**: Output dataset created by the demo (16,000 entries)
- **`DEMO_README.md`**: This documentation file

## How to Run the Demo

### Interactive Jupyter Notebook
```bash
jupyter notebook data_processing_demo.ipynb
```
## Data Processing Pipeline Overview

The demo replicates the complete data processing workflow used to create the final `combined.json` dataset:

### Step 1: Initial Data Cleaning (ChatGPT o3-Pro)
- **What was done**: Raw data was initially processed using ChatGPT o3-Pro through prompt engineering
- **Purpose**: Cleaned chat exports, normalized data formats, and handled various content types
- **Result**: Structured JSON files in green/yellow/red directories

### Step 2: Text Normalization
- **HTML Entities**: Replaced `&apos;`, `&quot;`, `&lt;`, `&gt;`, `&amp;`, etc. with regular characters
- **Unicode Escapes**: Normalized `\u0027`, `\u0022`, `\u003C`, `\u003E`, etc.
- **Content Types**: Handled audio transcription, image pointers, and other media types

### Step 3: Data Deduplication
- **Method**: Removed entries with identical `user_message` and `classification`
- **Result**: Eliminated 52.6% of duplicate entries (21,644 out of 41,151)
- **Purpose**: Prevented model bias and overfitting to repeated prompts

### Step 4: Balanced Random Sampling
- **Target Distribution**:
  - Class 0 (Green/Safe): 4,000 entries
  - Class 1 (Yellow/Morally Gray): 4,000 entries
  - Class 2 (Red/Harmful): 8,000 entries
- **Method**: Reproducible random sampling with seed=42
- **Total**: 16,000 entries for optimal training

## Key Statistics from Demo Run

### Input Data
- **Total Files**: 58 JSON files across 3 categories
- **Raw Entries**: 41,151 total entries
- **File Distribution**:
  - Green: 2 files (9,541 entries)
  - Yellow: 53 files (6,017 entries)
  - Red: 3 files (25,593 entries)

### Processing Results
- **Deduplication**: Removed 21,644 duplicates (52.6% reduction)
- **Final Dataset**: 16,000 balanced entries
- **Class Balance**: 25% Green, 25% Yellow, 50% Red

### Data Quality Metrics
- **Text Length**: Average 91.6 words, range 1-12,315 words
- **Length Distribution**:
  - Short (≤25 words): 30.6%
  - Medium (26-80 words): 46.2%
  - Long (>80 words): 23.3%
- **Target Compliance**: 85.4% within 8-322 word range
- **Quality**: 0 empty messages, 0 duplicates, minimal HTML/Unicode artifacts

## Comparison with Original Scripts

### `process_json.py` Functions Replicated
- `clean_user_message()`: Handles different content types (audio, images, text)
- Text extraction from complex data structures
- Filtering of empty or non-text content

### `combine_jsons.py` Functions Replicated
- `normalize_message()`: HTML entity and Unicode escape normalization
- File discovery and categorization by directory
- Classification assignment based on parent directory
- Deduplication logic
- Balanced random sampling with reproducible seed

## Key Improvements in Demo

1. **Visualization**: Added charts and graphs to show data distributions
2. **Detailed Statistics**: Comprehensive analysis of text lengths and quality metrics
3. **Step-by-Step Output**: Clear progress indicators and intermediate results
4. **Error Handling**: Robust processing with detailed error reporting
5. **Documentation**: Inline comments explaining each processing step

## Usage Notes

- The demo creates `combined_demo.json` to avoid overwriting the original `combined.json`
- All processing uses the same random seed (42) for reproducibility
- The demo can be run multiple times to verify consistent results
- Sample entries are displayed to show the quality of processed data

## Data Sources

The demo processes the same data sources as the original pipeline:

- **Green/Safe**: `GRE_Human_5387.json`, `GREchatgpt.json`
- **Yellow/Morally Gray**: 50 files (`yel_copy1.json` through `yel_copy50.json`)
- **Red/Harmful**: `hackaprompt.json`, `inject.json`, `red.json`

## Output Format

The final dataset contains entries with this structure:
```json
{
  "user_message": "normalized and cleaned text content",
  "classification": "0"  // 0=Green, 1=Yellow, 2=Red
}
```

This demo provides a complete, reproducible example of how the AI safety dataset was processed and prepared for model training. 