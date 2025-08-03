---
title: "Project Proposal"
author: "Luke Johnson" (AISecurityLuke)
date: "2025‑06‑14"
---

# Chatbot Filtration System - Pseudocode

## 1. Receive Prompt via API

### API Endpoint: `/receive_prompt`

- Accepts user input (`prompt`)
- Input filter for `prompt`→ `prompt` dropped to `Current_Prompt.json`
- Forwards `prompt` to the classification pipeline

## 2. Classification Pipeline

### Step 1: Initialize Multi-Class Classification Model & Configurable Sensitivity

- Define `classification_model = Model()`
- Model returns scoring for Safe, Suspicious, and Malicious prompts.
- Configurable threshold to allow or deny **Suspicious** prompts

### Step 2: Pass Prompt to Classification Model

- `prompt_classification = classification_model.predict(prompt)`

### Step 3: Filter Decision (Green, Yellow, Red Classification)

- Classified as **Safe** → **PASS** → Directly forward to RAG model
- Classified as **Suspicious** → Check Sensitivity Level → PASS / REJECT Determination → 
	- If PASS, forward to RAG model. 
	- If REJECT, Deny with message, "This prompt has been determined to be malicious and will be recorded." drop rejected prompt to .json log file.
- Classified as **Malicious** → **REJECT** → Deny with message, "This prompt has been determined to be malicious and will be recorded." drop rejected prompt to `log.json` with time stamp.
## 3. RAG Model Responds for Prompts Receiving "PASS"
- Model response is returned via API response
## 4. Example Workflow

1. **User sends prompt**: "How do databases get hacked?"
   - Prompt is dropped to `Current_Prompt.json` with time stamp
2. **Classification model returns a score**:
   - Model Example Score: `0.2,0.7,0.5` (Safe,Suspicious,Malicious)
3. **Decision**:
   - If Suspicious = PASS → Forward to RAG model but log in `log.json` for security
   - If Suspicious = REJECT → DO NOT forward to RAG model and log in `log.json` for security
4. **Response to user**:
   - If **REJECTED**: "Unsafe request detected."
   - If **PASSED**: "Here's some safe and ethical guidance."
