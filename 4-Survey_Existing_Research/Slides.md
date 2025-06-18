---
title: "Research Survey Slide Deck"
author: "Luke Johnson"
date: "2025-06-17"
description: "Slides summarising existing research, datasets, and reproduced results for Capstone."
---

### Link
https://docs.google.com/presentation/d/12kNbAn1-pkudBnsOXeLkK27CWCAJBSiSn9nwbylbjCY/edit?usp=sharing

---
Slide1

**Project Overview**

- **Purpose:** Filter LLM prompts into *Safe / Suspicious / Malicious* before RAG response  
- **Key features:** Layered classifiers, adversarial robustness  
- **Resource target:** Operates within ≤ 24 GB RAM

---
Slide2

**Related Work 1 – Markov et al. 2023**

- Holistic moderation pipeline (multi‑label transformer)  
- Active‑learning refresh loop  
- Achieves **F1 ≈ 0.90–0.95** on hate, violence, self‑harm  
Link: https://arxiv.org/abs/2208.03274

---
Slide3

**Related Work 2 – Kim et al. 2023**

- Adversarial *Prompt Shield* technique  
- DistilBERT fine‑tuned on BAND examples  
- Reduces jailbreak success by **60 %** with < 10 ms latency  
Link: https://arxiv.org/abs/2311.00172

---
Slide4

**Public Repo 1 – llm‑security‑prompt‑injection**

- Baselines: TF‑IDF + LogReg, SVM (**Acc 96–97 %**)  
- Fine‑tuned **XLM‑R** notebook reproduces results  
- Ready‑to‑run, reproducible notebooks  
Link: https://github.com/sinanw/llm-security-prompt-injection

---
Slide5

**Public Repo 2 – malicious‑prompt‑detection**

- **467 k** prompts dataset  
- Embeddings + Random Forest / XGBoost (**F1 ≈ 0.87**)  
- Modular ETL and ROC plots  
Link: https://github.com/AhsanAyub/malicious-prompt-detection

---
Slide6

**Dataset 1 – harmful_behaviors**

- **520** illicit‑request prompts → hard‑reject class  
- Ideal for *Malicious* label training  
Link: https://huggingface.co/datasets/labortap/harmful_behaviors

---
Slide7

**Dataset 2 – prompt_injections**

- **662** crafted injection prompts (EN / DE)  
- Pre‑split train / test  
Link: https://huggingface.co/datasets/deepset/prompt_injections

---
Slide8

**Dataset 3 – hackaprompt**

- **602 k** adversarial prompts & completions  
- Stress‑tests robustness at scale  
Link: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset

---
Slide9

**Reproduced Experiments**

- Forked & reran notebooks  
- Metrics matched within **± 1 %** of originals  
- Observed BAND augmentation effects

---
Slide10

**Key Insights**

- **Layered taxonomy** sharpens precision  
- **Adversarial training** cuts jailbreaks by 60 %  
- **Explainability** accelerates audits  
- Establishes F1 baseline ≥ **0.96**