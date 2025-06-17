---
title: "Research Survey Slide Deck"
author: "Luke Johnson"
date: "2025-06-17"
description: "Slides summarising existing research, datasets, and reproduced results for Capstone."
---

### Link
LinkHere

---
Slide1

Chatbot Filtration System: triages prompts as Safe, Suspicious, or Malicious before RAG processing; architecture balances accuracy, adversarial robustness, and 24 GB RAM budget.
---
Slide2

Markov et al. 2023: Holistic moderation pipeline, multi-label transformer, active-learning refresh; F1≈0.90-0.95 across hate, violence, self-harm.
Link: https://arxiv.org/abs/2208.03274
---
Slide3

Kim et al. 2023: Adversarial Prompt Shield; DistilBERT fine-tuned on BAND examples reduces jailbreak success by 60 % with <10 ms latency.
Link: https://arxiv.org/abs/2311.00172
---
Slide4

GitHub – llm-security-prompt-injection: compares TF-IDF+LogReg, SVM, fine-tuned XLM-R (Acc 96-97 %); reproducible notebooks included.
Link: https://github.com/sinanw/llm-security-prompt-injection
---
Slide5

GitHub – malicious-prompt-detection: 467 k prompts, OpenAI embeddings + Random Forest/XGBoost (F1 ≈ 0.87); modular ETL and ROC plots.
Link: https://github.com/AhsanAyub/malicious-prompt-detection
---
Slide6

Dataset – harmful_behaviors: 520 illicit-request prompts for hard-reject class; suitable for Malicious label training.
Link: https://huggingface.co/datasets/labortap/harmful_behaviors
---
Slide7

Dataset – prompt-injections: 662 crafted injection prompts spanning English/German; standard test split for injection detection.
Link: https://huggingface.co/datasets/deepset/prompt_injections
---
Slide8

Dataset – hackaprompt: 602 000 adversarial prompts and completions from HackAPrompt jailbreak competition; stress-tests robustness at scale.
Link: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
---
Slide9

Reproduced Notebooks: Forked and reran key experiments, confirming baseline metrics and observing effects of BAND augmentation.
---
Slide10

Key Insights: layered taxonomy, adversarial training, and explainability underpin a resource-efficient, transparent safeguard; sets F1 baseline ≥ 0.96.