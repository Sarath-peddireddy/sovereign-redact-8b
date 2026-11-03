# SovereignRedact-8B: Enterprise PII Data Firewall

**An on-premise, fine-tuned Llama-3-8B firewall for deterministic PII redaction and enterprise data sovereignty.**

## The Problem
Regulated industries (Banking, Healthcare, Legal) face a zero-trust compliance bottleneck. They are actively blocked from utilizing frontier cloud models (GPT-4, Claude 3.5) because transmitting raw Personally Identifiable Information (PII) or Protected Health Information (PHI) over external APIs violates SOC2, HIPAA, and GDPR data sovereignty mandates. 

## The Impact
By placing a highly compressed, instruction-tuned 8B model locally within the enterprise firewall, companies can deterministically sanitize input streams *before* routing the safe payloads to third-party LLM providers. This enables full adoption of generative AI workflows while maintaining a 100% local data perimeter.

## The Solution
A parameter-efficient fine-tune (PEFT) of Meta's Llama-3.1-8B-Instruct. The model was trained specifically as a deterministic Named Entity Recognition (NER) router to map raw PII to secure tags (e.g., mapping a name to `[FIRSTNAME] [LASTNAME]`). 

* **Zero-Hallucination Schema:** Enforced via a rigid Alpaca prompt structure and hardcoded End-Of-Sequence (`EOS_TOKEN`) injection.
* **Hardware Efficient:** Utilizes 4-bit quantization and `adamw_8bit` optimization, reducing the VRAM footprint to ~6GB. The pipeline runs entirely on a free-tier Nvidia Tesla T4.

## Technical Architecture
* **Base Model:** `unsloth/Llama-3.1-8B-Instruct-bnb-4bit`
* **Compute Engine:** Unsloth (2x faster inference, highly optimized QLoRA)
* **Dataset:** 1,000-row deterministic mapping slice from `ai4privacy/pii-masking-200k`
* **Training Time:** 60 steps (<10 minutes on T4 GPU)

## Product Manager Friction Log (Platform Opportunities)
Building this MVP surfaced several Developer Experience (DX) bottlenecks in the open-source fine-tuning ecosystem. These are clear whitespace opportunities for enterprise AI platforms to abstract away friction:

1. **Schema Drift:** Dataset maintainers frequently alter column headers without version bumps (e.g., changing `masked_text` to `target_text`). Hardcoded training pipelines instantly break. Platforms require visual data-mapping UIs.
2. **Syntax Strictness & Infinite Loops:** Failing to append an explicit `EOS_TOKEN` to the dataset array causes the model to hallucinate infinitely during inference. Prompt-schema logic must be entirely abstracted from the user.
3. **VRAM Anxiety:** Manually calculating LoRA rank sizes, gradient accumulation steps, and optimizer compatibility to avoid Out-Of-Memory (OOM) fatal errors is a massive barrier to entry. Platforms must auto-provision these hyperparameters based on hardware profiles.

## How to Run the Pipeline
1. Clone this repository and open `PII_Redactor_Training.ipynb` in Google Colab.
2. Ensure you are using a **T4 GPU** runtime.
3. Run the **Environment Initialization** cell to install Unsloth and mount Google Drive for persistent checkpoint storage.
4. Execute the cells sequentially to attach the LoRA adapters, format the dataset, and run the 60-step gradient descent.
5. (Optional) Skip the training cell and jump directly to **Inference** to load saved adapter weights from your Drive and test real-time text redaction.
