I experimented with parameter-efficient fine-tuning techniques by training Gemma 3 1B IT on a custom Smoking Cessation / Motivational Coaching dataset. The goal was to build a lightweight domain-specific assistant while comparing the trade-offs between LoRA and QLoRA.

Setup :
* Model: Gemma 3 1B IT
* Dataset: ~1,500 smoking cessation conversations
* Infrastructure: AWS EC2 (Tesla T4 16GB)
* Frameworks: PyTorch, Hugging Face Transformers, PEFT
* Methods: LoRA & QLoRA (4-bit NF4 Quantization)
