# PoliTo at SemEval-2025 Task 1: Beyond Literal Meaning: A Chain-of-Though Approach for Multimodal Idiomacity Understanding

Official repository of the paper: *"PoliTo at SemEval-2025 Task 1: Beyond Literal Meaning: A Chain-of-Though Approach for Multimodal Idiomacity Understanding"*

## 🏁 AdMIRe
Challenge website: [AdMIRe](https://semeval2025-task1.github.io/) </br>
Codabech website: [competition](https://www.codabench.org/competitions/4345/#/results-tab)

## 🔧 Code organization
Inside the folder *code*, you find:
- The script for the compound classification
- The scripts leverage CLIP
- The scripts leverage Gemini
- The scripts leverage Qwen

Scripts with the suffix **_TXT** are related to text-only experiments. </br>
The suffixes **_PLAIN**, **_CLS**, **_CoT**, **_FS** respectively represents plain setting, leveragin compound classification, Chain-Of-Thoughts and Few-Shot settings. </br>

The code has been tested only on English data. It can be adapted to any language by translating the prompts.

## 📌 Model
- Gemini 1.5 Flash: [Gemini API](https://ai.google.dev/)
- Gemini 2.0 Flash: [Gemini API](https://ai.google.dev/)
- Gemini 2.0 Flash Thinking: [Gemini API](https://ai.google.dev/) (Exp 01-21)
- Qwen 2.5 VL 7B: [Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- CLIP-large: [CLIP](https://huggingface.co/openai/clip-vit-large-patch14)

## 👤 Authors
- [Lorenzo Vaiani](mailto:lorenzo.vaiani@polito.it) - Politecnico di Torino
- [Davide Napolitano](mailto:davide.napolitano@polito.it) - Politecnico di Torino
- [Luca Cagliero](mailto:luca.cagliero@polito.it) - Politecnico di Torino </br>
For any questions, information, or if you want to extend our work by adding datasets, models, or metrics, please email us!

## 📖 References
- AdMIRe: [paper](https://arxiv.org/pdf/2503.15358)
- Gemini: [paper](https://arxiv.org/pdf/2312.11805)
- Qwen VL: [paper](https://arxiv.org/abs/2502.13923)

## ✍🏼 Citation
Citation
```bibtex
Accepted at SemEval 2025
```
