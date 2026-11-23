# Text-to-SQL with QLoRA: fine-tuning vs RAG on a 3B model

Fine-tuned Qwen2.5-3B-Instruct with QLoRA to turn natural-language questions plus a table schema into SQL, and compared it against the base model and a retrieval (RAG) baseline on the same held-out test set.

**Result:** fine-tuning raised accuracy from 49% to 65.5% (strict exact match) and from 62% to 75.5% (relaxed match), beating a few-shot RAG baseline by about 9 points on both metrics.

![demo](demo.png)

## Results (200 held-out test examples)

| Setup | Strict match | Relaxed match |
|---|---|---|
| Base model (zero-shot) | 49.0% | 62.0% |
| Base + RAG (top-3 similar examples) | 56.0% | 66.5% |
| Fine-tuned (QLoRA) | 65.5% | 75.5% |
| Fine-tuned + RAG | 68.0% | 77.0% |

- **Strict:** normalized exact match (lowercased, unified quotes, collapsed whitespace).
- **Relaxed:** strict, plus numbers in quotes treated as numbers, and the order of AND conditions ignored.
- Fine-tuned + RAG beat fine-tuned alone by 1.5 to 2.5 points, which is only 3 to 5 examples out of 200, so I treat it as no clear benefit.

## Setup

- **Data:** `b-mc2/sql-create-context`. 5,000 training examples and 200 held-out test examples, split with a fixed seed.
- **Model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`, trained on a free Colab T4.
- **QLoRA:** rank 16, alpha 16, all attention and MLP projection layers, learning rate 2e-4, 1 epoch with sequence packing (33 optimizer steps, final training loss about 0.67).
- **RAG baseline:** the 3 most similar training examples (all-MiniLM-L6-v2 embeddings, FAISS) are shown to the model as solved examples before the real question.
- **Same prompt, same test set, same metric** for all four setups.

## Failure analysis

I hand-read 15 of the 69 strict-metric failures of the fine-tuned model:

- 5 were aggregation inconsistencies: the dataset labels "how many" questions as COUNT in some rows and SUM in others, which caps achievable accuracy.
- 4 were correct SQL marked wrong (different condition order, `round = 6` vs `round = "6"`, a subquery instead of a JOIN).
- 4 were genuine model errors (an aggregate inside WHERE, a wrong selected column, an altered string literal).
- 2 were corrupted or odd labels.

This is a small sample, but it suggests the strict numbers understate true accuracy, which is why I report both metrics.

## Limitations

- Scoring is string matching, not execution accuracy, because this dataset has no databases to run queries against. Spider-style execution evaluation would be a stronger test.
- 200 test examples and a single training run, so differences of a few points are within noise.
- Trained for one epoch on a 5k subset. I did not run a rank ablation or a forgetting check.

## Links

- Model adapter: https://huggingface.co/Nive45/qwen2.5-3b-text2sql-qlora
- Notebook: in this repo
- Raw predictions for every setup: the `*_results.json` files

## Stack

Python, Hugging Face Transformers, PEFT, TRL, Unsloth, sentence-transformers, FAISS, Gradio.
