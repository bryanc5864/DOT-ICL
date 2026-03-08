# Characterizing the Computational Interface of In-Context Learning

Code and experimental results for mechanistic interpretability research on how transformer language models encode and transfer task identity during in-context learning (ICL).

## Key Findings

1. **Task identity is distributed across demo output tokens.** Replacing activations at ALL demo output positions at ~30% network depth achieves up to 96% task transfer (N=50, 95% CI: [87%, 99%]). Single-position intervention yields 0%.

2. **The optimal intervention depth is ~30% across architectures.** This holds for LLaMA (1B, 3B), Qwen (1.5B), and Gemma (2B) — layer 8/28, 5/16, 8/28, and 8/26 respectively.

3. **Transfer is governed by output token-count templates.** Tasks with matching output token counts transfer at 50-100%; mismatched counts yield 0%. Variable-length outputs (e.g., "cat 3" -> "cat cat cat") transfer perfectly only when N matches the source output length.

4. **Abstract task templates are separable from specific labels.** Sentiment variants with different labels (positive/negative vs good/bad vs +/-) transfer at 58% mean, confirming the model encodes classification rules abstractly.

5. **Encoding requires a critical mass of demonstrations.** 1-3 source demos yield 0% transfer; 5 demos yield 93%. There is no gradual degradation — it is all-or-nothing.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bryanc5864/DOT-ICL.git
cd DOT-ICL
```

### 2. Create Conda Environment

```bash
conda create -n icl python=3.10 -y
conda activate icl
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.40.0
pip install numpy scipy scikit-learn
pip install huggingface_hub
```

Or install all at once:

```bash
pip install torch transformers numpy scipy scikit-learn huggingface_hub
```

### 4. HuggingFace Authentication (Required for Gated Models)

Some models (Llama, Gemma) require authentication:

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted. You must have accepted the model licenses on HuggingFace:
- [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Gemma-2-2B-IT](https://huggingface.co/google/gemma-2-2b-it)

Qwen models do not require authentication.

### 5. Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
python -c "from src.model import load_model; print('Project imports OK')"
```

---

## Hardware Requirements

| Experiment | GPU Memory | Recommended |
|------------|-----------|-------------|
| Single model (3B) | 8GB+ | RTX 3080, RTX 4080, A100 |
| Multi-model (all 4) | 8GB+ per GPU | Multiple GPUs recommended |
| Full reproduction | 16GB+ | A100 40GB or better |

All experiments use FP16 inference by default. CPU-only is not supported.

---

## Quick Start: Reproduce Key Results

### Core Finding: Multi-Position Transfer (Experiment 8)

```bash
python scripts/08_multi_position.py --device cuda:0 --n-test 20 --output-dir results/exp8
```

**Expected output:** `results/exp8/multi_position_results.json` with ~90% transfer rate for uppercase→repeat_word at layer 8.

### Full Reproduction (All Experiments)

```bash
# Phase 1: Baseline experiments (Exp 1-7)
python scripts/01_baseline.py --device cuda:0
python scripts/02_localization.py --device cuda:0
python scripts/03_transplant.py --device cuda:0
python scripts/04_interpolation.py --device cuda:0
python scripts/05_locality.py --device cuda:0
python scripts/06_ontology.py --device cuda:0
python scripts/07_trajectory.py --device cuda:0

# Phase 2: Core intervention experiments (Exp 8-15)
python scripts/08_multi_position.py --device cuda:0
python scripts/09_query_intervention.py --device cuda:0
python scripts/10_attention_intervention.py --device cuda:0
python scripts/11_activation_patching.py --device cuda:0
python scripts/12_layer_ablation.py --device cuda:0
python scripts/13_instance_analysis.py --device cuda:0
python scripts/14_demo_ablation.py --device cuda:0
python scripts/15_cross_format_control.py --device cuda:0

# Phase 3: Reviewer response experiments (Exp 19-33)
python scripts/19_template_similarity.py --device cuda:0
python scripts/23_proper_stats.py --device cuda:0
python scripts/27_baselines.py --device cuda:0
python scripts/28_tokenization_analysis.py --device cuda:0
python scripts/29_expanded_transfer.py --device cuda:0
python scripts/30_single_demo_fv.py --device cuda:0
python scripts/31_output_position_scaling.py --device cuda:0
python scripts/32_sentiment_variants.py --device cuda:0
python scripts/33_variable_length.py --device cuda:0
```

### Multi-Model Replication (Experiment 16)

Run on multiple GPUs in parallel:

```bash
# GPU 0: Llama-3.2-1B
python scripts/08_multi_position.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda:0 --output-dir results/llama-3.2-1b-instruct/exp8

# GPU 1: Qwen2.5-1.5B
python scripts/08_multi_position.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda:1 --output-dir results/qwen2.5-1.5b-instruct/exp8

# GPU 2: Gemma-2-2B
python scripts/08_multi_position.py --model google/gemma-2-2b-it --device cuda:2 --output-dir results/gemma-2-2b-it/exp8
```

---

## Experiments Summary

| Exp | Script | Description | Key Result |
|-----|--------|-------------|------------|
| 1 | `01_baseline.py` | Baseline ICL accuracy | 96-100% across 8 tasks |
| 2 | `02_localization.py` | Representation probing | Demo: 100%, Query: 83% peak |
| 3 | `03_transplant.py` | Single-position intervention | 0% transfer |
| 4 | `04_interpolation.py` | Activation interpolation | Smooth degradation |
| 5 | `05_locality.py` | Layer sweep | 0% at all layers |
| 6 | `06_ontology.py` | Task clustering | Procedural tasks cluster |
| 7 | `07_trajectory.py` | Representation trajectory | Early change, late stability |
| **8** | `08_multi_position.py` | **Multi-position transfer** | **90-96% at ~30% depth** |
| 9 | `09_query_intervention.py` | Query position intervention | 0% (necessary not sufficient) |
| 10 | `10_attention_intervention.py` | Attention knockout | Demo info processed by L16 |
| 11 | `11_activation_patching.py` | Causal tracing | Query necessary, demo not |
| 12 | `12_layer_ablation.py` | Layer ablation | Layer 0 critical |
| 13 | `13_instance_analysis.py` | Instance-level analysis | Format matching required |
| 14 | `14_demo_ablation.py` | Demo count ablation | Distribution fundamental |
| 15 | `15_cross_format_control.py` | Format variants | Structure drives transfer |
| 19 | `19_template_similarity.py` | Similarity metric | r=-0.05, not predictive |
| 23 | `23_proper_stats.py` | N=50 with CIs | 96% [87%, 99%] |
| 27 | `27_baselines.py` | Control baselines | All controls yield 0% |
| 28 | `28_tokenization_analysis.py` | Tokenization confounds | Not explanatory |
| 29 | `29_expanded_transfer.py` | Full 56-pair matrix | Cluster structure |
| 30 | `30_single_demo_fv.py` | Single-demo FV | 0% with <5 demos |
| 31 | `31_output_position_scaling.py` | Position scaling | ALL positions needed |
| 32 | `32_sentiment_variants.py` | Sentiment variants | 58% abstract transfer |
| 33 | `33_variable_length.py` | Variable-length tasks | Token-count matching |

---

## Repository Structure

```
DOT-ICL/
├── scripts/                    # Experiment scripts
│   ├── 01_baseline.py         # Baseline ICL accuracy
│   ├── 08_multi_position.py   # Core multi-position transfer
│   ├── ...                    # Experiments 02-33
│   └── compare_models.py      # Cross-model comparison
├── src/                       # Core library
│   ├── model.py              # Model loading & intervention hooks
│   ├── probing.py            # Linear probing utilities
│   ├── intervention.py       # Activation intervention
│   ├── extraction.py         # Activation extraction
│   └── tasks/                # Task definitions
│       ├── base.py           # Task ABC and registry
│       ├── string_tasks.py   # uppercase, reverse, pig_latin
│       ├── string_tasks_extra.py  # first_letter, repeat_word
│       ├── numeric_tasks.py  # length, linear_2x
│       ├── semantic_tasks.py # sentiment, antonym
│       └── pattern_tasks.py  # pattern_completion
├── results/                   # All experimental outputs
│   ├── MASTER_SUMMARY.md     # Complete results narrative
│   ├── exp{N}/               # Per-experiment results (JSON, CSV, logs)
│   ├── llama-3.2-1b-instruct/    # Cross-model results
│   ├── qwen2.5-1.5b-instruct/
│   ├── gemma-2-2b-it/
│   └── cross_model/          # Comparison CSVs
├── config/                    # Configuration files
├── tests/                     # Unit tests
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Models Tested

| Model | Parameters | Layers | Optimal Layer | HuggingFace ID |
|-------|-----------|--------|---------------|----------------|
| Llama-3.2-3B-Instruct | 3B | 28 | 8 (~29%) | `meta-llama/Llama-3.2-3B-Instruct` |
| Llama-3.2-1B-Instruct | 1B | 16 | 5 (~31%) | `meta-llama/Llama-3.2-1B-Instruct` |
| Qwen2.5-1.5B-Instruct | 1.5B | 28 | 8 (~29%) | `Qwen/Qwen2.5-1.5B-Instruct` |
| Gemma-2-2B-IT | 2B | 26 | 8 (~31%) | `google/gemma-2-2b-it` |

---

## Tasks

8 ICL tasks spanning procedural, semantic, and retrieval regimes:

| Task | Type | Input | Output | Example |
|------|------|-------|--------|---------|
| `uppercase` | Procedural | word | WORD | apple → APPLE |
| `first_letter` | Procedural | word | letter | apple → a |
| `repeat_word` | Procedural | word | word word | apple → apple apple |
| `length` | Procedural | word | number | apple → 5 |
| `linear_2x` | Procedural | number | number×2 | 7 → 14 |
| `sentiment` | Semantic | word | positive/negative | happy → positive |
| `antonym` | Retrieval | word | antonym | hot → cold |
| `pattern_completion` | Pattern | sequence | next element | A B A B → A |

Additional tasks for Experiment 33:
| Task | Type | Input | Output | Example |
|------|------|-------|--------|---------|
| `repeat_n` | Variable-length | word N | word × N | cat 3 → cat cat cat |
| `spell_out` | Variable-length | number | word | 7 → seven |

---

## Output Format

Each experiment produces:
- `exp{N}.log` — Execution log with timestamps
- `*_results.json` — Full results with metadata
- `*.csv` — Tabular results for analysis

Example JSON structure:
```json
{
  "metadata": {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "n_test": 20,
    "timestamp": "2026-02-01T12:00:00"
  },
  "results": {
    "uppercase_to_repeat_word": {
      "transfer_rate": 0.95,
      "preserve_rate": 0.05,
      "layer": 8
    }
  }
}
```

---

## Causal Flow Model

```
Input Processing (Layer 0)           [CRITICAL]
         │
Demo Output Tokens (Layers 1-8)     Store task identity (distributed)
         │
Attention Aggregation (Layers 8-12)  Demo info → Query position
         │
Query Position (Layers 12-16)       Task identity finalized
         │
Output Generation (Layers 16-28)    Output format applied
```

**Intervention window:** ~30% depth, ALL demo output positions.

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or use a smaller model:
```bash
python scripts/08_multi_position.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda:0
```

### HuggingFace Authentication Error
```bash
huggingface-cli login
# Or set token directly:
export HF_TOKEN=your_token_here
```

### Model Not Found
Ensure you've accepted the model license on HuggingFace and are authenticated.

### Import Errors
Run from the repository root:
```bash
cd DOT-ICL
python scripts/01_baseline.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{icl2026,
  title={Characterizing the Computational Interface of In-Context Learning},
  author={Zhang, Jasper and Cheng, Bryan},
  year={2026}
}
```

---

## Full Results

See [results/MASTER_SUMMARY.md](results/MASTER_SUMMARY.md) for the complete experimental narrative, all results tables, and detailed interpretation.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
