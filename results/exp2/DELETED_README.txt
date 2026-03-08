# Deleted Cache File (2026-02-25)

# DELETED:
# - activations_cache.pkl (11GB)
#     Contains extracted residual stream activations from Llama-3.2-3B-Instruct
#     across all 8 ICL tasks x 50 test inputs x 28 layers.
#     Structure: {"activations": tensor, "labels": tensor}
#
# TO REGENERATE (~30 minutes):
#     cd /home/bcheng/iclproject
#     python scripts/02_localization.py
#     This will re-extract activations from the HuggingFace model and recreate the cache.
#
# NOTE: Results are NOT affected — all metrics are in:
#     results/exp2/localization_results.json
#     results/exp2/probe_accuracy.csv
