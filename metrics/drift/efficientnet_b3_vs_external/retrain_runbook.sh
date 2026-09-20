# Drift detected on tag='efficientnet_b3_vs_external' -- retrain recommended.
# Run this on cuda16. Check GPU availability first (this project's
# ECS boxes have no job scheduler -- pick whichever GPU has the most
# free memory, not just lowest utilisation):
#   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

cd ~/aiml589-wildlife-monitoring
git pull
source .venv/bin/activate

# Replace <GPU_ID> with the GPU index chosen above.
CUDA_VISIBLE_DEVICES=<GPU_ID> python -m src.training.boxcrop_experiment \
  configs/model/efficientnet_b3.yaml \
  --manifest-path data/manifest_combined.csv \
  --tag efficientnet_b3_boxcrop_retrained

# After training finishes, evaluate the retrained checkpoint two ways:

# 1) Regression check -- did it hold up on the original site-based test set?
python -m src.training.evaluate efficientnet_b3 boxcrop \
  --checkpoint-path models/checkpoint_efficientnet_b3_boxcrop_retrained.pt --tag efficientnet_b3_boxcrop_retrained

# 2) The real question -- did it close the external-distribution gap?
#    --split test restricts this to the held-out slice of external data
#    (see combine_manifests.py) -- WITHOUT it, this would partly
#    evaluate on images the retrain just trained on.
python -m src.external_data.evaluate_external efficientnet_b3 efficientnet_b3_boxcrop_retrained \
  --checkpoint-path models/checkpoint_efficientnet_b3_boxcrop_retrained.pt --split test
