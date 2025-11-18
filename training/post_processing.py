"""
post_processing.py
Compare MMSegmentation and SAM outputs using aggregated score.
"""

import json
import numpy as np

# -----------------------------
# Configurable weight
# -----------------------------
k = 0.7  # Weight for confidence score

# -----------------------------
# Load Predictions
# -----------------------------
mmseg_results_path = "mmseg_results.json"
sam_results_path = "sam_results.json"

with open(mmseg_results_path, 'r') as f:
    mmseg_results = json.load(f)

with open(sam_results_path, 'r') as f:
    sam_results = json.load(f)

# -----------------------------
# Compute Aggregated Score
# -----------------------------
def compute_aggregated_score(confidence, iou, k=0.7):
    return k * confidence + iou

# -----------------------------
# Compare Models
# -----------------------------
comparison_results = []
for img_mmseg, img_sam in zip(mmseg_results['images'], sam_results['images']):
    mmseg_conf = img_mmseg['annotations'][0]['confidence_score']
    sam_conf = img_sam['annotations'][0]['confidence_score']
    mmseg_iou = np.random.uniform(0.5, 0.9)  # Replace with actual IoU computation
    sam_iou = np.random.uniform(0.5, 0.9)

    mmseg_score = compute_aggregated_score(mmseg_conf, mmseg_iou, k)
    sam_score = compute_aggregated_score(sam_conf, sam_iou, k)

    best_model = "MMSegmentation" if mmseg_score > sam_score else "SAM"
    comparison_results.append({
        "image": img_mmseg['file_name'],
        "mmseg_score": mmseg_score,
        "sam_score": sam_score,
        "best_model": best_model
    })

# Print and save results
print("Comparison Results:")
for res in comparison_results:
    print(res)

with open("comparison_results.json", 'w') as f:
    json.dump(comparison_results, f, indent=4)
