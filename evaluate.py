import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm


def load_image_level_results(json_path, num_normal_prompts):
    with open(json_path, 'r') as f:
        results = json.load(f)

    y_true = []
    y_score = []

    for category in results:
        for defect_type in results[category]:
            label = 0 if defect_type == "good" else 1
            for item in results[category][defect_type]:
                scores = item["score"]
                # anomaly_score = max(scores[num_normal_prompts:])  # max of anomaly prompt scores
                anomaly_score = scores
                y_true.append(label)
                y_score.append(anomaly_score)

    return np.array(y_true), np.array(y_score)

def load_pixelwise_scores_and_masks(pred_dir, gt_dir, obj_cats):
    y_true_all = []
    y_score_all = []

    for category in tqdm(obj_cats):
        pred_category_dir = pred_dir / "segmentation" / category
        gt_category_dir = gt_dir / category / "ground_truth"
        for defect_dir in pred_category_dir.iterdir():
            defect_name = defect_dir.name
            for pred_path in defect_dir.glob("*.png"):
                index = pred_path.stem  # e.g., '000'

                gt_path = gt_category_dir / defect_name / f"{index}_mask.png"

                # Load prediction
                pred = np.array(Image.open(pred_path)).astype(np.float32)
                pred = pred / 255.0
                pred = pred.flatten()

                # Load GT or fallback to zero mask
                if gt_path.exists():
                    gt = np.array(Image.open(gt_path).resize((448, 448)))
                    gt = (gt > 127).astype(np.uint8).flatten()
                else:
                    gt = np.zeros(pred.size, dtype=np.uint8)
                assert gt.shape == pred.shape, f"Shape mismatch at {pred_path}: gt {gt.shape}, pred {pred.shape}"
                y_true_all.append(gt)
                y_score_all.append(pred)

    return np.concatenate(y_true_all), np.concatenate(y_score_all)



def evaluate(image_json_path, heatmap_root, mvtec_root, method, obj_cats, num_normal_prompts):
    print(f"Evaluating method: {method}")

    print("→ Image-level AUROC")
    y_true_img, y_score_img = load_image_level_results(image_json_path, num_normal_prompts)
    img_auroc = roc_auc_score(y_true_img, y_score_img)
    print(f"Image-level AUROC: {img_auroc:.4f}")

    print("→ Pixel-level AUROC")
    y_true_pix, y_score_pix = load_pixelwise_scores_and_masks(heatmap_root, mvtec_root, obj_cats)
    pix_auroc = roc_auc_score(y_true_pix, y_score_pix)
    print(f"Pixel-level AUROC: {pix_auroc:.4f}")

    return img_auroc, pix_auroc


if __name__ == "__main__":
    method = "clip"  # "gem" or "clip"
    num_normal_prompts = 1  # adjust based on your json file structure
    image_size = 448

    base_dir = Path.cwd()
    predictions_dir = base_dir / "predictions"
    mvtec_dir = base_dir / "mvtec"

    image_json_path = predictions_dir / "json" / f"results_{method}.json"
    heatmap_root = predictions_dir
    mvtec_root = mvtec_dir

    obj_cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
                'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    evaluate(image_json_path, heatmap_root, mvtec_root, method, obj_cats, num_normal_prompts)
