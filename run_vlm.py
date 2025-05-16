import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from einops import rearrange
import gem

# Type aliases
Tensor = torch.Tensor
Array = np.ndarray
PathLike = Union[str, Path]

def save_heatmap(heatmap: Array, save_path: PathLike) -> None:
    """Save a heatmap as an image file.
    
    Args:
        heatmap: The heatmap array to save
        save_path: Path where the heatmap will be saved
    """
    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap.squeeze(0)
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(save_path)


def load_json(json_path: PathLike) -> Dict:
    """Load and parse a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Parsed JSON content as a dictionary
    """
    with open(json_path, 'r') as file:
        return json.load(file)


def mean_top1p(distances: Tensor) -> float:
    """Calculate mean of top 1% distances.
    
    Args:
        distances: Tensor of distances
        
    Returns:
        Mean of top 1% distances
    """
    distances = distances.cpu().numpy().flatten()
    top_k = max(1, int(len(distances) * 0.01))
    return np.mean(sorted(distances, reverse=True)[:top_k])


def mean_simple_average(distances: Tensor) -> float:
    """Calculate simple mean of distances.
    
    Args:
        distances: Tensor of distances
        
    Returns:
        Mean of all distances
    """
    return distances.cpu().numpy().mean()


def textual_visual_feature_matching(
    query_image_feat: Tensor,
    text_feat: Tensor
) -> Tensor:
    """Compute image-text matching scores.
    
    Args:
        query_image_feat: Image features
        text_feat: Text features
        
    Returns:
        Image-text matching scores
    """
    query_image_feat = F.normalize(query_image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    return query_image_feat[:, 1:] @ text_feat.transpose(-1, -2)


def similarity_scores(
    image: Tensor,
    text: str,
    model: torch.nn.Module,
    method: str,
    image_size: int = 448,
    patch_size: int = 32
) -> Tuple[float, Array]:
    """Compute similarity scores between image and text.
    
    Args:
        image: Input image tensor
        text: Text prompt
        model: Model to use for feature extraction
        method: Method to use ('gem' or 'clip')
        image_size: Size of input image
        patch_size: Size of image patches
        
    Returns:
        Tuple of (image similarity score, pixel-wise similarity map)
    """
    with torch.no_grad():
        gem_features, clip_features = model.model.visual(image)
        image_features = gem_features if method == "gem" else clip_features
        
        if method == "gem":
            text_features = model.encode_text([text]).squeeze(1)
        else:
            text_tokens = clip.tokenize(text).to(image.device)
            text_features = model.encode_text(text_tokens)

        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        
        pix_sim = image_features[:, 1:, :] @ text_features.squeeze(0).T
        pix_sim = rearrange(pix_sim, 'b (w h) c -> b c w h', 
                          w=image_size//patch_size, h=image_size//patch_size)
        pix_sim = F.interpolate(pix_sim, size=(image_size, image_size), 
                              mode='bilinear', align_corners=False)
        pix_sim = model.min_max(pix_sim)
        
        pooled_feat = image_features[:, 1:, :].mean(dim=1)
        img_sim = pooled_feat @ text_features.squeeze(0).T
        
        return img_sim.item(), pix_sim.squeeze(0).cpu().numpy()


def multiscale_similarity_scores(
    image: Tensor,
    text: str,
    model: torch.nn.Module,
    method: str,
    image_size: int = 448,
    patch_size: int = 32,
    tile_ratio: float = 0.5
) -> Tuple[float, Array, Tensor, Array, List[Array]]:
    """Compute multiscale similarity scores between image and text.
    
    Args:
        image: Input image tensor
        text: Text prompt
        model: Model to use for feature extraction
        method: Method to use ('gem' or 'clip')
        image_size: Size of input image
        patch_size: Size of image patches
        tile_ratio: Ratio of tile size to image size
        
    Returns:
        Tuple containing:
        - Combined image similarity score
        - Combined pixel-wise similarity map
        - Image tiles
        - Global pixel-wise similarity map
        - List of tile pixel-wise similarity maps
    """
    with torch.no_grad():
        _, _, H, W = image.shape
        tile_H, tile_W = int(H * tile_ratio), int(W * tile_ratio)

        # Extract tiles from corners
        tiles = []
        for i in [0, H - tile_H]:
            for j in [0, W - tile_W]:
                tile = image[..., i:i+tile_H, j:j+tile_W]
                tile = F.interpolate(tile, size=(image_size, image_size), 
                                  mode='bilinear', align_corners=False)
                tiles.append(tile)
        tiles = torch.cat(tiles, dim=0)

        # Process whole image
        global_img_sim, global_pix_sim = similarity_scores(
            image, text, model, method, image_size, patch_size
        )
        
        # Process tiles
        tile_img_sims = []
        tile_pix_sims = []
        for tile in tiles:
            tile = tile.unsqueeze(0)
            img_sim, pix_sim = similarity_scores(
                tile, text, model, method, image_size, patch_size
            )
            tile_img_sims.append(img_sim)
            tile_pix_sims.append(pix_sim)

        # Combine scores
        avg_tile_score = sum(tile_img_sims) / len(tile_img_sims)
        combined_img_sim = (avg_tile_score + global_img_sim) / 2

        # Combine heatmaps
        tile_pix_sims = [hmap.squeeze() if hmap.ndim == 3 else hmap 
                        for hmap in tile_pix_sims]
        
        # Build 2x2 layout
        top_row = np.concatenate([tile_pix_sims[0], tile_pix_sims[1]], axis=1)
        bottom_row = np.concatenate([tile_pix_sims[2], tile_pix_sims[3]], axis=1)
        tiled_map = np.concatenate([top_row, bottom_row], axis=0)

        # Resize to original size
        tiled_map_tensor = torch.tensor(tiled_map).unsqueeze(0).unsqueeze(0).float()
        combined_pix_sim = F.interpolate(
            tiled_map_tensor, 
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        )
        combined_pix_sim = combined_pix_sim.squeeze().cpu().numpy()
        combined_pix_sim = (combined_pix_sim + global_pix_sim.reshape(image_size, image_size)) / 2

        return combined_img_sim, combined_pix_sim, tiles, global_pix_sim, tile_pix_sims


def visualize_multiscale_tiles_and_heatmaps(
    image_tensor: Tensor,
    tiles: Tensor,
    heatmap_multiscale: Array,
    tile_heatmaps: List[Array],
    save_path_base: Optional[PathLike] = None
) -> None:
    """Visualize multiscale tiles and their corresponding heatmaps.
    
    Args:
        image_tensor: Original image tensor
        tiles: Image tiles tensor
        heatmap_multiscale: Combined multiscale heatmap
        tile_heatmaps: List of heatmaps for each tile
        save_path_base: Base path for saving visualizations
    """
    # Normalize image
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # Plot original image and tiles
    fig1, axs1 = plt.subplots(2, 3, figsize=(15, 10))
    axs1[0, 0].imshow(image_np)
    axs1[0, 0].set_title("Original Image")
    axs1[0, 0].axis('off')

    for idx, tile in enumerate(tiles):
        row, col = divmod(idx + 1, 3)
        tile_np = tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tile_np = (tile_np - tile_np.min()) / (tile_np.max() - tile_np.min())
        axs1[row, col].imshow(tile_np)
        axs1[row, col].set_title(f"Tile {idx + 1}")
        axs1[row, col].axis('off')

    axs1[1, 2].axis('off')

    if save_path_base:
        fig1.savefig(f"{save_path_base}_image_tiles.png", bbox_inches='tight')
    plt.close(fig1)

    # Plot heatmaps
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
    axs2[0, 0].imshow(heatmap_multiscale.squeeze(), cmap='jet')
    axs2[0, 0].set_title("Multiscale Heatmap")
    axs2[0, 0].axis('off')

    for idx, hmap in enumerate(tile_heatmaps):
        row, col = divmod(idx + 1, 3)
        axs2[row, col].imshow(hmap.squeeze(), cmap='jet')
        axs2[row, col].set_title(f"Tile {idx + 1} Heatmap")
        axs2[row, col].axis('off')

    axs2[1, 2].axis('off')

    if save_path_base:
        fig2.savefig(f"{save_path_base}_heatmap_tiles.png", bbox_inches='tight')
    plt.close(fig2)


def compute_textual_heatmap(
    image_tensor: Tensor,
    text: str,
    model: torch.nn.Module,
    method: str,
    image_size: int = 448,
    patch_size: int = 32
) -> Array:
    """Compute textual heatmap for anomaly detection.
    
    Args:
        image_tensor: Input image tensor
        text: Text prompt
        model: Model to use for feature extraction
        method: Method to use ('gem' or 'clip')
        image_size: Size of input image
        patch_size: Size of image patches
        
    Returns:
        Anomaly heatmap
    """
    with torch.no_grad():
        gem_features, clip_features = model.model.visual(image_tensor)
        image_features = gem_features if method == "gem" else clip_features
        image_features = F.normalize(image_features[:, 1:, :], dim=-1)

        if method == "gem":
            text_features = model.encode_text([text]).squeeze(1)
        else:
            text_tokens = clip.tokenize([text]).to(image_tensor.device)
            text_features = model.encode_text(text_tokens).squeeze(0)
        text_features = F.normalize(text_features, dim=-1)

        pix_sim = image_features @ text_features.T
        grid_size = image_size // patch_size
        pix_sim = rearrange(pix_sim, 'b (h w) c -> b c h w', h=grid_size, w=grid_size)
        pix_sim = F.interpolate(pix_sim, size=(image_size, image_size), 
                              mode='bilinear', align_corners=False)

        return (1 - pix_sim).squeeze().detach().cpu().numpy()


def normalize_heatmap(hmap: Array) -> Array:
    """Normalize heatmap values to [0, 1] range.
    
    Args:
        hmap: Input heatmap
        
    Returns:
        Normalized heatmap
    """
    hmin, hmax = hmap.min(), hmap.max()
    if hmax - hmin < 1e-6:
        return np.zeros_like(hmap)
    return (hmap - hmin) / (hmax - hmin)

def position_aware_pixelwise_heatmap(
    image_tensor: Tensor,
    memory_dict: Dict[str, Tensor],
    model: torch.nn.Module,
    method: str,
    text_prompt: Optional[str],
    image_size: int = 448,
    patch_size: int = 32,
    tile_ratio: float = 0.5
) -> Array:
    """Compute position-aware pixel-wise heatmap for anomaly detection.
    
    Args:
        image_tensor: Input image tensor
        memory_dict: Dictionary of memory features
        model: Model to use for feature extraction
        method: Method to use ('gem' or 'clip')
        text_prompt: Optional text prompt for textual guidance
        image_size: Size of input image
        patch_size: Size of image patches
        tile_ratio: Ratio of tile size to image size
        
    Returns:
        Combined anomaly heatmap
    """
    def extract_patch_features(tensor: Tensor) -> Tensor:
        gem_feat, clip_feat = model.model.visual(tensor)
        feats = gem_feat if method == "gem" else clip_feat
        feats = feats[:, 1:, :]
        return F.normalize(feats, dim=-1).squeeze(0)

    # Process whole image
    q_feats = extract_patch_features(image_tensor)
    m_feats = memory_dict["whole"].to(q_feats.device)
    dists = torch.norm(q_feats.unsqueeze(0) - m_feats, dim=-1)
    min_dists = dists.min(dim=0).values
    heatmap_whole = min_dists.reshape(image_size // patch_size, image_size // patch_size)
    heatmap_whole = F.interpolate(
        heatmap_whole.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    # Process tiles
    _, _, H, W = image_tensor.shape
    tile_H, tile_W = int(H * tile_ratio), int(W * tile_ratio)
    tile_coords = [(0, 0), (0, W - tile_W), (H - tile_H, 0), (H - tile_H, W - tile_W)]

    tile_canvas = torch.zeros((1, 1, image_size * 2, image_size * 2), 
                            device=image_tensor.device)

    for idx, (i, j) in enumerate(tile_coords):
        tile = image_tensor[..., i:i+tile_H, j:j+tile_W]
        tile = F.interpolate(tile, size=(image_size, image_size), 
                           mode='bilinear', align_corners=False)
        q_feats = extract_patch_features(tile)
        m_feats = memory_dict[f"tile_{idx}"].to(q_feats.device)
        dists = torch.norm(q_feats.unsqueeze(0) - m_feats, dim=-1)
        min_dists = dists.min(dim=0).values
        tile_map = min_dists.reshape(image_size // patch_size, image_size // patch_size)
        tile_map = F.interpolate(
            tile_map.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        )

        y_offset = 0 if i == 0 else image_size
        x_offset = 0 if j == 0 else image_size
        tile_canvas[:, :, y_offset:y_offset+image_size, x_offset:x_offset+image_size] = tile_map

    # Combine results
    tile_heatmap = F.interpolate(
        tile_canvas,
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    final_heatmap = 0.5 * (tile_heatmap + heatmap_whole)

    if text_prompt is not None:
        text_heatmap = compute_textual_heatmap(
            image_tensor, text_prompt, model, method, image_size, patch_size
        )
        final_heatmap = 0.5 * (final_heatmap.detach().cpu().numpy() + text_heatmap)

    return normalize_heatmap(final_heatmap)

def run_vlm(
    model: torch.nn.Module,
    preprocess: callable,
    data_dir: PathLike,
    save_dir: PathLike,
    obj_cats: List[str],
    prompts: Dict,
    image_size: int,
    method: str
) -> None:
    """Run VLM-based anomaly detection.
    
    Args:
        model: Model to use for feature extraction
        preprocess: Image preprocessing function
        data_dir: Directory containing dataset
        save_dir: Directory to save results
        obj_cats: List of object categories
        prompts: Dictionary of prompts for each category
        image_size: Size of input images
        method: Method to use ('gem' or 'clip')
    """
    results = {}
    save_dir = Path(save_dir)
    data_dir = Path(data_dir)
    
    (save_dir / "json").mkdir(parents=True, exist_ok=True)
    (save_dir / "segmentation").mkdir(parents=True, exist_ok=True)

    for object_category in tqdm(obj_cats):
        test_dir = data_dir / object_category / "test"
        obj_classes = [d for d in os.listdir(test_dir)]
        print(f"Evaluating {object_category}")
        results[object_category] = {}

        for obj_class in obj_classes:
            obj_class_dir = test_dir / obj_class
            image_files = os.listdir(obj_class_dir)
            print(obj_class)
            results[object_category][obj_class] = []

            for i, image_file in enumerate(image_files):
                # Load and preprocess image
                image_path = obj_class_dir / image_file
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize((image_size, image_size))
                image_tensor = preprocess(image).unsqueeze(0).to(next(model.parameters()).device)

                # Get prompts
                all_prompts = prompts[object_category]['normal'] + prompts[object_category]['anomaly']
                num_normal = len(prompts[object_category]['normal'])

                # Process each prompt
                img_scores = []
                pix_scores = []
                for t in all_prompts:
                    img_score, pix_score, tiles, global_heatmap, tile_heatmaps = multiscale_similarity_scores(
                        image_tensor, t, model, method, image_size=image_size, patch_size=32
                    )

                    # Save visualizations
                    index_str = f"{i:03d}"
                    vis_path_base = save_dir / "segmentation" / object_category / obj_class / index_str
                    os.makedirs(vis_path_base.parent, exist_ok=True)
                    visualize_multiscale_tiles_and_heatmaps(
                        image_tensor, tiles, pix_score, tile_heatmaps,
                        save_path_base=str(vis_path_base)
                    )

                    img_scores.append(img_score)
                    pix_scores.append(pix_score)

                # Save results
                img_scores_tensor = torch.tensor(img_scores)
                pred_idx = img_scores_tensor.argmax().item()
                pred_label = 'normal' if pred_idx < num_normal else 'anomaly'

                results[object_category][obj_class].append({
                    "file": image_file,
                    "pred": pred_label,
                    "scores": img_scores
                })

                # Save heatmap
                heatmap = pix_scores[num_normal]
                heatmap_path = save_dir / "segmentation" / object_category / obj_class / f"{i:03d}.png"
                os.makedirs(heatmap_path.parent, exist_ok=True)
                save_heatmap(heatmap, heatmap_path)

    # Save results
    with open(save_dir / "json" / f"results_{method}.json", 'w') as f:
        json.dump(results, f, indent=2)

def run_vlm_patchcore(
    model: torch.nn.Module,
    preprocess: callable,
    data_dir: PathLike,
    save_dir: PathLike,
    obj_cats: List[str],
    text_prompt: Dict,
    image_size: int,
    method: str,
    memory_banks: Dict[str, Dict[str, Tensor]]
) -> None:
    """Run VLM-PatchCore based anomaly detection.
    
    Args:
        model: Model to use for feature extraction
        preprocess: Image preprocessing function
        data_dir: Directory containing dataset
        save_dir: Directory to save results
        obj_cats: List of object categories
        text_prompt: Dictionary of prompts for each category
        image_size: Size of input images
        method: Method to use ('gem' or 'clip')
        memory_banks: Dictionary of memory banks for each category
    """
    save_dir = Path(save_dir)
    data_dir = Path(data_dir)
    
    (save_dir / "json").mkdir(parents=True, exist_ok=True)
    (save_dir / "segmentation").mkdir(parents=True, exist_ok=True)

    results = {}

    for object_category in tqdm(obj_cats):
        test_dir = data_dir / object_category / "test"
        obj_classes = sorted(os.listdir(test_dir))
        print(f"Evaluating {object_category}")
        results[object_category] = {}
        prompt = text_prompt[object_category]['anomaly'][0]

        for obj_class in obj_classes:
            obj_class_dir = test_dir / obj_class
            image_files = sorted(os.listdir(obj_class_dir))
            print(obj_class)
            results[object_category][obj_class] = []

            for i, image_file in enumerate(image_files):
                # Load and preprocess image
                image_path = obj_class_dir / image_file
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize((image_size, image_size))
                image_tensor = preprocess(image).unsqueeze(0).to(next(model.parameters()).device)

                # Compute heatmap
                heatmap = position_aware_pixelwise_heatmap(
                    image_tensor,
                    memory_banks[object_category],
                    model,
                    method,
                    prompt,
                    image_size=448,
                    patch_size=32,
                    tile_ratio=0.5
                )

                # Compute global score
                global_score = mean_top1p(torch.tensor(heatmap))

                # Save results
                results[object_category][obj_class].append({
                    "file": image_file,
                    "score": float(global_score)
                })

                # Save heatmap
                heatmap_path = save_dir / "segmentation" / object_category / obj_class / f"{i:03d}.png"
                os.makedirs(heatmap_path.parent, exist_ok=True)
                save_heatmap(heatmap, heatmap_path)

    # Save results
    with open(save_dir / "json" / f"results_patchcore_{method}.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set model parameters
    method = "clip"  # "gem" or "clip"
    model_name = 'ViT-B/32'
    pretrained = 'openai'
    
    # Initialize model
    preprocess = gem.get_gem_img_transform()
    model = gem.create_gem_model(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    
    # Set paths and parameters
    data_dir = Path.cwd() / "mvtec"
    save_dir = Path.cwd() / "predictions"
    image_size = 448
    obj_cats = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
        'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
        'transistor', 'wood', 'zipper'
    ]
    
    # Load prompts
    prompts_path = Path.cwd() / "configs" / "mvtec_vlm.json"
    prompts = load_json(prompts_path)
    
    # Run VLM
    run_vlm(model, preprocess, data_dir, save_dir, obj_cats, prompts, image_size, method)
    print("VLM evaluation complete!")
    
    # Load memory banks
    memory_banks = {
        cat: torch.load(Path.cwd() / "memory_banks" / f"{cat}.pt")
        for cat in obj_cats
    }
    
    # Run VLM-PatchCore
    run_vlm_patchcore(
        model=model,
        preprocess=preprocess,
        data_dir=data_dir,
        save_dir=save_dir,
        obj_cats=obj_cats,
        text_prompt=prompts,
        image_size=image_size,
        method=method,
        memory_banks=memory_banks
    )
    print("VLM-PatchCore evaluation complete!")
