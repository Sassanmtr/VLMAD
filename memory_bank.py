import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import gem
import torch.nn.functional as F

def build_all_memory_banks(data_dir, obj_cats, model, preprocess, method, image_size=448, patch_size=32, tile_ratio=0.5, num_samples=16):
    save_dir = Path.cwd() / "memory_banks"
    save_dir.mkdir(exist_ok=True)

    for category in tqdm(obj_cats, desc="Building memory banks"):
        train_dir = data_dir / category / "train" / "good"
        image_paths = sorted(list(train_dir.glob("*.png")))[:num_samples]

        mem_dict = {
            "whole": [],
            "tile_0": [],
            "tile_1": [],
            "tile_2": [],
            "tile_3": [],
        }

        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB').resize((image_size, image_size))
            img_tensor = preprocess(img).unsqueeze(0).to(next(model.parameters()).device)

            # Whole image
            gem_feat, clip_feat = model.model.visual(img_tensor)
            feats = gem_feat if method == "gem" else clip_feat
            feats = feats[:, 1:, :]  # remove CLS
            feats = F.normalize(feats, dim=-1)
            mem_dict["whole"].append(feats.squeeze(0).cpu())  # [196, D]

            # Tiles
            _, _, H, W = img_tensor.shape
            tile_H, tile_W = int(H * tile_ratio), int(W * tile_ratio)
            tile_coords = [(0, 0), (0, W - tile_W), (H - tile_H, 0), (H - tile_H, W - tile_W)]

            for idx, (i, j) in enumerate(tile_coords):
                tile = img_tensor[..., i:i+tile_H, j:j+tile_W]
                tile = F.interpolate(tile, size=(image_size, image_size), mode='bilinear', align_corners=False)
                gem_tile, clip_tile = model.model.visual(tile)
                tile_feats = gem_tile if method == "gem" else clip_tile
                tile_feats = tile_feats[:, 1:, :]
                tile_feats = F.normalize(tile_feats, dim=-1)
                mem_dict[f"tile_{idx}"].append(tile_feats.squeeze(0).cpu())  # [196, D]

        # Stack all image tensors into shape [16, 196, D]
        mem_dict = {k: torch.stack(v, dim=0) for k, v in mem_dict.items()}  # [B, 196, D]
        torch.save(mem_dict, save_dir / f"{category}.pt")
        print(f"Saved structured memory bank for {category} at {save_dir / f'{category}.pt'}")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = "clip"
    model = gem.create_gem_model(model_name='ViT-B/32', pretrained='openai', device=device)
    preprocess = gem.get_gem_img_transform()
    data_dir = Path.cwd() / "mvtec"
    # obj_cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    #             'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    obj_cats = ['wood', 'zipper']

    build_all_memory_banks(data_dir, obj_cats, model, preprocess, method)
    print("Memory banks saved.")
