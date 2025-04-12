import os
import random
import shutil
import argparse

def get_pairs(images_dir, masks_dir, ext=".png"):
    """
    Find image files in images_dir and return a list of (image_path, mask_path) pairs.
    Assumes that mask filename matches the image basename with the same extension.
    """
    pairs = []
    
    # Create a lookup for masks by name (without extension)
    mask_lookup = {}
    for file in os.listdir(masks_dir):
        name, file_ext = os.path.splitext(file)
        if file_ext.lower() == ext:
            mask_lookup[name] = os.path.join(masks_dir, file)
    
    # Process images that have the specified extension
    for file in os.listdir(images_dir):
        name, file_ext = os.path.splitext(file)
        if file_ext.lower() == ext:
            image_path = os.path.join(images_dir, file)
            mask_path = mask_lookup.get(name)
            if mask_path:
                pairs.append((image_path, mask_path))
            else:
                print(f"Warning: No mask found for image {file}")
    return pairs

def copy_files(pairs, dest_images_dir, dest_masks_dir):
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_masks_dir, exist_ok=True)
    for image_path, mask_path in pairs:
        shutil.copy2(image_path, dest_images_dir)
        shutil.copy2(mask_path, dest_masks_dir)

def main(args):
    random.seed(args.seed)

    # Get all image-mask pairs from the training directories
    pairs = get_pairs(args.images_dir, args.masks_dir, ext=".png")
    if not pairs:
        print("No matching image/mask pairs found. Exiting.")
        return

    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Found {len(pairs)} pairs. {len(train_pairs)} for training, {len(val_pairs)} for validation.")

    # Copy training pairs
    train_images_dir = os.path.join(args.out_dir, "train", "images")
    train_masks_dir = os.path.join(args.out_dir, "train", "masks")
    copy_files(train_pairs, train_images_dir, train_masks_dir)

    # Copy validation pairs
    val_images_dir = os.path.join(args.out_dir, "val", "images")
    val_masks_dir = os.path.join(args.out_dir, "val", "masks")
    copy_files(val_pairs, val_images_dir, val_masks_dir)

    print("Segmentation dataset split completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split training images and masks into training and validation sets for semantic segmentation."
    )
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing training images (.png).")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory containing training masks (.png).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to store the split dataset.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of images to use for training (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    args = parser.parse_args()
    main(args)