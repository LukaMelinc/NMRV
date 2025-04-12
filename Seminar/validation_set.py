import os
import random
import shutil
import argparse

def get_pairs(images_dir, annotations_dir, image_exts={'.jpg', '.jpeg', '.png'}):
    """
    Find image files in images_dir and return a list of (image_path, annotation_path) pairs.
    Assumes that the annotation filename (with any extension) matches the image basename.
    """
    pairs = []
    # Create a lookup for annotations by name (without extension)
    annotation_lookup = {}
    for file in os.listdir(annotations_dir):
        name, ext = os.path.splitext(file)
        annotation_lookup[name] = os.path.join(annotations_dir, file)

    # Process images that have a valid extension
    for file in os.listdir(images_dir):
        name, ext = os.path.splitext(file)
        if ext.lower() in image_exts:
            image_file = os.path.join(images_dir, file)
            annotation_file = annotation_lookup.get(name)
            if annotation_file is not None:
                pairs.append((image_file, annotation_file))
            else:
                print(f"Warning: No annotation found for image {file}")
    return pairs

def copy_files(pairs, dest_images_dir, dest_annotations_dir):
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_annotations_dir, exist_ok=True)
    for image_path, annotation_path in pairs:
        shutil.copy2(image_path, dest_images_dir)
        shutil.copy2(annotation_path, dest_annotations_dir)

def main(args):
    random.seed(args.seed)

    # Get all pairs
    pairs = get_pairs(args.images_dir, args.annotations_dir)
    if not pairs:
        print("No matching image/annotation pairs found. Exiting.")
        return

    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Create destination directories and copy files
    print(f"Found {len(pairs)} pairs. {len(train_pairs)} for training, {len(val_pairs)} for validation.")
    
    # Training split directories
    train_images_dir = os.path.join(args.out_dir, "train", "images")
    train_annotations_dir = os.path.join(args.out_dir, "train", "annotations")
    copy_files(train_pairs, train_images_dir, train_annotations_dir)
    
    # Validation split directories
    val_images_dir = os.path.join(args.out_dir, "val", "images")
    val_annotations_dir = os.path.join(args.out_dir, "val", "annotations")
    copy_files(val_pairs, val_images_dir, val_annotations_dir)
    
    print("Dataset split completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split training images and annotations into training and validation sets.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory containing training annotations.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to store the split dataset.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of images to use for training (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    args = parser.parse_args()
    main(args)