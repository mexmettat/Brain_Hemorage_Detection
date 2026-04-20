import os
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

def augment_data():
    base_path = "data"
    raw_path = os.path.join(base_path, "raw")
    aug_path = os.path.join(base_path, "augmented")
    
    # 1. Setup folders
    categories = ["hemorrhage", "normal"]
    for cat in categories:
        os.makedirs(os.path.join(aug_path, cat), exist_ok=True)
        
    # 2. Define augmentations
    augmenter = T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.RandomRotation(15)
    ])
    
    # 3. Process each category
    for cat in categories:
        src_dir = os.path.join(raw_path, cat)
        dest_dir = os.path.join(aug_path, cat)
        
        # Recursive search for all images
        image_files = []
        for root, _, filenames in os.walk(src_dir):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Store relative path to handle subfolders
                    rel_path = os.path.relpath(os.path.join(root, f), src_dir)
                    image_files.append(rel_path)
        
        print(f"Augmenting {cat} ({len(image_files)} originals including subfolders)...")
        
        for rel_f in tqdm(image_files):
            img_path = os.path.join(src_dir, rel_f)
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Cleanup filename for saving (replace slashes with underscores to flatten)
                clean_name = rel_f.replace(os.sep, "_")
                name, ext = os.path.splitext(clean_name)
                
                # Create 5 augmented versions
                for i in range(5):
                    aug_img = augmenter(img)
                    aug_filename = f"{name}_aug_{i}{ext}"
                    aug_img.save(os.path.join(dest_dir, aug_filename))
                    
            except Exception as e:
                print(f"Error processing {rel_f}: {e}")

    print("Augmentation complete.")

if __name__ == "__main__":
    augment_data()
