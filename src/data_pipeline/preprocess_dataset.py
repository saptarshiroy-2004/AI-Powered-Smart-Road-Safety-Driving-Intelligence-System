import os
import cv2
import glob
from tqdm import tqdm

# We will read from the newly downloaded raw data
RAW_DIR = "./data/raw/state-farm/imgs/train"
PROCESSED_DIR = "./data/processed/state-farm/train"
IMAGE_SIZE = (224, 224) # Standard input size for modern CNNs (MobileNet, ResNet)

def preprocess():
    if not os.path.exists(RAW_DIR):
        print(f"Error: {RAW_DIR} does not exist. Please run download_dataset.py first.")
        return

    print("======================================================")
    print("   Starting Preprocessing for Driver Monitoring AI")
    print(f"   - Resizing to: {IMAGE_SIZE}")
    print("   - Converting to: Grayscale (Simulation for IR Cabin Cams)")
    print("======================================================")
    
    # The dataset has 10 classes labeled c0 through c9
    for i in range(10):
        class_name = f"c{i}"
        class_path = os.path.join(RAW_DIR, class_name)
        
        if not os.path.exists(class_path):
            continue
            
        save_path = os.path.join(PROCESSED_DIR, class_name)
        os.makedirs(save_path, exist_ok=True)
        
        images = glob.glob(os.path.join(class_path, "*.jpg"))
        
        # We use tqdm to show a nice progress bar
        for img_path in tqdm(images, desc=f"Processing Class {class_name} (Total: {len(images)} imgs)"):
            # 1. Read the raw image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 2. Convert to Grayscale
            # Real in-cabin cameras often shoot in IR (black & white) for night vision.
            # Training on grayscale helps our model generalize to night driving!
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 3. Resize Image
            # Deep Learning models expect fixed image dimensions.
            resized = cv2.resize(gray, IMAGE_SIZE)
            
            # 4. Save to processed folder
            base_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(save_path, base_name), resized)
            
    print("\nPreprocessing complete! Your clean data is inside ./data/processed/")

if __name__ == "__main__":
    preprocess()
