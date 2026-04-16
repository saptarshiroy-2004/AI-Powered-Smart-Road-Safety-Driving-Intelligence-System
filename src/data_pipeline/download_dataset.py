import os
import subprocess

def check_kaggle_api():
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json):
        print(f"Error: Kaggle API key not found at {kaggle_json}")
        print("Please log in to Kaggle, go to your Account settings, and click 'Create New API Token'.")
        print("Then place the downloaded kaggle.json file in the ~/.kaggle/ directory.")
        print("Make sure to run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True

def download():
    print("Checking Kaggle API credentials...")
    if not check_kaggle_api():
        return
    
    os.makedirs("./data/raw", exist_ok=True)
    
    print("Downloading State Farm Distracted Driver Detection dataset...")
    # Because we're in a virtualenv, use the venv's kaggle path
    cmd = "./venv/bin/kaggle competitions download -c state-farm-distracted-driver-detection -p ./data/raw/"
    subprocess.run(cmd, shell=True)
    
    # We also need to unzip it
    zip_path = "./data/raw/state-farm-distracted-driver-detection.zip"
    if os.path.exists(zip_path):
        print("Extracting images... (This might take a few minutes as it's several GBs)")
        os.makedirs("./data/raw/state-farm", exist_ok=True)
        cmd_unzip = f"unzip -q {zip_path} -d ./data/raw/state-farm"
        subprocess.run(cmd_unzip, shell=True)
        print("Done! The raw data is available in ./data/raw/state-farm")
    else:
        print("Download failed or zip file not found. Ensure you have accepted the competition rules on Kaggle's website.")

if __name__ == "__main__":
    download()
