#!/usr/bin/env python3
"""Download VibeVoice-1.5B model and voice samples."""
import os
import requests
from huggingface_hub import snapshot_download

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def download_from_github(url, dest_path):
    """Download file from GitHub."""
    print(f"  Downloading {os.path.basename(dest_path)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    model_name = "microsoft/VibeVoice-Realtime-0.5B"
    model_dir = f"./models/{model_name}"
    voices_dir = f"{model_dir}/voices"
    
    print(f"Downloading {model_name}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {model_dir}")
    
    # Download voice samples from GitHub (they're not on HuggingFace)
    # Available English voices: Carter, Davis, Emma, Frank, Grace, Mike
    voices = [
        ("Carter", "en-Carter_man.pt"),
        ("Davis", "en-Davis_man.pt"),
        ("Emma", "en-Emma_woman.pt"),
    ]
    
    print("\nDownloading voice samples...")
    for name, filename in voices:
        url = f"https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model/{filename}"
        dest_path = f"{voices_dir}/{name}.pt"
        download_from_github(url, dest_path)
    
    print(f"Voice samples downloaded to {voices_dir}")
    print(f"  Available: {[v[0] for v in voices]}")
    
    # Create test data
    os.makedirs("test_data", exist_ok=True)
    with open("test_data/sample.txt", "w") as f:
        f.write("""Hello, this is a test of the VibeVoice text-to-speech system. 
This sample text is short enough to generate quickly but long enough to demonstrate the quality.
Thank you for listening.""")
    print("\nCreated test_data/sample.txt")
    
    print("\n=== Setup Complete ===")
    print(f"Model: {model_dir}")
    print(f"Voices: {voices_dir}")

if __name__ == "__main__":
    main()
