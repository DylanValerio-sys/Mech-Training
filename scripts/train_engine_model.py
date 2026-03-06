"""
Train a YOLOv8 model on engine parts dataset.

Downloads a labeled engine parts dataset and fine-tunes
YOLOv8n to detect parts like battery, radiator, alternator, etc.

SETUP (one-time):
    1. Go to https://app.roboflow.com and create a free account
    2. Go to Settings > API Keys > copy your Private API Key

USAGE:
    python scripts/train_engine_model.py --api-key YOUR_API_KEY

The trained model will be saved to: models/engine_parts_best.pt
"""

import os
import sys
import argparse
import shutil
import zipfile

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")


def download_dataset(api_key):
    """Download engine parts dataset using direct API calls (bypasses buggy Roboflow package)."""
    import requests

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Datasets to try in order (workspace/project/version)
    datasets = [
        ("final-year-project-5vbtb", "engine-parts-detector", [2, 3, 1], "Engine Parts Detector"),
        ("team-data", "car-parts-ybiev", [1, 2, 3], "Car Parts"),
        ("computervision-nyq7i", "car-components-dataset-otfbq", [2, 1, 3], "Car Components"),
    ]

    for workspace, project, versions, name in datasets:
        for ver in versions:
            url = f"https://universe.roboflow.com/ds/{project}?key={api_key}&format=yolov8&version={ver}"
            # Try the proper Roboflow download API
            api_url = f"https://api.roboflow.com/{workspace}/{project}/{ver}/yolov8?api_key={api_key}"

            print(f"\nTrying: {name} v{ver}...")

            try:
                # First get the download link from Roboflow API
                resp = requests.get(api_url, timeout=30)
                if resp.status_code != 200:
                    print(f"  API returned {resp.status_code}, trying next...")
                    continue

                data = resp.json()
                export_data = data.get("export", data.get("version", {}))
                download_link = export_data.get("link", "")

                if not download_link:
                    # Try alternate key names
                    for key in ["link", "download", "url"]:
                        if key in data:
                            download_link = data[key]
                            break

                if not download_link:
                    print(f"  No download link in API response, trying next...")
                    continue

                # Download the zip file
                print(f"  Downloading from: {download_link[:60]}...")
                zip_path = os.path.join(DATASET_DIR, "dataset.zip")

                zip_resp = requests.get(download_link, stream=True, timeout=300)
                if zip_resp.status_code != 200:
                    print(f"  Download failed ({zip_resp.status_code}), trying next...")
                    continue

                total_size = int(zip_resp.headers.get("content-length", 0))
                downloaded = 0

                with open(zip_path, "wb") as f:
                    for chunk in zip_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  Downloaded: {downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB ({pct:.0f}%)", end="", flush=True)
                        else:
                            print(f"\r  Downloaded: {downloaded // 1024 // 1024}MB", end="", flush=True)

                print()  # New line after progress

                # Verify it's a zip file
                if not zipfile.is_zipfile(zip_path):
                    print(f"  Downloaded file is not a valid zip, trying next...")
                    os.remove(zip_path)
                    continue

                # Extract
                extract_dir = os.path.join(DATASET_DIR, "engine_parts")
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)

                print(f"  Extracting...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)

                os.remove(zip_path)  # Clean up zip

                # Find data.yaml
                data_yaml = None
                for root, dirs, files in os.walk(extract_dir):
                    if "data.yaml" in files:
                        data_yaml = os.path.join(root, "data.yaml")
                        break

                if data_yaml:
                    dataset_dir = os.path.dirname(data_yaml)
                    print(f"  SUCCESS! Dataset ready at: {dataset_dir}")
                    return dataset_dir
                else:
                    print(f"  Extracted but no data.yaml found, trying next...")
                    continue

            except requests.exceptions.Timeout:
                print(f"  Timed out, trying next...")
                continue
            except Exception as e:
                print(f"  Error: {e}")
                continue

    print("\nERROR: Could not download any dataset.")
    print("Possible fixes:")
    print("  1. Double-check your API key at app.roboflow.com > Settings > API Keys")
    print("  2. Make sure you have internet connection")
    print("  3. Try again in a few minutes")
    sys.exit(1)


def train_model(dataset_dir, epochs=50, batch_size=8, img_size=640):
    """Fine-tune YOLOv8n on the downloaded dataset."""
    from ultralytics import YOLO

    data_yaml = os.path.join(dataset_dir, "data.yaml")

    # Show what we're training on
    print(f"\n{'='*50}")
    print(f"  TRAINING CONFIGURATION")
    print(f"{'='*50}")
    print(f"  Dataset config : {data_yaml}")
    print(f"  Base model     : yolov8n.pt (nano - fast on CPU)")
    print(f"  Epochs         : {epochs}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Image size     : {img_size}px")
    print(f"  Device         : CPU")

    # Show the classes
    try:
        with open(data_yaml, "r") as f:
            content = f.read()
        print(f"\n  Dataset contents:")
        for line in content.strip().split("\n"):
            print(f"    {line}")
    except Exception:
        pass

    print(f"\n{'='*50}")
    print(f"  Training starting... this will take 2-4 hours on CPU.")
    print(f"  You can leave this running and come back later.")
    print(f"  DO NOT close this window!")
    print(f"{'='*50}\n")

    # Load pre-trained YOLOv8n as starting point
    model = YOLO("yolov8n.pt")

    # Fine-tune
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="engine_parts",
        project=os.path.join(PROJECT_ROOT, "training"),
        patience=15,
        save=True,
        exist_ok=True,
        device="cpu",
        workers=2,
        verbose=True,
    )

    # Copy best weights to models/ folder
    best_weights = os.path.join(PROJECT_ROOT, "training", "engine_parts", "weights", "best.pt")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    dest_path = os.path.join(models_dir, "engine_parts_best.pt")
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, dest_path)
        print(f"\n{'='*50}")
        print(f"  TRAINING COMPLETE!")
        print(f"{'='*50}")
        print(f"  Model saved to: {dest_path}")
        print(f"\n  To test it, run:")
        print(f"    python main.py --auto")
        print(f"{'='*50}")
    else:
        print("WARNING: Could not find best.pt weights file.")
        print(f"  Check: {os.path.join(PROJECT_ROOT, 'training', 'engine_parts')}")

    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on engine parts dataset")
    parser.add_argument("--api-key", type=str, required=True,
                        help="Roboflow API key (get one free at app.roboflow.com)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8, lower to 4 if you run out of RAM)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Mech Training AI - Model Training")
    print("=" * 50)

    # Step 1: Download dataset
    dataset_dir = download_dataset(args.api_key)

    # Step 2: Train model
    train_model(dataset_dir, epochs=args.epochs, batch_size=args.batch)


if __name__ == "__main__":
    main()
