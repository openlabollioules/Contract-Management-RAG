import os
import shutil
from huggingface_hub import snapshot_download

def download_models():
    # Create directories for each model
    model_dirs = {
        "layout": "offline_models/marker/layout",
        "texify": "offline_models/marker/texify",
        "text_recognition": "offline_models/marker/text_recognition",
        "table_recognition": "offline_models/marker/table_recognition",
        "text_detection": "offline_models/marker/text_detection",
        "inline_math_detection": "offline_models/marker/inline_math_detection"
    }

    # Model repositories on Hugging Face
    model_repos = {
        "layout": "vikp/surya_layout",
        "texify": "vikp/surya_texify",
        "text_recognition": "vikp/surya_ocr",
        "table_recognition": "vikp/surya_table",
        "text_detection": "vikp/surya_detector",
        "inline_math_detection": "vikp/surya_math"
    }

    # Create directories if they don't exist
    for dir_path in model_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Download each model
    for model_name, repo_id in model_repos.items():
        print(f"Downloading {model_name} model...")
        try:
            # Download the model
            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=model_dirs[model_name],
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {model_name} model to {local_path}")
        except Exception as e:
            print(f"Error downloading {model_name} model: {str(e)}")

if __name__ == "__main__":
    download_models() 