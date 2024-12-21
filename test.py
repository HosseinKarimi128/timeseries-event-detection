from main import train_model_gradio
from glob import glob
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# More specific pattern matching
labels_files = glob('data/**/Gaussian_Cp_EGMS_L3_*.csv', recursive=True)
features_files = glob('data/**/time_series_EGMS_L3_*.csv', recursive=True)

# Wrap your training function with accelerator
with accelerator.main_process_first():
    train_model_gradio(
        model_type='attention',
        labels_paths=labels_files,
        features_paths=features_files
    )