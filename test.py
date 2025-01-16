from main import train_model_gradio, predict_model_gradio
from glob import glob
import pandas as pd
import os
# More specific pattern matching
# labels_files = glob('data/**/Gaussian_Cp_EGMS_L3_*.csv', recursive=True)
# features_files = glob('data/**/time_series_EGMS_L3_*.csv', recursive=True)

# train_model_gradio(
#     model_type='cnn',
#     labels_paths=labels_files,
#     features_paths=features_files
# )

data_paths = glob('data/zeenvlxysg5fit4y/processed/*.csv', recursive=True)
for i, path in enumerate(data_paths):
    model_types = ['lstm', 'cnn', 'attention']
    file_name = str(data_paths[i]).split('/')[-1]
    for mt in model_types:
        predict_model_gradio(
            model_path=f'results/{mt}_model',
            labels_paths=None,
            features_paths=[data_paths[i]],
            sample_size=None,
            batch_size=32,
            predictions_csv=f'results/{mt}_model/'+'pred_'+file_name,
            plot_save_path='plots',
            save_plots=True,
            num_plot_samples=0,
            model_type=mt,
            input_indices=(1,-1)
        )