from main import train_model_gradio, predict_model_gradio
from glob import glob
import pandas as pd
import os
# More specific pattern matching

# features_files = 'data/sampled_data.csv'
# labels_files = 'data/sampled_labels.csv'

# features_files = glob(features_files)
# labels_files = glob(labels_files)



# train_model_gradio(
#     model_type='attention',
#     output_dir='results/attention_model',
#     labels_paths=labels_files,
#     epochs=200,
#     features_paths=features_files)

# model_path = 'results/lstm_model'  # Specify the model path
# predictions_csv = 'results/lstm_model/predictions.csv'  # Specify the path for saving predictions
# plot_save_path = 'plots'  # Specify the path for saving plots
# save_plots = True  # Set to True to save plots
# num_plot_samples = 10  # Specify the number of samples to plot

# predict_model_gradio(
#     model_path=model_path,
#     labels_paths=labels_files,
#     features_paths=features_files,
#     sample_size=20,
#     batch_size=32,
#     predictions_csv=predictions_csv,
#     plot_save_path=plot_save_path,
#     save_plots=save_plots,
#     num_plot_samples=num_plot_samples,
#     model_type='lstm',  # Specify the model type
#     input_indices=None
# )

data_paths = sorted(glob('data/data-for-doc/features/*.csv', recursive=True))
labels_paths = sorted(glob('data/data-for-doc/labels/*.csv', recursive=True))

for i, path in enumerate(data_paths):
    model_types = ['attention']
    file_name = str(data_paths[i]).split('/')[-1]
    for mt in model_types:
        predict_model_gradio(
            model_path=f'results/{mt}_model',
            # model_path = 'results/attention_model/checkpoint-80000',
            labels_paths=[labels_paths[i]],
            features_paths=[data_paths[i]],
            batch_size=10,
            predictions_csv=f'results/gap_results/{mt}/'+'pred_'+file_name,
            plot_save_path=f'results/gap_results/{mt}/'+'plot_'+file_name[:-4]+'.png',
            save_plots=False,
            num_plot_samples=0,
            delta_t_force_recreate = True, 
            model_type=mt,
            input_indices=None
        )
