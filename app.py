# app.py

import gradio as gr
from main import train_model_gradio, predict_model_gradio
import pandas as pd

def gradio_train_interface(
    train_labels_files,
    train_features_files,
    sample_size,
    epochs,
    batch_size,
    learning_rate,
    model_type
):
    # The file paths are already provided when type='filepath'
    train_labels_paths = train_labels_files
    train_features_paths = train_features_files

    # Determine output directory based on model type
    output_dir = f'results/{model_type}_model'

    # Call the training function
    metrics = train_model_gradio(
        labels_paths=train_labels_paths,
        features_paths=train_features_paths,
        sample_size=int(sample_size),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        output_dir=output_dir,
        model_type=model_type
    )

    # Process metrics to display
    # Extract training loss and epoch numbers
    training_logs = [(log['epoch'], log['loss']) for log in metrics if 'loss' in log and 'epoch' in log and 'eval_loss' not in log]
    eval_logs = [(log['epoch'], log['eval_loss']) for log in metrics if 'eval_loss' in log and 'epoch' in log]

    if training_logs and eval_logs:
        # Separate epochs and losses
        training_epochs, training_losses = zip(*training_logs)
        eval_epochs, eval_losses = zip(*eval_logs)

        # Plot both training and evaluation loss
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(training_epochs, training_losses, label='Training Loss')
        plt.plot(eval_epochs, eval_losses, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plot_path = f'results/{model_type}_model/loss_over_epochs.png'
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    else:
        return "No loss data available."

def gradio_predict_interface(
    model_path,
    predict_labels_files,
    predict_features_files,
    sample_size,
    batch_size,
    save_plots,
    num_plot_samples,
    model_type
):
    # The file paths are already provided when type='filepath'
    predict_labels_paths = predict_labels_files
    predict_features_paths = predict_features_files

    # Convert save_plots to boolean
    save_plots = bool(save_plots)

    # Determine output paths based on model type
    output_dir = f'results/{model_type}_model'
    predictions_csv = f'{output_dir}/predictions.csv'
    plot_save_path = f'{output_dir}/model_output_sample.png'

    # Call the prediction function
    predictions_csv_path, plots = predict_model_gradio(
        model_path=model_path,
        labels_paths=predict_labels_paths,
        features_paths=predict_features_paths,
        sample_size=int(sample_size),
        batch_size=int(batch_size),
        predictions_csv=predictions_csv,
        plot_save_path=plot_save_path,
        save_plots=save_plots,
        num_plot_samples=int(num_plot_samples),
        model_type=model_type
    )

    # Read predictions CSV to display
    predictions_df = pd.read_csv(predictions_csv_path)
    # Return the predictions and plots
    return predictions_df, plots

# Set up Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Trend Change Detection GUI")

    with gr.Tabs():
        with gr.TabItem("Train"):
            gr.Markdown("## Training")

            train_labels_files = gr.File(label="Upload Training Labels CSV Files", file_count="multiple", type='filepath')
            train_features_files = gr.File(label="Upload Training Features CSV Files", file_count="multiple", type='filepath')

            sample_size = gr.Number(value=1000, label="Sample Size")
            epochs = gr.Number(value=100, label="Epochs")
            batch_size = gr.Number(value=32, label="Batch Size")
            learning_rate = gr.Number(value=0.001, label="Learning Rate")
            model_type = gr.Radio(choices=['lstm', 'cnn', 'attention'], value='lstm', label="Model Type")

            train_button = gr.Button("Start Training")
            train_output = gr.Image(label="Loss Over Epochs")

            train_button.click(
                gradio_train_interface,
                inputs=[
                    train_labels_files,
                    train_features_files,
                    sample_size,
                    epochs,
                    batch_size,
                    learning_rate,
                    model_type
                ],
                outputs=train_output
            )

        with gr.TabItem("Predict"):
            gr.Markdown("## Prediction")

            model_path = gr.Textbox(label="Model Path")
            predict_labels_files = gr.File(label="Upload Prediction Labels CSV Files", file_count="multiple", type='filepath')
            predict_features_files = gr.File(label="Upload Prediction Features CSV Files", file_count="multiple", type='filepath')

            sample_size = gr.Number(value=1000, label="Sample Size")
            batch_size = gr.Number(value=32, label="Batch Size")
            save_plots = gr.Checkbox(value=True, label="Save Plots")
            num_plot_samples = gr.Number(value=1, label="Number of Plot Samples")
            model_type = gr.Radio(choices=['lstm', 'cnn', 'attention'], value='lstm', label="Model Type")

            predict_button = gr.Button("Start Prediction")
            prediction_output = gr.Dataframe(label="Predictions")
            plot_output = gr.Gallery(label="Prediction Plots")

            predict_button.click(
                gradio_predict_interface,
                inputs=[
                    model_path,
                    predict_labels_files,
                    predict_features_files,
                    sample_size,
                    batch_size,
                    save_plots,
                    num_plot_samples,
                    model_type
                ],
                outputs=[prediction_output, plot_output]
            )


demo.launch()
