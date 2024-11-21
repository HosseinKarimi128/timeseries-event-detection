import gradio as gr
from main import train_model_gradio, predict_model_gradio
import pandas as pd
import json  # Import json for formatting

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
    train_labels_paths = [file.name for file in train_labels_files] if train_labels_files else []
    train_features_paths = [file.name for file in train_features_files] if train_features_files else []

    # Determine output directory based on model type
    output_dir = f'results/{model_type}_model'

    # Call the training function
    metrics, evaluation_results = train_model_gradio(
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
    training_train_samples_per_second = [log['train_samples_per_second'] for log in metrics if 'train_samples_per_second' in log]
    evaluation_eval_samples_per_second = [log['eval_samples_per_second'] for log in metrics if 'eval_samples_per_second' in log]
    training_eval_runtime = [log['train_runtime'] for log in metrics if 'train_runtime' in log]
    evaluation_train_runtime = [log['eval_runtime'] for log in metrics if 'eval_runtime' in log]

    evaluation_results['# of epochs'] = len(training_logs)
    evaluation_results['final evaluation loss'] = eval_logs[-1][1] if eval_logs else None
    evaluation_results['final training loss'] = training_logs[-1][1] if training_logs else None
    evaluation_results['eval_samples_per_second'] = evaluation_eval_samples_per_second[-1] if evaluation_eval_samples_per_second else None
    evaluation_results['train_samples_per_second'] = training_train_samples_per_second[-1] if training_train_samples_per_second else None
    evaluation_results['eval_runtime'] = evaluation_train_runtime[-1] if evaluation_train_runtime else None
    evaluation_results['train_runtime'] = training_eval_runtime[-1] if training_eval_runtime else None

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

        # Convert evaluation_results to a DataFrame
        evaluation_df = pd.DataFrame(list(evaluation_results.items()), columns=['Metric', 'Value'])

        return plot_path, evaluation_df  # Return both outputs
    else:
        # Return an empty DataFrame with a message
        empty_df = pd.DataFrame(columns=['Metric', 'Value'])
        return None, empty_df.append({"Metric": "Message", "Value": "No loss data available."}, ignore_index=True)

def gradio_predict_interface(
    predict_labels_files,  # Optional input
    predict_features_files,
    sample_size,
    batch_size,
    save_plots,
    num_plot_samples,
    model_type,
    start_index=None,
    end_index=None
):

    # Determine model path based on model_type
    model_path = f'results/{model_type}_model'

    # Convert Gradio File objects to paths
    predict_features_paths = [file.name for file in predict_features_files] if predict_features_files else []
    predict_labels_paths = [file.name for file in predict_labels_files] if predict_labels_files else []

    # Determine output directory based on model type
    output_dir = f'results/{model_type}_model'

    if start_index is not None and end_index is not None:
        sample_indices = (start_index, end_index)
    else:
        sample_indices = None
    # Call the prediction function
    predictions_csv_path, plots = predict_model_gradio(
        model_path=model_path,
        labels_paths=predict_labels_paths,  # Pass actual paths or empty list
        features_paths=predict_features_paths,
        sample_size=int(sample_size),
        batch_size=int(batch_size),
        predictions_csv=f'{output_dir}/predictions.csv',
        plot_save_path=f'{output_dir}/model_output_sample.png',
        save_plots=save_plots,
        num_plot_samples=int(num_plot_samples),
        model_type=model_type,
        input_indices = sample_indices
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

            train_features_files = gr.File(label="Upload Training Features CSV Files", file_count="multiple", type='filepath')
            train_labels_files = gr.File(label="Upload Training Labels CSV Files", file_count="multiple", type='filepath')
        
            sample_size = gr.Number(value=1000, label="Sample Size")
            epochs = gr.Number(value=100, label="Epochs")
            batch_size = gr.Number(value=32, label="Batch Size")
            learning_rate = gr.Number(value=0.001, label="Learning Rate")
            model_type_train = gr.Radio(choices=['lstm', 'cnn', 'attention'], value='lstm', label="Model Type")

            train_button = gr.Button("Start Training")
            train_output_plot = gr.Image(label="Loss Over Epochs")
            train_output_log = gr.Dataframe(
                headers=["Metric", "Value"],
                label="Evaluation Results",
                interactive=False
            )  # Updated to Dataframe

            train_button.click(
                gradio_train_interface,
                inputs=[
                    train_labels_files,
                    train_features_files,
                    sample_size,
                    epochs,
                    batch_size,
                    learning_rate,
                    model_type_train
                ],
                outputs=[train_output_plot, train_output_log]  # Updated outputs
            )

        with gr.TabItem("Predict"):
            gr.Markdown("## Prediction")

            predict_features_files = gr.File(
                label="Upload Prediction Features CSV Files", 
                file_count="multiple", 
                type='filepath'
            )
            # Removed model_path textbox
            predict_labels_files = gr.File(
                label="Upload Prediction Labels CSV Files (Optional)", 
                file_count="multiple", 
                type='filepath'
            )

            start_index = gr.Number(value=None, label="Start Index")
            end_index = gr.Number(value=None, label="End Index")
            sample_size_predict = gr.Number(value=1000, label="Sample Size")
            batch_size_predict = gr.Number(value=32, label="Batch Size")
            save_plots = gr.Checkbox(value=True, label="Save Plots")
            num_plot_samples = gr.Number(value=1, label="Number of Plot Samples")
            model_type_predict = gr.Radio(choices=['lstm', 'cnn', 'attention'], value='lstm', label="Model Type")

            predict_button = gr.Button("Start Prediction")
            prediction_output = gr.Dataframe(label="Predictions")
            plot_output = gr.Gallery(label="Prediction Plots")

            predict_button.click(
                gradio_predict_interface,
                inputs=[
                    predict_labels_files,    # Optional labels
                    predict_features_files,
                    sample_size_predict,
                    batch_size_predict,
                    save_plots,
                    num_plot_samples,
                    model_type_predict,
                    start_index, 
                    end_index
                ],
                outputs=[prediction_output, plot_output]
            )

demo.launch()
