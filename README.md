# Trend Change Detection

![Trend Change Detection](https://github.com/HosseinKarimi128/change-point-detection/blob/main/banner.webp)

**Trend Change Detection** is a user-friendly graphical interface designed to facilitate the training and prediction of trend change detection models. Leveraging powerful machine learning architectures such as LSTM, CNN, and Attention mechanisms, this tool empowers users to analyze time-series data efficiently without delving deep into the underlying code.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Models](#models)
- [Dependencies](#dependencies)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Intuitive GUI**: Built with Gradio, providing an easy-to-navigate interface for both training and prediction tasks.
- **Multiple Model Architectures**: Choose between LSTM, CNN, and Attention-based models to suit your specific trend detection needs.
- **Data Upload**: Upload multiple CSV files for training and prediction seamlessly.
- **Customizable Training Parameters**: Adjust sample size, epochs, batch size, learning rate, and model type directly from the interface.
- **Real-time Visualization**: Monitor training loss over epochs and visualize prediction results with generated plots.
- **Prediction Flexibility**: Make predictions on new data, optionally upload ground truth labels, and specify sample indices for targeted analysis.
- **Comprehensive Logging**: Detailed logs are maintained to track the training and prediction processes for debugging and analysis.

## Demo

![Gradio Interface Screenshot](https://github.com/yourusername/trend-change-detection-gui/blob/main/assets/gradio_interface.png)

*Screenshot of the Gradio interface showcasing the Training and Prediction tabs.*

## Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/trend-change-detection-gui.git
cd trend-change-detection-gui
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Ensure you have `pip` updated to the latest version.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you encounter issues related to specific packages (e.g., `torch`), refer to their official installation guides to install compatible versions based on your system and hardware.

### 4. Prepare Directories

Ensure the following directories exist and are properly structured. If not, create them:

- `data/`: Place your training and prediction CSV files here.
- `results/`: This directory will store the trained models and prediction results.
  - `cnn_model/`
  - `lstm_model/`
  - `attention_model/`

**Note**: These directories are not tracked by Git. Ensure they are created before running the application.

## Usage

### Running the Application

Start the Gradio interface by executing the `app.py` script:

```bash
python app.py
```

Upon successful launch, Gradio will provide a local URL (e.g., `http://127.0.0.1:7860/`). Open this URL in your web browser to access the GUI.

### Training a Model

1. **Navigate to the "Train" Tab**: This section allows you to configure and start the training process.
2. **Upload Training Data**:
   - **Training Features CSV Files**: Upload one or multiple CSV files containing feature data.
   - **Training Labels CSV Files**: Upload corresponding label CSV files. These are optional but recommended for supervised training.
3. **Configure Training Parameters**:
   - **Sample Size**: Number of samples to use for training.
   - **Epochs**: Number of training iterations.
   - **Batch Size**: Number of samples per gradient update.
   - **Learning Rate**: Step size for the optimizer.
   - **Model Type**: Choose between LSTM, CNN, or Attention models.
4. **Start Training**: Click the "Start Training" button. A plot showing the loss over epochs will be generated upon completion.

### Making Predictions

1. **Navigate to the "Predict" Tab**: This section facilitates making predictions using a trained model.
2. **Upload Prediction Data**:
   - **Prediction Features CSV Files**: Upload feature CSV files for which you want to make predictions.
   - **Prediction Labels CSV Files (Optional)**: Upload corresponding label CSV files for evaluation.
3. **Configure Prediction Parameters**:
   - **Start Index & End Index**: Specify indices to focus predictions on a subset of the data.
   - **Sample Size**: Number of samples to use for prediction.
   - **Batch Size**: Number of samples per prediction batch.
   - **Save Plots**: Enable to generate and save prediction plots.
   - **Number of Plot Samples**: Specify how many samples to visualize.
   - **Model Type**: Choose the model architecture used for prediction.
4. **Start Prediction**: Click the "Start Prediction" button. The predictions will be displayed in a table, and plots will be generated if enabled.

## Project Structure

```
trend-change-detection-gui/
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/                  # Data directory (not tracked by Git)
│
├── results/               # Results directory (not tracked by Git)
│   ├── cnn_model/
│   ├── lstm_model/
│   └── attention_model/
│
└── assets/                # Contains images for README
    ├── banner.png
    └── gradio_interface.png
```

- **src/**: Contains all the source code modules for data processing, modeling, training, prediction, and utility functions.
- **app.py**: The main application script that sets up and launches the Gradio interface.
- **main.py**: Orchestrates the training and prediction processes, integrating with the Gradio interface.
- **requirements.txt**: Lists all Python dependencies required for the project.
- **data/**: Directory to store input CSV files for training and prediction.
- **results/**: Directory where trained models and prediction results are saved.
- **assets/**: Contains visual assets like images used in the README.

## Data Preparation

To ensure smooth training and prediction processes, adhere to the following guidelines for your CSV files:

- **Features CSV Files**:
  - Should contain time-series data with each row representing a time step.
  - No header row; data should start from the first row.
  - Each CSV should have consistent sequence lengths or be appropriately padded/truncated.

- **Labels CSV Files**:
  - Should correspond to the features CSV files.
  - Each row represents the label for the corresponding time step in the features.
  - Ensure labels are binary or scaled appropriately based on the model's requirements.

**Example Structure**:

| Time Step | Feature 1 | Feature 2 | ... | Feature N |
|-----------|-----------|-----------|-----|-----------|
| 0         | 0.5       | 1.2       | ... | 0.8       |
| 1         | 0.6       | 1.1       | ... | 0.7       |
| ...       | ...       | ...       | ... | ...       |

**Labels CSV**:

| Time Step | Label |
|-----------|-------|
| 0         | 0     |
| 1         | 1     |
| ...       | ...   |

## Models

The project supports three types of models:

1. **LSTM (Long Short-Term Memory)**:
   - Suitable for capturing temporal dependencies in time-series data.
   - Configurable hidden size, number of layers, and dropout rates.

2. **CNN (Convolutional Neural Network)**:
   - Effective for spatial feature extraction.
   - Configurable sequence length and number of features.

3. **Attention-Based Models**:
   - Utilizes attention mechanisms to focus on relevant parts of the input.
   - Configurable attention dimensions, LSTM layers (if combined), and dropout rates.

### Saving and Loading Models

- Trained models are saved in the `results/` directory under their respective model type folders (`lstm_model/`, `cnn_model/`, `attention_model/`).
- During prediction, models are loaded from these directories based on the selected model type.

**Note**: Ensure that the `results/` directory and its subdirectories exist and have the necessary read/write permissions.

## Dependencies

All required Python packages are listed in the `requirements.txt` file. Key dependencies include:

- **Gradio**: For building the web-based GUI.
- **PyTorch**: For building and training neural network models.
- **Transformers**: For leveraging pre-trained model configurations.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Scikit-learn**: For data preprocessing and splitting.
- **Matplotlib**: For generating plots and visualizations.
- **Torchinfo**: For model summaries.
- **Logging**: For detailed logging of processes.

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Logging

The project implements comprehensive logging to track the training and prediction workflows. Logs are saved in the `logs/` directory.

- **Setup**: Logging is configured in `src/utils.py` and initialized at the start of training and prediction processes.
- **Log Levels**: Includes DEBUG, INFO, WARNING, and ERROR levels to capture detailed information.
- **Log File**: `logs/project.log` captures all log entries.

**Note**: Ensure the `logs/` directory exists or is created automatically by the application.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please follow the steps below:

1. **Fork the Repository**: Click the "Fork" button at the top-right corner of the repository page.
2. **Create a Feature Branch**:  
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Commit Your Changes**:  
   ```bash
   git commit -m "Add some feature"
   ```
4. **Push to the Branch**:  
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request**: Navigate to your forked repository and click the "New Pull Request" button.

Please ensure that your contributions adhere to the project's coding standards and include relevant tests where applicable.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this software as per the terms of the license.

## Contact

For any inquiries, issues, or contributions, please reach out to:

- **Email**: hossein.karimi.0128@gmail.com
- **GitHub**: [@hosseinkarimi128](https://github.com/hosseinkarimi128)

---

*Developed with ❤️ by [HoKa_128](https://github.com/hosseinkarimi128)*
