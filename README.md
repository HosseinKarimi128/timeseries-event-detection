## Time Series Event Detection with Change Point Detection Models

![Time Series Event Detection](banner.png)


This project provides a solution for detecting events in time series data using state-of-the-art models. It is designed to support applications such as change point detection in InSAR data, but it can be used for any event detection task by training models on the relevant dataset. The system includes a user-friendly interface, making it accessible to both technical and non-technical users.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Training a Model](#training-a-model)
    - [Making Predictions](#making-predictions)
5. [Supported Models](#supported-models)
6. [Requirements](#requirements)
7. [How It Works](#how-it-works)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

- **Multi-Model Support**: Train and use three types of models for event detection:
  - Long Short-Term Memory (LSTM)
  - Convolutional Neural Networks (CNN)
  - Attention-Based Models
- **Custom Data Compatibility**: Train models on your own time series datasets.
- **Interactive User Interface**: Built with Gradio, enabling users to train models and make predictions without any coding.
- **Event Visualizations**: Automatically generates plots for predicted events.
- **Change Point Detection for InSAR**: Specifically optimized for InSAR data but adaptable to other domains.

---

## Project Structure

```plaintext
- src/
    - __init__.py
    - data_preprocessing.py   # Utilities for processing time series data
    - model.py                # Definitions of LSTM, CNN, and Attention models
    - predict.py              # Functions for generating predictions and visualizations
    - train.py                # Training routines for the models
    - utils.py                # Logging and utility functions
- app.py                      # User interface with Gradio
- main.py                     # Core logic for training and prediction
- requirements.txt            # List of Python dependencies
- results/                    # Directory for storing trained models and results (not tracked in git)
    - cnn_model/
    - lstm_model/
    - attention_model/
- data/                       # Directory for storing datasets (not tracked in git)
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Launch the user interface:

```bash
python app.py
```

The Gradio GUI will open in your browser. It contains two tabs: **Train** and **Predict**.

### Training a Model

1. **Upload Data**:
   - Training Features: CSV files containing the time series data.
   - Training Labels: CSV files with corresponding labels indicating the presence of events.

2. **Set Parameters**:
   - Sample Size: Number of samples for training.
   - Epochs, Batch Size, Learning Rate: Standard training parameters.
   - Model Type: Choose from `lstm`, `cnn`, or `attention`.

3. **Start Training**:
   - Click the **Start Training** button. The system will train the model and display a plot of training and validation loss.

### Making Predictions

1. **Upload Data**:
   - Prediction Features: CSV files containing the time series data for prediction.
   - (Optional) Prediction Labels: Labels for evaluation purposes.

2. **Set Parameters**:
   - Sample Size, Batch Size: Control the data processing.
   - Save Plots: Optionally save visualizations of the predictions.
   - Number of Plot Samples: Specify the number of plots to generate.
   - Model Type: Choose the model type used for prediction.

3. **Start Prediction**:
   - Click the **Start Prediction** button. The system will generate predictions, display results in a table, and provide visualizations of detected events.

---

## Supported Models

- **LSTM**: Effective for sequential data modeling with memory capabilities.
- **CNN**: Suitable for capturing spatial and temporal features in time series data.
- **Attention-Based Models**: Focuses on the most relevant parts of the sequence for accurate event detection.

---

## Requirements

- Python 3.8 or later
- See `requirements.txt` for the complete list of dependencies.

---

## How It Works

1. **Data Preprocessing**:
   - Time series features are processed, scaled, and augmented with custom features (e.g., time deltas).
   - Labels are trimmed and synchronized with features.

2. **Model Training**:
   - Selected models are trained using PyTorch and Hugging Face's Trainer.
   - Training and validation loss are logged and visualized.

3. **Prediction**:
   - The trained model predicts event probabilities for each time step.
   - Results are saved to a CSV file, and plots are generated for visualization.

---

## Contributing

If you'd like to contribute to this project, feel free to submit issues or pull requests. Ensure any changes are tested and adhere to Python best practices.

---

## License

This project is not publicly available yet and is under private development for a specific paper. For usage rights, please contact the repository owner.

--- 

Enjoy using the tool for your time series event detection tasks!
