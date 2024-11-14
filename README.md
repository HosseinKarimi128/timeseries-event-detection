# CPD Project

## Overview

This project is a refactored version of the original Jupyter Notebook (`cpd.ipynb`). It is organized into a function-driven structure with separate modules for data processing, model definition, training, and prediction. Comprehensive logging is integrated throughout the project to facilitate debugging and monitoring.

## Project Structure

project/ ├── data/ │ └── (Your CSV data files) ├── src/ │ ├── init.py │ ├── data_processing.py │ ├── model.py │ ├── train.py │ ├── predict.py │ └── utils.py ├── logs/ │ └── project.log ├── main.py ├── requirements.txt └── README.md

bash
Copy code

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/project.git
   cd project
Install Dependencies

It's recommended to use a virtual environment.

bash
Copy code
pip install -r requirements.txt
Prepare Data

Place your CSV data files in the data/ directory or update the paths in main.py accordingly.

Usage
Training the Model
To train the model, run:

bash
Copy code
python main.py --train --mount_drive
--train: Flag to initiate training.
--mount_drive: (Optional) Flag to mount Google Drive if running on Google Colab.
Making Predictions
To make predictions using a trained model, run:

bash
Copy code
python main.py --predict --model_path /path/to/your/model --mount_drive
--predict: Flag to initiate prediction.
--model_path: Path to the trained model checkpoint.
--mount_drive: (Optional) Flag to mount Google Drive if running on Google Colab.
Combined Training and Prediction
You can combine training and prediction by using both flags:

bash
Copy code
python main.py --train --predict --mount_drive