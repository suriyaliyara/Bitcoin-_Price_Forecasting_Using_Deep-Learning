Dataset used:

Bitcoin Historical Data CSV

Features:

Price
Open
High
Low
Volume
Change %
Project Workflow
1. Data Cleaning

Performed preprocessing on raw crypto dataset:

Removed commas from numeric values
Converted volume units (K/M/B)
Cleaned percentage columns
Removed null values
Sorted data by date
2. Feature Scaling

Used:

MinMaxScaler

Purpose:

Normalize feature values
Improve deep learning model convergence
Prevent gradient instability

Saved scaler:

feature_scaler.pkl
3. Sequence Generation

Created sliding window sequences for time series forecasting.

Configuration:

Sequence Length = 60

Forecast Horizons:

1 Day
3 Days
7 Days
4. Deep Learning Models
CNN Model

Used convolution layers for extracting local temporal patterns from time series data.

Advantages:

Fast training
Good short-term forecasting
RNN Model

Used recurrent neural networks for sequential learning.

Advantages:

Learns historical dependencies
Simple architecture

Limitations:

Vanishing gradient problem
LSTM Model

Used Long Short-Term Memory networks for long-term dependency learning.

Advantages:

Better memory handling
Stable time series prediction
Best overall forecasting performance
Transformer Model

Used self-attention mechanism for advanced sequence learning.

Advantages:

Parallel processing
Captures complex relationships
State-of-the-art architecture

Limitations:

Higher computation cost
5. Training Process

Used:

Adam Optimizer
MSE Loss Function
Gradient Clipping
GPU Acceleration (CUDA if available)

Training Parameters:

Epochs: 30/50
Batch Size: 32
Learning Rate: 0.001

6. Evaluation Metrics

The following metrics were used:

Mean Absolute Error (MAE)

Measures average prediction error.

Root Mean Squared Error (RMSE)

Measures squared prediction error magnitude.

Mean Absolute Percentage Error (MAPE)

Measures percentage forecasting error.

R² Score

Measures model prediction accuracy.

7. Visualization

Generated:

Forecast vs Actual plots
Error distribution plots
Multi-model comparison charts

Visualization includes:

CNN
RNN
LSTM
Transformer
for:
1D
3D
7D forecasting
8. Model Saving

Saved trained models:

.pt files (PyTorch models)

Saved prediction history:

.pkl files

Purpose:

Reuse models in Streamlit dashboard
Faster inference without retraining
9. Streamlit Dashboard

Interactive dashboard features