# README

## 📄 About the Repository

This repository contains all the code used to generate experiments on petroleum derivative sales data using the **LSTM (Long Short-Term Memory)** deep learning model.  

To execute the experiments, use the **`main.py`** file, which provides the structure necessary to run the different model configurations.  

The experiments are organized into three main versions:  
1. **Last Year**: Forecasts with a 12-month horizon.  
2. **5 Years**: Consecutive models trained starting in 2019, where each year predicts the next, enabling a robust analysis over 60 forecast points.  
3. **Implementations**: The LSTM model is available in two different versions:  
   - **PyTorch**  
   - **Keras**  

The goal is to evaluate the efficiency of the LSTM model in forecasting petroleum derivative sales across different horizons and conditions, including periods marked by extraordinary events such as the pandemic.
