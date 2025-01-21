# README

## 📄 About the Repository

This repository contains all the code used to generate experiments on petroleum derivative sales data using the **N-BEATS (Neural Basis Expansion Analysis Time Series)** deep learning model.  

To execute the experiments, use the **`main.py`** file, which provides the structure necessary to run the different model configurations.  

The experiments are organized into two main versions:  
1. **Last Year**: Forecasts with a 12-month horizon.  
2. **5 Years**: Consecutive models trained starting in 2019, where each year predicts the next, enabling a robust analysis over 60 forecast points.

The goal is to evaluate the efficiency of the N-BEATS model in forecasting petroleum derivative sales across different horizons and conditions, including periods marked by extraordinary events such as the pandemic.
