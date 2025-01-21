# README

---

## 📄 About the Repository

This repository contains all results obtained from experiments conducted with various time series forecasting models. The evaluated models include:

- **LSTM (Long Short-Term Memory)**  
- **N-BEATS (Neural Basis Expansion Analysis Time Series)**  
- **Time-MoE (Time Mixture of Experts)**  
- **LagLlama**  

The main objective of this repository is to provide a foundation for comparative analysis of the performance of these models in different forecasting scenarios.

---

## 📂 Repository Structure

The repository is organized into three main folders:

### **1. results_last_year**
- Contains results of forecasts with a 12-month horizon.  
- Each model was trained to predict the sale of petroleum products over the past year based on historical data.

### **2. results_5_years**
- Contains results from more robust experiments.  
- Models were consecutively trained starting in 2019, where each model predicted the following year.  
    - Example: The model trained up to 2019 predicted 2020; the 2020 model predicted 2021, and so on.  
- This approach generated **60 forecast points** and allowed performance evaluation during the pandemic event and beyond a single annual horizon.

### **3. other_best_results**
- Gathers the best results obtained by other colleagues, used for direct comparison with the models in this repository.

---

## 📊 Experiment Objectives

- **Model Evaluation**: Analyze the performance of different architectures in challenging scenarios, such as unpredictable events (e.g., the pandemic).  
- **Benchmark Comparison**: Assess the competitiveness of the implemented models compared to other studies.  
- **Temporal Robustness**: Evaluate how models perform over multiple years, considering long-term trends and seasonality.

