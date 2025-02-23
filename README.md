# **Drug Prescription Prediction Using Machine Learning & Neural Networks**
### **📌 Project Overview**
This project predicts **drug prescriptions** based on patient characteristics using **machine learning and neural networks**. It explores **optimization techniques, regularization, and hyperparameter tuning** to improve performance.

---

## **📌 Dataset**
- **Dataset:** [Drugs Prediction Dataset](https://www.kaggle.com/code/dharshanadhanendran/drugs-prediction-dataset/input)
- **Features:** Disease, Age, Gender, Severity
- **Target Variable:** Drug Prescribed
- **Goal:** Optimize ML & Neural Network models for best accuracy.

---

## **📌 Model Comparison & Findings**
| **Training Instance** | **Optimizer** | **Regularizer** | **Epochs** | **Early Stopping** | **Layers** | **Learning Rate** | **Accuracy** | **Loss** | **Precision** | **Recall** | **F1 Score** |
|----------------|-------------|--------------|--------|---------------|--------|---------------|----------|------|-----------|--------|---------|
| **Instance 1** | Default (None) | None | 50 | No | 3 | Default | 84.11% | 0.7201 | 90.36% | 84.11% | 82.14% |
| **Instance 2** | Adam | L1 (`0.001`) | 250 | Yes | 4 | 0.0017 | 90.31% | 0.8767 | 93.90% | 90.31% | 88.08% |
| **Instance 3** | RMSProp | L2 (`0.005`) | 300 | Yes | 4 | 0.001 | 89.53% | 0.8198 | 93.61% | 89.53% | 87.04% |
| **Instance 4** | SGD | L1 (`0.005`) | 200 | Yes | 4 | 0.005 | 85.66% | 1.2266 | 90.76% | 85.66% | 82.91% |
| **Instance 5** | Logistic Regression | None | 5000 Iterations | N/A | N/A | N/A | 67.83% | N/A | 70.16% | 67.83% | 63.77% |


✅ **Best Model:** **Adam + L1 Regularization**  
✔ **Highest Accuracy (`90.31%`)**  
✔ **Balanced Precision & Recall (`93.90%`, `90.31%`)**  
✔ **Best generalization with regularization techniques**  

---

## **📌 Conclusion**
- **Neural Networks significantly outperformed Logistic Regression.**
- **Adam + L1 Regularization achieved the best accuracy (`90.31%`).**
- **Hyperparameters for Logistic Regression:**
  - **Solver:** `saga`
  - **Regularization Strength (C):** `1.5`
  - **Max Iterations:** `5000`

---

## **📌 Video Presentation**

[Link to Video](https://drive.google.com/file/d/1-_9000000000000000000000000000000000000000/view?usp=sharing)


---**Thank You**


