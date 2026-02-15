# ml-classification-models
Implementaiton of Different Machine Learning Classification Models.


### a. Problem Statement
The mobile phone market is highly competitive, making accurate price estimation crucial for manufacturers and consumers. This project aims to build a Machine Learning classification system to predict the Price Range of mobile phones (0: Low, 1: Medium, 2: High, 3: Very High) based on technical specifications like RAM, Battery Power, and Camera Resolution. By automating this classification, the system aids in effective market segmentation and pricing strategy optimization.


---

### b. Dataset Description
* **Dataset Name:** Mobile Price Classification
* **Source:** Publicly available dataset (Kaggle/UCI equivalent).
* **Size:**
    * **Instances:** 2,000 rows
    * **Features:** 20 input attributes (mix of numerical and categorical)
* **Target Variable:** `Price_Range` (Multi-class: 0, 1, 2, 3)
* **Key Features:**
    * **Hardware:** `Ram`, `Battery_Power`, `Int_Memory`, `N_Cores`, `Clock_Speed`.
    * **Display:** `Pixel_H`, `Pixel_W`, `Sc_H` (Screen Height), `Sc_W` (Screen Width).
    * **Camera:** `PC` (Primary Camera), `FC` (Front Camera).
    * **Connectivity:** `Blue` (Bluetooth), `WiFi`, `Four_G`, `Three_G`, `Dual_SIM`.

---

### C. Training Metrics
### Model Performance Results

| Model               | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| :------------------ | :------- | :---- | :-------- | :----- | :------- | :---- |
| Logistic Regression | 0.975    | 1.000 | 0.976     | 0.975  | 0.975    | 0.967 |
| Decision Tree       | 0.825    | 0.884 | 0.830     | 0.825  | 0.825    | 0.768 |
| KNN                 | 0.530    | 0.775 | 0.570     | 0.530  | 0.541    | 0.379 |
| Naive Bayes         | 0.797    | 0.958 | 0.806     | 0.797  | 0.799    | 0.731 |
| Random Forest       | 0.877    | 0.980 | 0.881     | 0.877  | 0.878    | 0.837 |
| XGBoost             | 0.902    | 0.992 | 0.903     | 0.902  | 0.902    | 0.870 |


### d. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Best Performer.** The model achieved near-perfect accuracy, indicating that the relationship between mobile specs (specifically RAM) and Price Range is highly linear. It is the most reliable model for this task. |
| **XGBoost** | **Second Best.** Gradient boosting provided excellent accuracy and effectively handled the multi-class nature of the data, but slightly overfitted compared to the simpler Logistic Regression. |
| **Random Forest** | **Strong Performance.** As an ensemble method, it corrected the overfitting issues of single Decision Trees, resulting in a highly robust classification boundary. |
| **Decision Tree** | **Moderate Performance.** The single tree structure struggled to capture the smooth, linear decision boundaries between price classes, leading to lower accuracy than the ensemble methods. |
| **Naive Bayes** | **Acceptable Baseline.** It performed decently by assuming feature independence, but failed to capture the strong direct correlation between specific features like RAM and the target Price. |
| **KNN** | **Lowest Performance.** KNN struggled significantly due to the high dimensionality of the data (20 features). The "noise" from less relevant features (like Touch Screen or 3G) diluted the distance metric, confusing the classifier. |