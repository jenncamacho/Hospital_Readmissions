# Hospital_Readmissions | Capstone Project
## A ML/AI MODEL PREDICTING HOSPITAL READMISSION RATES

## Documents
Notebook: https://github.com/jenncamacho/Hospital_Readmissions/blob/main/Capstone_Readmissions.ipynb](https://github.com/jenncamacho/Hospital_Readmissions/blob/main/Patient_Readmission_Capstone_Final.ipynb
Data: https://github.com/jenncamacho/Hospital_Readmissions/blob/main/Patient_informationread_readmit.xls

# Executive Summary

## Business Understanding 
Patient readmission rates are an important indicator used by doctors and hospital administrators to assess the quality of care.
Gaining better insight and prediction of readmission can be used at time of treatment to make better decisions whether a patient is ready to be released from the hospital and help the medical care providers improve the patient’s treatment plan during their stay.

## Problem statement
The objective is to develop a predictive model that can accurately determine the likelihood of hospital readmission for patients within a specific period following their discharge. By employing classification algorithms—such as Logistic Regression, Decision Trees, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN)—the model will be trained on a labeled dataset. The labels in this dataset categorically indicate whether each patient experienced a readmission post-discharge. This model aims to assist healthcare providers in identifying at-risk patients, thereby enabling timely and targeted interventions to reduce readmission rates.

## Findings
The best model for predicting hospital readmission of patients is the Logistic Regression Model with an accuracy score of 0.8440, precision score of 0.4220, recall score of 0.5000, and low training time of 0.0365 seconds. 

| Model              | Training Time (seconds) | Accuracy | Precision | Recall  |
|--------------------|-------------------------|----------|-----------|---------|
| LogisticRegression | 0.0365                  | 0.8440   | 0.4220    | 0.5000  |
| KNearestNeighbors  | 0.0046                  | 0.8199   | 0.5401    | 0.5131  |
| SVC                | 21.4163                 | 0.8440   | 0.4220    | 0.5000  |
| DecisionTree       | 0.0789                  | 0.7440   | 0.5172    | 0.5174  |
| RandomForest       | 1.8248                  | 0.8049   | 0.5313    | 0.5152  |

The Logistic Regression Model performed better than the KNN, SVC, DecisionTree, and RandomForest models, none of the models generated an impressive recall score: all fell in the 0.500-0.5174 range.  Implications of a recall score close to 0.500:

- Missed Readmissions: The model is missing 50% of the patients who were actually readmitted. These patients are classified as "not readmitted" (False Negatives).
- Risk of Missing Critical Cases: In a healthcare context, failing to identify patients at risk of readmission could lead to missed opportunities for timely intervention and care.
- Model Improvement Needed: The model might need improvement, such as using different features, fine-tuning the model parameters, or trying different algorithms that might better capture the patterns related to patient readmission.
- Recognizing that other contributing factors beyond those captured in the patient record could exist which are not available for use in the model.

## Recommendations

The Logistic Regression Model is useful to identify features that have the strongest correlation to whether a patient will be readmitted to the hospital.  

![image](https://github.com/jenncamacho/Hospital_Readmissions/assets/161406309/0fb5704b-8935-4dff-b04c-89c14aaa6a36)

It is recommended that the hospital administration and health care providers (doctors) give the following greater consideration when making a decision whether to discharge a patient from the hospital:

- Age: The higher the age the more likely to be readmitted
- ASA Rating: The more critical in nature the procedure, the higher more likely to be readmitted
- Discharge to Home Healthcare: Patients discharged to home health care are more likely to be readmitted
- Discharge to Acute Care Facility: Patients discharged to another acute care facility are more likely to not be readmitted
- Lengh of Stay: the longer the length of stay the more likely to not be readmitted.

Providers and administraters can group like features to apply weights to patient data when making determiniation to discharge a patient.  For example, SEX_Male and SEX_Female are two features that can be grouped together to see that male patients are slightly more likely to be readmitted, while female patients are slightly less likely to be readmitted. Outpatients are more likely to be readmitted than inpatients, suggesting that admitted a patient as an inpatient instead of receiving medical treatment as an outpatient may result in lower readmission rates. 

Logistic Regression Model will be  to assist healthcare providers in identifying at-risk patients, thereby enabling timely and targeted interventions to reduce readmission rates.

Expected results:
#### Feature importance. 
The hospital should consider the following numeric and categorical features which provide the greatest impact to the Logistic Regression model and target patient quality improvement strategies based on these features:

- LOS  (length of stay)            
- BIRTH_DATE
- ASA_RATING_C (health of patient prior to precedure)
- PATIENT_CLASS_NM (inpatient/outpatient)
- BMI   (body mass index)

### Data Understanding
Collected patient data includes demographics, vitals, medical history, previous admissions, lab results, medications, and other relevant features.
UCI MOVER dataset most closely comprises the data features I had outlines in my early proposal and is already de-identified and has been approved for public use.  I chose this dataset over an interal dataset from UCSF.  The UCI dataset compiles hospital data from 58,799 patients across 83,468 surgeries, including medical histories, surgery specifics, and high-fidelity physiological waveforms. 

### Dataset Overview

Rows:  65728 entries
Data columns: 22 columns

#### Patient Record Data and Definitions:

| Column Name           | Description                                                                                          |
|-----------------------|------------------------------------------------------------------------------------------------------|
| LOG_ID                | Hospital visit or encounter unique ID                                                                |
| MRN                   | Medical Record Number or Patient ID                                                                  |
| DISCH_DISP            | Discharge Disposition or Description                                                                 |
| HOSP_ADMSN_TIME       | Hospital Admission Time                                                                              |
| HOSP_DISCH_TIME       | Hospital Discharge Time                                                                              |
| LOS                   | Length of Stay                                                                                       |
| ICU_ADMIN_FLAG        | Admitted to ICU flag (true/false)                                                                    |
| READMIT               | Patient has been readmitted                                                                          |
| SURGERY_DATE          | Surgery Date                                                                                         |
| AGE                   | Patient Age                                                                                          |
| HEIGHT                | Height                                                                                               |
| WEIGHT                | Weight                                                                                               |
| SEX                   | Sex                                                                                                  |
| PRIMARY_ANES_TYPE_NM  | Primary anesthesia type                                                                               |
| ASA_RATING_C          | American Society of Anesthesiologists (ASA) Code - physical status of patients before surgery        |
| PATIENT_CLASS_GROUP   | Patient Class Group                                                                                  |
| PATIENT_CLASS_NM      | Patient Class Name: inpatient, outpatient                                                            |
| PRIMARY_PROCEDURE_NM  | Primary Procedure Name                                                                               |
| IN_OR_DTTM            | In Operating Room DateTime                                                                           |
| OUT_OR_DTTM           | Out of Operating Room DateTime                                                                       |
| AN_START_DATETIME     | Anesthesia Start DateTime                                                                            |
| AN_STOP_DATETIME      | Anesthesia Stop DateTime                                                                             |

### Exploratory Data Analysis (EDA) -Exploration:
⦁ remove spaces
⦁ make all lower case
⦁ remove or solve for missing value
⦁ Remove redundant features that don't add value to the model or predicting the object
⦁ numerics conversion
⦁ cardinality for categorical data to see the counts of unique values, drop feature if too much cardinality
⦁ remove duplicates
⦁ convert to integer


### List of columns to drop

- MRN and LOG_ID are ID numbers do not contribute to model prediction.
- HOSP_ADMSN_TIME, HOSP_DISCH_TIME, IN_OR_DTTM, OUT_OR_DTTM, AN_START_DATETIME, SURGERY_DATE, and AN_STOP_DATETIME provide admission, operating room and anastesia start and stop dates and times. These start dates are often the same, giving little additional insight to predicting a readmission- These should be removed.
- WEIGHT and HEIGHT were used to generate BMI. BMI has more relavance to evaluate patient health.
  

  ![image](https://github.com/jenncamacho/Hospital_Readmissions/assets/161406309/5555e865-fbff-4217-a876-992193f55281)

# EDA

Pairwise Relationships Between Variables: A pairplot creates scatter plots between each pair of features in the dataset. This helps you visualize how two variables relate to each other, whether there is a linear relationship, a non-linear relationship, or no relationship at all. For example, if you see a linear trend in a scatter plot, it suggests a correlation between the two variables.
Distribution of Each Variable: Along the diagonal of a pairplot, you typically see the distribution of each variable (often shown as a histogram or kernel density estimate). This helps you understand the distribution of individual features—whether they are normally distributed, skewed, or have outliers. These distributions can inform you about the variability of each feature and whether any preprocessing (like normalization or transformation) might be needed.
Categorical Separation (using hue): When you use the hue parameter, the pairplot colors the data points based on different categories of a categorical variable. This allows you to observe how different categories are distributed in relation to the features. It can help you identify whether certain categories are clustered together or separated from others, providing insights into how categorical variables interact with the continuous features.
Identifying Outliers: Outliers can often be spotted in the scatter plots of a pairplot. If you see points that are far removed from the general cluster of data points, these might be outliers. Identifying outliers is important because they can skew the results of statistical analyses and machine learning models.
Detecting Multicollinearity: Multicollinearity occurs when two or more variables are highly correlated with each other. In a pairplot, this might be indicated by a very strong linear relationship between two variables (i.e., a straight line in a scatter plot). High multicollinearity can be problematic in regression models, so a pairplot can help you detect and address this issue.
Data Clustering: If your dataset has natural groupings, these may become apparent in the scatter plots of a pairplot. For example, points that cluster together in several pairwise plots may indicate distinct groups or clusters in the data. This insight can be useful for tasks like clustering or classification.
Comparing Feature Interactions: By examining the plots, you can compare how different features interact with each other. For example, you might observe that certain features interact differently across categories defined by a hue variable. This can help you generate hypotheses about the underlying relationships in your data. Summary: A pairplot provides a compact way to visualize the relationships between pairs of variables, the distribution of individual variables, and the interaction between features and categorical labels. It is a valuable tool for initial exploratory data analysis (EDA), helping you uncover patterns, relationships, and potential issues (like outliers or multicollinearity) in your data.

![image](https://github.com/user-attachments/assets/ad90f83b-0f10-4d7c-95c2-bb9e323023a2)

###Correlation


![image](https://github.com/user-attachments/assets/9c0e8c5d-de96-4660-8ae1-c401d4351010)

#### Target Variable

**Output Variable**: `y` - Indicates whether a patient will be readmitted to the hospital following a previous hospital admission.  (binary: "yes:1", "no:0")

#### Expected results:
The target variable classifies patients into one of two categories:

- Readmitted: The patient will be readmitted to the hospital within the specified period.
- Not Readmitted: The patient will not be readmitted within the specified period.

## Expected techniques:
- Data Preprocessing: Handle missing values, encode categorical variables, and normalize/scale numerical features.
- Feature Engineering: Create meaningful features from raw data, such as the number of previous admissions, time since last admission, specific lab results, etc.
- Model Training: Use classification algorithms logistic regression, decision trees, SVM, or KNN to train a model on the labeled dataset where the label indicates whether the patient was readmitted.
- Prediction: Apply the trained model to test patient data to mimic future patient data to predict whether they are likely to be readmitted. ​
- Model Evaluation: Evaluate the model using appropriate metrics such as accuracy, precision, recall, F1 score.

### Data Preprocessing

- one hot encoding
- target encoding
- standard scaling
  
### Train/Test Split
With your data prepared, split it into a train and test set.
The goal was to develop the best model to predict whether a client will subscribe a term deposit by: 

- Building a Baseline Model before building the first model
- Applying various classification methods to the business problem
- Comparing the results of k-nearest neighbors, logistic regression, decision trees, and support vector machines

## Baseline Model Performance to Exceed:


| Dummy Classifier          | Accuracy                                   |
|---------------------------|--------------------------------------------|
| Train                     | 84.41%                                     |
| Test                      | 84.40%                                     |



## Logistic Regression 


## Logistic Regression Model Accuracy: 


| Logistic Regression         | Accuracy                                   |
|-----------------------------|--------------------------------------------|
| Train                       | 84.00%                                     |

### Interpretation

- The negative value means it decreases the log odds of readmission. So being an Inpatient Admission has a strong effect on reducing the likelihood of readmission.
- The positive values means it increases the log odds of readmission.  The ASA_RATING_C has a strong effect on increasing the likelihood of readmission.

![image](https://github.com/user-attachments/assets/1c56e163-ef73-471a-832f-33842342775c)
Interpretation¶
The negative value means it decreases the log odds of readmission. So being an Inpatient Admission has a strong effect on reducing the likelihood of readmission.
The positive values means it increases the log odds of readmission. The ASA_RATING_C has a strong effect on increasing the likelihood of readmission.     

### Model Comparison 

| Model              | Training Time (seconds) | Accuracy | Precision | Recall  |
|--------------------|-------------------------|----------|-----------|---------|
| LogisticRegression | 0.0365                  | 0.8440   | 0.4220    | 0.5000  |
| KNearestNeighbors  | 0.0046                  | 0.8199   | 0.5401    | 0.5131  |
| SVC                | 21.4163                 | 0.8440   | 0.4220    | 0.5000  |
| DecisionTree       | 0.0789                  | 0.7440   | 0.5172    | 0.5174  |
| RandomForest       | 1.8248                  | 0.8049   | 0.5313    | 0.5152  |



### Model Performance and Evaluation 

- Model Stability and Generalization:
- Training Time Consideration:
  
### Final Recommendation:

Logistic Regression is recommended for its overall balance between accuracy, precision, recall, and training time. K-Nearest Neighbors is also a strong candidate, especially if training time is a critical factor.

The cost or risk of a false negative is high given that a patient's health and the hospital's liability are at stake.

### Instructions
<pre>
Code Used: Python
Packages: Pandas, sklearn, numpy, scipy
Instructions: Please run the notebook in sequence
<< https://github.com/jenncamacho/Hospital_Readmissions/blob/main/Capstone_Readmissions.ipynb >>
</pre>

### License

This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
This project is open source.

