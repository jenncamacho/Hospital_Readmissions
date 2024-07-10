# Hospital_Readmissions | Capstone Project
## A ML/AI MODEL PREDICTING HOSPITAL READMISSION RATES

## Documents
Notebook: https://github.com/jenncamacho/Hospital_Readmissions/blob/main/Capstone_Readmissions.ipynb
Data: https://github.com/jenncamacho/Hospital_Readmissions/blob/main/Patient_informationread_readmit.xls

## Business Understanding 
Patient readmission rates are an important indicator used by doctors and hospital administrators to assess the quality of care.
Gaining better insight and prediction of readmission can be used at time of treatment to make better decisions whether a patient is ready to be released from the hospital and help the medical care providers improve the patient’s treatment plan during their stay.

## Problem statement
The objective is to develop a predictive model that can accurately determine the likelihood of hospital readmission for patients within a specific period following their discharge. By employing classification algorithms—such as Logistic Regression, Decision Trees, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN)—the model will be trained on a labeled dataset. The labels in this dataset categorically indicate whether each patient experienced a readmission post-discharge. This model aims to assist healthcare providers in identifying at-risk patients, thereby enabling timely and targeted interventions to reduce readmission rates.


## Data Understanding
Collect patient data including demographics, vitals, medical history, previous admissions, lab results, medications, and other relevant features.
UCI MOVER dataset most closely comprises the data features I had outlines in my early proposal and is already de-identified and has been approved for public use.  I chose this dataset over an interal dataset from UCSF.  The UCI dataset compiles hospital data from 58,799 patients across 83,468 surgeries, including medical histories, surgery specifics, and high-fidelity physiological waveforms. 

#### Dataset Overview

Rows:  65728 entries
Data columns: 22 columns

#### Exploratory Data Analysis (EDA) -Exploration:
⦁ remove spaces
⦁ make all lower case
⦁ remove or solve for missing value
⦁ Remove redundant features that don't add value to the model or predicting the object
⦁ numerics and date conversion
⦁ cardinality for categorical data to see the counts of unique values, drop feature if too much cardinality
⦁ remove duplicates
⦁ convert to integer

# List of columns to drop

- MRN and LOG_ID are ID numbers do not contribute to model prediction.
- HOSP_ADMSN_TIME, HOSP_DISCH_TIME, IN_OR_DTTM, OUT_OR_DTTM, AN_START_DATETIME, SURGERY_DATE, and AN_STOP_DATETIME provide admission, operating room and anastesia start and stop dates and times. These start dates are often the same, giving little additional insight to predicting a readmission.
- WEIGHT and HEIGHT were used to generate BMI. BMI has more relavance to evaluate patient health.
- These should be removed.

#### Target Variable

- **Output Variable**: `y` - Indicates whether a patient will be readmitted to the hospital following a previous hospital admission.  (binary: "yes:1", "no:0")

#### Expected results:
The target variable classifies patients into one of two categories:

Readmitted: The patient will be readmitted to the hospital within the specified period.
Not Readmitted: The patient will not be readmitted within the specified period.

## Data Preprocessing

- one hot encoding
- target encoding
- standard scaling

## Histogram showing distribution of numeric features

![image](https://github.com/jenncamacho/Hospital_Readmissions/assets/161406309/5555e865-fbff-4217-a876-992193f55281)


# List of columns to drop
FEATURE DESCRIPTION AND RELEVANCE (cont.)
MRN and LOG_ID are ID numbers do not contribute to model prediction.
HOSP_ADMSN_TIME, HOSP_DISCH_TIME, IN_OR_DTTM, OUT_OR_DTTM, AN_START_DATETIME, SURGERY_DATE, and AN_STOP_DATETIME provide admission, operating room and anastesia start and stop dates and times. These start dates are often the same, giving little additional insight to predicting a readmission.
WEIGHT and HEIGHT were used to generate BMI. BMI has more relavance to evaluate patient health.
These should be removed.


## Expected techniques:
- Data Preprocessing: Handle missing values, encode categorical variables, and normalize/scale numerical features.
- Feature Engineering: Create meaningful features from raw data, such as the number of previous admissions, time since last admission, specific lab results, etc.
- Model Training: Use classification algorithms logistic regression, decision trees, SVM, or KNN to train a model on the labeled dataset where the label indicates whether the patient was readmitted.
- Prediction: Apply the trained model to test patient data to mimic future patient data to predict whether they are likely to be readmitted. ​
- Model Evaluation: Evaluate the model using appropriate metrics such as accuracy, precision, recall, F1 score.

### Train/Test Split
With your data prepared, split it into a train and test set.
The goal was to develop the best model to predict whether a client will subscribe a term deposit by: 

- Building a Baseline Model before building the first model
- Applying various classification methods to the business problem
- Comparing the results of k-nearest neighbors, logistic regression, decision trees, and support vector machines

### Logistic Regression

![image](https://github.com/jenncamacho/Hospital_Readmissions/assets/161406309/0fb5704b-8935-4dff-b04c-89c14aaa6a36)

### Interpretation

- The negative value means it decreases the log odds of readmission. So being an Inpatient Admission has a strong effect on reducing the likelihood of readmission.
- The positive values means it increases the log odds of readmission.  The ASA_RATING_C has a strong effect on increasing the likelihood of readmission.
  
### Recommendations

#### The hospital should consider the following numeric and categorical features which provide the greatest impact to the best model and target patient quality improvement strategies based on these features:

- LOS              
- BIRTH_DATE
- ASA_RATING_C
- PATIENT_CLASS_NM
- BMI         

#### The best model based on accuracy of the test dataset

| Model              | Training Time (seconds) | Accuracy | Precision | Recall  |
|--------------------|-------------------------|----------|-----------|---------|
| LogisticRegression | 0.0190                  | 0.5978   | 0.5496    | 0.5742  |
| KNearestNeighbors  | 0.0558                  | 0.6655   | 0.4767    | 0.4774  |
| SVC                | 33.3366                 | 0.7959   | 0.3980    | 0.5000  |
| DecisionTree       | 0.0780                  | 0.7351   | 0.4603    | 0.4815  |


### Model Performance and Evaluation 

- Model Stability and Generalization:
- Training Time Consideration:
  
### Final Recommendation:

Logistic Regression is recommended for its overall balance between accuracy, precision, recall, and training time. K-Nearest Neighbors is also a strong candidate, especially if training time is a critical factor.

## Instructions
<pre>
Code Used: Python
Packages: Pandas, sklearn, numpy, scipy
Instructions: Please run the notebook in sequence
<< https://github.com/jenncamacho/Hospital_Readmissions/blob/main/Capstone_Readmissions.ipynb >>
</pre>

### License

This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
This project is open source.

