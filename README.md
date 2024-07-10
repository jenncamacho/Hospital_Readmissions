# Hospital_Readmissions
Capstone Project
## A ML/AI MODEL PREDICTING HOSPITAL READMISSION RATES

# Documents
Notebook: https://github.com/jenncamacho/PA3_bank_marketing/blob/main/prompt_III_PREP.ipynb

bank-additional-full.csv with all examples (41188) and 20 inputs

## Business Understanding 
Patient readmission rates are an important indicator used by doctors and hospital administrators to assess the quality of care.
Gaining better insight and prediction of readmission can be used at time of treatment to make better decisions whether a patient is ready to be released from the hospital and help the medical care providers improve the patient’s treatment plan during their stay.

## Problem statement
The objective is to develop a predictive model that can accurately determine the likelihood of hospital readmission for patients within a specific period following their discharge. By employing classification algorithms—such as Logistic Regression, Decision Trees, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN)—the model will be trained on a labeled dataset. The labels in this dataset categorically indicate whether each patient experienced a readmission post-discharge. This model aims to assist healthcare providers in identifying at-risk patients, thereby enabling timely and targeted interventions to reduce readmission rates.


## Data Understanding
Collect patient data including demographics, vitals, medical history, previous admissions, lab results, medications, and other relevant features.

#### Dataset Overview


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


## Pairplot


# List of columns to drop
FEATURE DESCRIPTION AND RELEVANCE (cont.)
MRN and LOG_ID are ID numbers do not contribute to model prediction.
HOSP_ADMSN_TIME, HOSP_DISCH_TIME, IN_OR_DTTM, OUT_OR_DTTM, AN_START_DATETIME, SURGERY_DATE, and AN_STOP_DATETIME provide admission, operating room and anastesia start and stop dates and times. These start dates are often the same, giving little additional insight to predicting a readmission.
WEIGHT and HEIGHT were used to generate BMI. BMI has more relavance to evaluate patient health.
These should be removed.

## The Plot demonstrates that the dataset needs to be scaled:



## This plot shows that the dataset has been successfully scaled:


## Expected techniques:
- Data Preprocessing: Handle missing values, encode categorical variables, and normalize/scale numerical features.
- Feature Engineering: Create meaningful features from raw data, such as the number of previous admissions, time since last admission, specific lab results, etc.
- Model Training: Use classification algorithms logistic regression, decision trees, SVM, or KNN to train a model on the labeled dataset where the label indicates whether the patient was readmitted.
- Prediction: Apply the trained model to test patient data to mimic future patient data to predict whether they are likely to be readmitted. ​
- Model Evaluation: Evaluate the model using appropriate metrics such as accuracy, precision, recall, F1 score.

## Data Processing and Modeling
<pre>
Code Used: Python
Packages: Pandas, sklearn, numpy, scipy
Instructions: Please run the notebook in sequence
<<Notebook link>>
</pre>

### Train/Test Split
With your data prepared, split it into a train and test set.
The goal was to develop the best model to predict whether a client will subscribe a term deposit by: 

- Building a Baseline Model before building the first model
- Applying various classification methods to the business problem
- Comparing the results of k-nearest neighbors, logistic regression, decision trees, and support vector machines


### Recommendations

#### The hospital should consider the following numeric and categorical features which provide the greatest impact to the best model and target patient quality improvement strategies based on these features:

- age
- campaign
- previous
- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

#### The best model based on accuracy of the test dataset is Logistic Regression with an accuracy of 87.3%


| Model              | Train Time (s) | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall |
|--------------------|----------------|----------------|---------------|----------------|---------------|--------------|-------------|
| LogisticRegression | 0.398813       | 0.871146       | 0.873367      | 0.727586       | 0.714596      | 0.562705     | 0.556600    |
| KNearestNeighbors  | 0.028972       | 0.892789       | 0.869883      | 0.794366       | 0.697347      | 0.674464     | 0.618765    |
| SVC                | 3.387884       | 0.866269       | 0.870580      | 0.433134       | 0.435290      | 0.500000     | 0.500000    |
| DecisionTree       | 0.047040       | 0.952447       | 0.848284      | 0.962203       | 0.642278      | 0.829642     | 0.611516    |

### Model Performance
Logistic Regression has the highest test accuracy (0.873367) and a balanced precision (0.714596) and recall (0.556600). It performs consistently well on both the training and test sets, indicating a good generalization. K-Nearest Neighbors also shows high test accuracy (0.869883) and relatively balanced precision (0.697347) and recall (0.618765). It has the fastest training time, which might be an advantage if training speed is crucial. SVC has similar test accuracy (0.870580) to Logistic Regression and KNN but shows significantly lower precision and recall. This suggests that while it can separate the classes well, it might not be as reliable for imbalanced classes or more nuanced predictions. Decision Tree has the highest train accuracy (0.952447) but lower test accuracy (0.848284), indicating overfitting. Although it has high precision (0.642278) and recall (0.611516) on the test data, the drop from training performance suggests it may not generalize as well as the other models.

### Model Stability and Generalization:
Logistic Regression shows stable performance with minimal overfitting, making it a reliable choice. K-Nearest Neighbors also shows minimal overfitting with a slightly lower but still competitive test accuracy. SVC's low precision and recall might make it less desirable unless further tuning or different kernel functions can improve its performance. Decision Tree shows signs of overfitting, indicating it might need pruning, more data, or additional tuning to improve generalization.

### Training Time Consideration:
K-Nearest Neighbors has the fastest training time, which might be beneficial for very large datasets or when quick retraining is necessary. Logistic Regression has a reasonable training time and good performance, making it a balanced choice. SVC takes significantly longer to train, which might not be ideal for all applications. Decision Tree also has a quick training time but suffers from overfitting.

### Final Recommendation:
**Based on the analysis, the best models for predicting whether a client will subscribe to a deposit term are Logistic Regression and K-Nearest Neighbors.**

Logistic Regression is recommended for its overall balance between accuracy, precision, recall, and training time. K-Nearest Neighbors is also a strong candidate, especially if training time is a critical factor.



### License

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
This project is open source.

