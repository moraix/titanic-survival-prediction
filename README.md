# Titanic Survival Prediction
Feature Engineering and Model Comparison

## Project Overview

This project implements an end-to-end machine learning pipeline to predict passenger survival on the Titanic dataset. The goal is not only to train predictive models, but also to demonstrate the importance of data preprocessing, feature engineering, and model comparison in a real-world ML workflow.

Two models were trained and evaluated:

- Logistic Regression (baseline, interpretable model)
    
- Random Forest (non-linear, ensemble-based model)

## Dataset

The dataset contains passenger-level information such as:

- Socio-economic class (Pclass)

- Demographics (Age, Sex)

- Family relationships (SibSp, Parch)

- Ticket fare and embarkation port

The target variable is:

- Survived (0 = Not survived, 1 = Survived)

## Data Preprocessing

The following preprocessing steps were applied:

- One-hot encoding for categorical variables (Sex, Embarked)

- Normalization of numerical features (Age, Fare) to the range [0, 1]

- Conversion of boolean features to numeric (0/1)

This ensured the dataset was fully numeric and suitable for models like Logistic Regression.

## Feature Engineering

To improve model performance and capture meaningful patterns, the following features were created:

- FamilySize = SibSp + Parch + 1

- IsAlone = 1 if FamilySize == 1, else 0

- FarePerPerson = Fare / FamilySize

These features help the models better understand social and economic factors affecting survival.

## Train-Test Split

The dataset was split into:

- 80% training set

- 20% test set

Stratified sampling was used to preserve class balance between survived and non-survived passengers.

## Model Training and Evaluation
### Logistic Regression

Logistic Regression was used as a strong and interpretable baseline model.
Its performance was evaluated using accuracy, precision, recall, F1-score, and a Confusion Matrix Visualization, which clearly shows the distribution of correct and incorrect predictions across both classes.

<img width="770" height="545" alt="output01" src="https://github.com/user-attachments/assets/824e8a2f-b2e9-46da-8332-87fb9e7bff4d" />

The confusion matrix visualization helps interpret:

- True positives (correctly predicted survivors)

- True negatives (correctly predicted non-survivors)

- False positives and false negatives

### Random Forest

A Random Forest classifier was trained with hyperparameter tuning using grid search and cross-validation.
Although the model can capture non-linear relationships and feature interactions, its test performance was similar to (and slightly lower than) Logistic Regression.

<img width="770" height="545" alt="output" src="https://github.com/user-attachments/assets/18abe9b5-b1e1-4045-be77-cb468ceb4ff0" />

A Confusion Matrix Visualization was also generated for Random Forest to visually compare its prediction behavior with the baseline model.

## Model Comparison

Both models achieved around 80% accuracy, with very similar precision, recall, and F1-scores.
Despite Random Forest’s complexity, Logistic Regression performed slightly better overall.

This comparison demonstrates that:

Well-engineered features with a simple model can outperform or match more complex models.

## Conclusion

This project shows the complete lifecycle of a machine learning task—from raw data to model evaluation. The results highlight that feature engineering and data quality often matter more than model complexity, and that interpretable baseline models should not be underestimated in practical ML applications.

### Tools & Libraries

- Python

- Pandas, NumPy

- Scikit-learn

- Jupyter Notebook
