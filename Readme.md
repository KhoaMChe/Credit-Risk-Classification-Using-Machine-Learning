AI-Powered Credit Risk Modeling
Predicting Loan Default using Machine Learning
Project Summary

This project builds an end-to-end Machine Learning pipeline to predict loan default risk using real-world financial data from LendingClub.

Unlike simple academic models, this project emphasizes:

Data quality & preprocessing
Avoiding data leakage
Model interpretability
Business-oriented evaluation (Recall for risk detection)

The goal is to simulate how a real financial institution would assess borrower risk before approving a loan.

Business Problem

Financial institutions face significant losses due to loan defaults.

The key challenge:

How can we identify high-risk borrowers before issuing a loan?

This project formulates the problem as a binary classification task:

1 → High risk (Default)
0 → Low risk (Fully Paid)
Key Contributions

✔ Built a clean ML pipeline from raw data to model evaluation
✔ Identified and removed data leakage features (critical in finance)
✔ Performed feature engineering + encoding strategies
✔ Analyzed feature importance for business insights
✔ Focused on Recall optimization to reduce financial risk

Dataset
Source: LendingClub (Kaggle)
Size: ~600,000+ records
Features: 100+ variables
Time range: 2007–2018
Feature Categories:
    Loan: loan_amnt, int_rate, installment
    Borrower: annual_inc, home_ownership
    Credit: fico_range, dti, revol_util
    Target: loan_default
    Data Preprocessing Pipeline
1. Missing Value Handling
Removed columns with >70% missing values
Imputed remaining:
Numerical → median
Categorical → mode
2. Feature Selection

Removed irrelevant features:

id, url, desc, zip_code, emp_title, ...

3. Target Engineering
loan_status → loan_default
Charged Off → 1
Fully Paid → 0

4. Data Leakage Removal 

Removed features that contain future information:

recoveries
collection_recovery_fee
total_rec_late_fee
total_rec_int


This step is critical to ensure real-world applicability.

Exploratory Data Analysis
Correlation analysis with target
Feature distribution analysis
Heatmap visualization
Key Observations:
Higher int_rate → higher default probability
High dti → increased financial risk
Credit behavior features strongly impact outcomes

Modeling
Models Implemented:
Logistic Regression
Random Forest
Evaluation Metrics:
Accuracy
Precision
Recall (priority)
F1-score

Focus:

Minimize False Negatives (missing risky borrowers)

Feature Importance (Insights)

Top predictive factors:

Credit score (FICO)
Loan amount
Interest rate
Credit utilization

Insight:
Some features initially showed high importance due to data leakage → removed for realistic modeling.

Results
Achieved strong classification performance
Improved Recall to better detect high-risk borrowers
Built a model closer to real-world credit scoring systems
What I Learned
Real-world ML is 80% data preprocessing
Data leakage can completely invalidate a model
Feature importance ≠ trustworthy without validation
Business metrics (Recall) matter more than Accuracy
Tech Stack
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn

Project Structure
├── data/                # Raw & processed data
├── notebooks/           # EDA & modeling
├── preprocessing/       # Data cleaning scripts
├── models/              # Trained models
├── README.md

Future Improvements
Hyperparameter tuning (GridSearch / Optuna)
Handle class imbalance (SMOTE, class weighting)
Try advanced models (XGBoost, LightGBM)
Deploy model (API / Web App)
Author

Khoa Dang

Computer Science Student
Interested in Data Science & Machine Learning
Ho Chi Minh City University of Technology
