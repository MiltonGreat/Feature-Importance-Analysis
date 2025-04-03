# Loan Default Risk Analysis Feature Importance

![screenshot-localhost_8888-2025 04 03-10_05_44](https://github.com/user-attachments/assets/4dcb45c2-a9dd-4cce-95aa-55cc3216dbb5)

### Project Overview

This module analyzes key drivers of loan default risk using machine learning (XGBoost/LightGBM) and interpretability techniques. Focused on income, credit history, and employment status, it answers:

- Which factors most influence default risk?
- How do income and credit scores interact?
- What thresholds should guide lending policies?

#### Techniques Used:
- Feature Importance (Global)
- Partial Dependence Plots (Local Effects)
- Feature Interaction Analysis

### Dataset

The dataset contains financial and demographic information related to credit applicants. The key features used in this model include:

- Credit History – Previous loan repayment behavior.
- Purpose – The reason for applying for credit.
- Savings – Amount of savings available.
- Amount – Loan amount requested.
- Duration – Loan repayment period.

The target variable is Credit Risk (Good/Bad Credit), where 1 represents bad credit and 0 represents good credit.

### Key Findings

- High-Risk: EXT_SOURCE_3 < 0.3 + AMT_INCOME_TOTAL < $30K → 3× higher defaults
- Low-Risk: EXT_SOURCE_3 > 0.7 + stable employment → Safe for lower interest rates

### Business Applications

1. Loan Approval Rules
- Auto-approve: EXT_SOURCE_3 > 0.7 + income > loan amount.
- Manual review: EXT_SOURCE_3 0.3–0.7 or employment <1 year.

2. Risk Monitoring
- Track drift in EXT_SOURCE_3 monthly (alert if distribution shifts).
- Audit loans where predictions disagree with SHAP explanations.

### Conclusion

This project demonstrated that credit history is the most critical factor in default prediction, even more than income or employment status. By using interpretable machine learning, we enabled data-driven lending decisions while maintaining transparency for stakeholders.

### Source

![Home Credit Default Risk Dataset from Kaggle](https://www.kaggle.com/datasets/anggundwilestari/home-credit)
