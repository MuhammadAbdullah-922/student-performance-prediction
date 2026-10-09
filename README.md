# Student Performance Prediction

Predicting students' final grade (G3) and pass/fail outcome using
machine learning on the UCI Student Performance dataset (Portuguese course, 649 students).

## Dataset
- Source: UCI Machine Learning Repository, Student Performance
- 649 rows, 33 columns, no missing values
- Target: `G3` (final grade, 0-20)

## Approach
1. Explored data (shape, missing values, correlations)
2. Converted categorical columns using one-hot encoding
3. Split data 80% train / 20% test
4. Trained Random Forest models (regression and classification)

## Results

### Regression (predicting G3)
| Input features | MAE | R2 |
|---|---|---|
| Without G1, G2 | 2.06 | 0.18 |
| With G1, G2 | 0.75 | 0.84 |

### Classification (pass if G3 >= 10)
Used `class_weight="balanced"` because the data is imbalanced
(about 88% pass, 12% fail).

| Input features | Accuracy | Fail recall | Fail precision |
|---|---|---|---|
| Without G1, G2 | 0.88 | 0.33 | 0.50 |
| With G1 only | 0.89 | 0.67 | 0.53 |

## Key Findings
- `failures` is the strongest negative factor for final grade
- Previous grades (G1, G2) are very strong predictors
- Accuracy alone is misleading on imbalanced data, so recall for the
  fail class was used as the main metric
- Using G1 (first-term grade) lets us identify at-risk students early

## Tech Stack
Python, Pandas, scikit-learn

## How to Run
pip install pandas scikit-learn
python src/first_ml.py