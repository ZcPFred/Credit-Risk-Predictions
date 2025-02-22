# Credit-Risk-Prediction
## Description
The aim is to predict the likelihood of borrowers experiencing financial distress—specifically, becoming 90 days or more overdue on loan payments within a two-year period.  A set of classification models including KNN,Random Forest, GradientBoosting, AdaBoost,and XGBoost has been trained, tuned, and evaluated with a 5-fold cross validation.

## Introduction
The dataset provided comprises approximately 250,000 anonymized borrower records, totaling around 15MB in size. This data is divided into a training set of 150,000 entries (10,026 is positive and 139,974 is negative) and a test set containing 101,503 entries.Each record includes 11 variables: one target variable, "SeriousDlqin2yrs," indicating whether the borrower experienced 90 days past due delinquency or worse , and 10 predictor variables offering various financial and demographic insights.


| Variable Name                        | Description                                                                                                                                              | Type       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| SeriousDlqin2yrs                     | Person experienced 90 days past due delinquency or worse                                                                                                 | Y/N        |
| RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits | percentage |
| age                                  | Age of borrower in years                                                                                                                                 | integer    |
| NumberOfTime30-59DaysPastDueNotWorse | Number of times borrower has been 30-59 days past due but no worse in the last 2 years.                                                                  | integer    |
| DebtRatio                            | Monthly debt payments, alimony,living costs divided by monthy gross income                                                                               | percentage |
| MonthlyIncome                        | Monthly income                                                                                                                                           | real       |
| NumberOfOpenCreditLinesAndLoans      | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)                                                     | integer    |
| NumberOfTimes90DaysLate              | Number of times borrower has been 90 days or more past due.                                                                                              | integer    |
| NumberRealEstateLoansOrLines         | Number of mortgage and real estate loans including home equity lines of credit                                                                           | integer    |
| NumberOfTime60-89DaysPastDueNotWorse | Number of times borrower has been 60-89 days past due but no worse in the last 2 years.                                                                  | integer    |
| NumberOfDependents                   | Number of dependents in family excluding themselves (spouse, children etc.)                                                                              | integer    |



The data contains no categorical feature except for the target variable. Supervised classification model is the natural approach.KNN and Logistic Regression will be used as baseline model, and more advanced approach like Random Forest and three Boosting models will be trained.

## Exploratory Data Analysis (EDA)
Statistical descriptions for each features of the data

|  | <br>SeriousDlqin2yrs | RevolvingUtilizationOfUnsecuredLines | age           | NumberOfTime30-59DaysPastDueNotWorse | DebtRatio     | MonthlyIncome | NumberOfOpenCreditLinesAndLoans | NumberOfTimes90DaysLate | NumberRealEstateLoansOrLines | NumberOfTime60-89DaysPastDueNotWorse | NumberOfDependents |
|---------| -------------------- | ------------------------------------ | ------------- | ------------------------------------ | ------------- | ------------- | ------------------------------- | ----------------------- | ---------------------------- | ------------------------------------ | ------------------ |
count| 150000.000000        | 150000.000000                        | 150000.000000 | 150000.000000                        | 150000.000000 | 1.202690e+05  | 150000.000000                   | 150000.000000           | 150000.000000                | 150000.000000                        | 146076.000000      |
mean| 0.066840             | 6.048438                             | 52.295207     | 0.421033                             | 353.005076    | 6.670221e+03  | 8.452760                        | 0.265973                | 1.018240                     | 0.240387                             | 0.757222           |
std| 0.249746             | 249.755371                           | 14.771866     | 4.192781                             | 2037.818523   | 1.438467e+04  | 5.145951                        | 4.169304                | 1.129771                     | 4.155179                             | 1.115086           |
min| 0.000000             | 0.000000                             | 0.000000      | 0.000000                             | 0.000000      | 0.000000e+00  | 0.000000                        | 0.000000                | 0.000000                     | 0.000000                             | 0.000000           |
25%| 0.000000             | 0.029867                             | 41.000000     | 0.000000                             | 0.175074      | 3.400000e+03  | 5.000000                        | 0.000000                | 0.000000                     | 0.000000                             | 0.000000           |
50%| 0.000000             | 0.154181                             | 52.000000     | 0.000000                             | 0.366508      | 5.400000e+03  | 8.000000                        | 0.000000                | 1.000000                     | 0.000000                             | 0.000000           |
75%| 0.000000             | 0.559046                             | 63.000000     | 0.000000                             | 0.868254      | 8.249000e+03  | 11.000000                       | 0.000000                | 2.000000                     | 0.000000                             | 1.000000           |
max| 1.000000             | 50708.000000                         | 109.000000    | 98.000000                            | 329664.000000 | 3.008750e+06  | 58.000000                       | 98.000000               | 54.000000                    | 98.000000                            | 20.000000          |

Below shows the distribution of each features. Age is approximately normally distributed. NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines,and NumberOfDependents are right skewed. The rest of the features are influenced by outliers/extreme values, and therefore not showing obvious distribution pattern.
![Distribution](https://github.com/user-attachments/assets/f14d32f3-e3c8-4e6d-ab14-d666a71efccd)

Correlation Analysis among features reflect that NumberOfTime30-59DaysPastDueNotWorse, 60-89DaysPastDueNotWorse, and NumberOfTimes90DaysLate are highly correlated (0.98-0.99), the rest of the variables are not obviusly correlated. While performing feature importance analysis is a regular approach to handel highly correlated variables, the set of algorithms we choose (not considering baseline models) are fairly robust to multicollinearity, so no further processing implemented for now.
![Correlation](https://github.com/user-attachments/assets/bb41f54e-17b5-45c2-b95e-d1e4bb4af82a)

The target variable is highly imbalanced.Only 6.7% of the data is positive, 93.3% are negative, which is very intuitive because you would expect defaulting will be a relative rare case. The imbalanced data means that accuracy will not be a good metric to evaluate model performence, AUC, Recall, Precision, and F1-Score will be used instead.

![Imbalanced](https://github.com/user-attachments/assets/c864205c-0fda-4cb3-b0e6-80ee103d87a1)

### Missing Value Processing
The MonthlyIncome feature contains 29,731 missing values (19.8% of the training data).The NumberOfDependents feature had 3924 missing values (2.6% of the training data). From the visulizations generated by Missingno, it seems the missig values in these two features are randomly distributed. And the missing value correlation analysis shows no obvious correlation between the missed values and the rest of the features. These indicates that the missing values in these two features is likely to be MAR (Missing At Random), and therefore median imputation has been implemented, taking the fact that these two features both influenced by outliers into consideration.
![Missingno Matrix](https://github.com/user-attachments/assets/2fdb4ddb-474e-4482-b719-70da857c39e4)
![Missing Value Count](https://github.com/user-attachments/assets/d9157a86-3319-45ea-ab91-bd7581077cf1)
![Missing Correlation](https://github.com/user-attachments/assets/133f833b-8ac4-4e6a-b985-3adb903d12fc)



### Outlier Processing

![Boxplot](https://github.com/user-attachments/assets/88b0b8e4-a785-4473-9c27-f6e62fb6faec)

### Result
 
