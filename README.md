# Predicting 2025 GDP: USA, Finland, and Denmark

[Kaggle.com: rosaaestrada - Predicting 2025 GDP: USA, Finland, and Denmark](https://www.kaggle.com/code/rosaaestrada/predicting-2025-gdp-usa-finland-and-denmark)

## Purpose
In this project I wanted to predict which country is likely to experience higher GDP growth rates in 2025 using Logistic Regression, Random Forest, and R-squared.

## Built with: 
- Python=
- 

## Files:
- Data - Contains raw data and preprocessed data
- Jupyter Notebook - The full source code along with explanations as a .ipynb file
- Python Code - The full source code along with explanations as a .py file
- Src - location on where the data was collected from
- Results - Summary Statistics, visualizations, and final evaluation of the project






## Data
**World Happiness Report:** Located on 
- [Kaggle.com - World Happiness Report up to 2022](https://www.kaggle.com/datasets/mathurinache/world-happiness-report)
- [Kaggle.com - World Happiness Report 2023](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2023)
- [Kaggle.com - World Happiness Report 2024](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2024)

------------------------------------------------------------------------------------------------------------------------
### Project overview and Objectives

🟤**Research Question:**

Which country, among the USA, Finland, and Denmark, is likely to experience higher GDP growth rates in 2025, using historical data from the World Happiness Report's from 2015 to 2024 while leveraging Advanced Machine Learning Algorithms for predictive analysis?

🟤 **Null Hypothesis (H0):**

There is no significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.

🟤 **Alternative Hypothesis (H1):**

There is a significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.

------------------------------------------------------------------------------------------------------------------------

### Methodology

This project employs a structured methodology consisting of several key stages: data cleaning, Exploratory Data Analysis (EDA), feature engineering, and feature selection. Following those steps, predictive modeling and analysis is conducted utilizing Logistic Regression and Random Forest algorithms to ensure the target variable perfroms well. Finally, the project culminates with a comprehensive predictive modeling of the selected countries.


------------------------------------------------------------------------------------------------------------------------
## Results
### Evaluation on the target variable performance

*Logistic Regression*
- Mean Absolute Error: 0.364
- Mean squared Error: 0.262
- R-squared (R^2): 0.983

*Random Forest*
- Mean Absolute Error: 0.286
- Mean Squared Error: 0.208
- R-squared (R^2): 0.987

R-squared values for both models are close to 1, indicating that the models explain a high percentage of the variance in the target variable (economic- GDP growth rate)
- Demonstrate strong potential in predicting economic indicators like GDP growth rates

------------------------------------------------------------------------------------------------------------------------

### Final Evaluation

**Mean Squared Error (MSE) & Mean Absolute Error (MAE)**

How close the model's predictions are to the actual happiness score
- Lower values indicate better performance
- Lower values indicate better performance

**R-Squared (R^2):**

Indicates the proportion of variance in the happiness scores that is explained by the model
- A higher R-squared value indicates better model fit to the data

**Feature Importance**

Reveals which features (predictors) have the most significant influence on predicting happiness scores
- The Higher ranked, the better the feature

------------------------------------------------------------------------------------------------------------------------

**USA**

*Evaluation metrics for Logistic Regression*
- Mean Absolute Error: 1.385301280854837
- Mean Squared Error: 2.0800079468125108
- R-squared (R^2): 0.8971424641686121

*Evaluation metrics for Random Forest*
- Mean Absolute Error: 2.6
- Mean Squared Error: 7.886666666666667
- R-squared (R^2): 0.6100000000000001

**FINLAND**

*Evaluation metrics for Logistic Regression*
- Mean Absolute Error: 1.0654896341077709
- Mean Squared Error: 2.2261193236481738
- R-squared (R^2): 0.8763267042417682

*Evaluation metrics for Random Forest*
- Mean Absolute Error: 2.3033333333333332
- Mean Squared Error: 5.8137
- R-squared (R^2): 0.6770166666666667

**DENMARK**

*Evaluation metrics for Logistic Regression*
- Mean Absolute Error: 1.1684256088259053
- Mean Squared Error: 1.73330246016317
- R-squared (R^2): 0.9037054188798239

*Evaluation metrics for Random Forest*
- Mean Absolute Error: 2.2866666666666666
- Mean Squared Error: 6.285666666666667
- R-squared (R^2): 0.6507962962962963

**R-Squared:** Represents the proportion of the variance in the dependent variable (GDP growth rates) that is predictable from the independent variables (features from the World Happiness Report) 
- Denmark consistently has the highest R-squared values from both Logistic Regression (0.9037) and Random Forest (0.6508) models.
  - This indicates that the model's prediction aligns more closely with the actual GDP growth rates for Denmark compared to the USA and Finland 

------------------------------------------------------------------------------------------------------------------------

🟤**Research Question:**

Which country, among the USA, Finland, and Denmark, will likely experience higher GDP growth rates in 2025, using historical data from the World Happiness Report's from 2015 to 2024 while leveraging Advanced Machine Learning Algorithms for predictive analysis?

🟤 **Null Hypothesis (H0):**

There is no significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.

🟤 **Alternative Hypothesis (H1):**

There is a significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.

🟢 Based on the provided data and analysis, Denmark will likely experience higher GDP growth rates in 2025 than the USA and Finland.

🟢 Therefore, we can reject the Null Hypothesis and accept the Alternative Hypothesis, suggesting that there is a significant difference in the predicted GDP growth rates among these countries.



![Economy, Life Expectancy and Happiness Score by Year](https://github.com/rosaaestrada/Predicting-2025-GDP-USA-FIN-DEN/blob/main/Visualizations/Economy%20-%20Chart.png?raw=true)

![Economy, Life Expectancy and Happiness Score by Year](https://github.com/rosaaestrada/Predicting-2025-GDP-USA-FIN-DEN/blob/main/Visualizations/Economy%20-%20TreeMap.png?raw=true)

![Social Support, Freedom, Corruption, and Happiness Score by Year](https://github.com/rosaaestrada/Predicting-2025-GDP-USA-FIN-DEN/blob/main/Visualizations/Social%20Support%20-%20Chart.png?raw=true)

![Social Support, Freedom, Corruption, and Happiness Score by Year](https://github.com/rosaaestrada/Predicting-2025-GDP-USA-FIN-DEN/blob/main/Visualizations/Social%20Support%20-%20TreeMap.png?raw=true)
