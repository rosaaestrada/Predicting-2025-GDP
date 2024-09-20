### Comparison of R-squared values for Logistic Regression and Random Forest Models for the Target Variable Economy (GDP)
<img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Comparison%20of%20R-squared%20for%20LR%20RF.png?raw=true" alt= "Comparison of R-squared values for Logistic Regression and Random Forest Models" width= "500" height= "350"> 

- This chart shows that both models have high predictive accuracy, with Random Forest (R^2 = 0.987) slightly out performing Logistic Regression (R^2 = 0.983) in explaining GDP growth rate variance. 


### Comparison of Metrics and Model Perfromance Across All Countries
<img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Comparison%20of%20metrics%20across%20all%20countries.png?raw=true" alt= "Comparison of Metrics Across All Countries" width= "" height=""> <img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Summary%20plot%20of%20model%20perfromance.png?raw=true" alt= "Summary Plot of Model Perfromance Across All countries" width= "" height="">

- Logistic Regression consistently outperforms Random Forest in terms of R-aquared values for all three countries.
- For Denmark, the Logistic Regression model achieves an impressive R-square value of 0.9037, while Random Forest falls behind at 0.6508.


### Residual Plot for Random Forest Model
<img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Residual%20plot.png?raw=true" alt= "Residual Plot for Random Forest Model" width= "" height="">

- As the actual economy increases, the residuals also tend to increase. In other words, at higher actual economy values, this Random Forest model tens to underpredict.


### Feature importance
<img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Feature%20importance.png?raw=true" alt= "Feature Importance" width= "" height="">

- The horizontal bar chart highlights the most influential features that drive the GDP growth predictions, which are:
  - Happiness rank
  - Social support
  - Happiness score


### Correlation Heatmap for USA
<img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Corr%20usa.png?raw=true" alt= "Correlation Heatmap for USA" width= "" height="">

- The heatmap represents the correlation matrix for various features related to happiness metrics in the USA.
- The darker shades of blue and green indicate stronger negative and positve correletions.
- Happiness rank, social support, and happiness score are positively correlated with each other, while corruption perception shows a negative corelation with other factors. 


### Correlation Heatmap for Finland
<img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Corr%20finland.png?raw=true" alt= "Correlation Heatmap for Finland" width= "" height="">


### Correlation Heatmap for Denmark
<img src= "https://github.com/rosaaestrada/Predicting-2025-GDP/blob/main/Results/Images/Corr%20denmark.png?raw=true" alt= "Correlation Heatmap for Denmark" width= "500" height="350">

&nbsp;






