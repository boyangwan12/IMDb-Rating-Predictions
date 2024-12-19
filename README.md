# IMDb-Rating-Predictions
## Technologies and Skills
### Technologies Used:
- 📊 **Programming Language**: R
- 🛠️ **Libraries and Tools**: ggplot2, dplyr, caret, MASS
- 📈 **Regression Models**: Linear, Polynomial, and Spline Regression
- 🔍 **Validation Techniques**: Leave-One-Out Cross-Validation (LOOCV)

### Skills Demonstrated:
- 🧹 **Data Preprocessing**: Cleaning and transforming datasets, handling outliers, and creating binary indicators for categorical variables.
- 🔢 **Regression Modeling**: Testing and comparing linear, polynomial, and spline models to select the best predictors.
- 📉 **Model Evaluation**: Using metrics like Adjusted R², Residual Standard Error, and RMSE to assess model performance.
- 🧠 **Feature Engineering**: Creating and selecting key predictors such as genres, IMDb Pro rankings, and media coverage for improved model accuracy.
- 📊 **Data Visualization**: Generating meaningful graphs and charts to present insights using ggplot2.
- 📈 **Business Insights**: Translating technical results into actionable strategies for filmmakers, studios, and marketers.
- 🤝 **Collaboration**: Working effectively in a team to ensure accurate analysis and impactful deliverables.
- 🎤 **Communication**: Delivering project outcomes and recommendations through detailed reports and engaging presentations.


## Overview

This project aims to predict IMDb ratings for movies using a combination of film characteristics, cast information, and production-related details. By analyzing historical data, we identified key factors influencing ratings and developed a predictive model to assist filmmakers, studios, and marketers in making data-driven decisions.

---

## Objective

- Develop a predictive model to estimate IMDb ratings for upcoming films.
- Identify key predictors influencing audience perception and critical success.

---
## Dataset

The dataset contains information on approximately 2,000 films spanning from 1936 to 2018. Features include:
- **Numerical Variables**: Budget, duration, number of news articles, IMDb Pro rankings, etc.
- **Categorical Variables**: Genre, release month, maturity rating, language, production country, etc.

**Target Variable**: IMDb Rating (1 to 10 scale).

---

## Methods and Workflow

### 1. Data Preprocessing
- Removed redundant features such as director and production company due to high cardinality.
- Handled outliers in numerical predictors like budget, release year, and duration.
- Created binary indicators for key categorical variables (e.g., "Is English" for language).

### 2. Model Selection
- Tested linear, polynomial, and spline regression models for individual predictors.
- Used ANOVA and adjusted R² to select the best models for each feature.
- Applied HC1 heteroskedasticity-consistent covariance matrix estimation.

### 3. Final Model
- The final model includes predictors such as:
  - **Genres**: Drama, Animation, Documentary, etc.
  - **Duration**: Movie length (minutes).
  - **IMDb Pro Rankings**: Popularity on IMDb Pro.
  - **Number of News Articles**: Media coverage.
  - **Maturity Rating**: Audience content rating.
  - **Release Year and Month**: Temporal factors.
- Adjusted R²: 0.505
- Residual Standard Error: 0.746

### 4. Model Validation
- Used Leave-One-Out Cross-Validation (LOOCV) to assess performance:
  - **MSE**: 0.575
  - **RMSE**: 0.758

---

## Results and Insights

### Predicted Scores for Sample Movies:
| Movie Name                     | Predicted IMDb Rating |
|--------------------------------|-----------------------|
| Venom: The Last Dance          | 5.5                   |
| Your Monster                   | 5.2                   |
| HitPig!                        | 5.6                   |
| A Real Pain                    | 6.3                   |
| Elevation                      | 4.7                   |

### Key Insights:
- **Positive Predictors**:
  - **Documentary and Animation**: Genres strongly associated with higher ratings.
  - **Number of News Articles**: Higher media coverage correlates with better ratings.
- **Negative Predictors**:
  - **IMDb Pro Rankings**: Popular movies tend to receive lower ratings.
  - **Color Films**: Lower ratings compared to black-and-white films.
  - **PG-13 Maturity Rating**: Slight negative impact.

---

## Repository Structure

- **Final_Code.R**: R script containing the analysis and modeling code.
- **IMDb Rating Prediction Report.pdf**: Detailed report summarizing the project, methodology, and results.
- **data/**: Folder containing the datasets and supporting files:
  - **Train_Data.csv**: Training dataset for the model.
  - **Test_Data.csv.csv**: Testing dataset for validation.
  - **Data_Dictionary.csv**: Descriptions for dataset variables.



---

## Future Work

- Incorporate additional predictors such as social media engagement and sequel/remake indicators.
- Experiment with machine learning models like Random Forests or Gradient Boosting to enhance predictive power.
- Address data limitations such as inflation adjustments for budget and better categorization of directors/producers.

---

## Acknowledgments

This project was completed as part of the MGSC661: Multivariate Statistics course at McGill University, under the guidance of Professor Juan Camilo Serpa. This project is in collaboration with Alexandra Guion, Madeleine Dinh, Zaid Mahmood, Rajiha Mehdi.

