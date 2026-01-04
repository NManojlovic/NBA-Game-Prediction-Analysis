# NBA Game Prediction using Linear and Logistic Regression from Scratch

## Project Overview

This project is a practical demonstration of implementing fundamental machine learning algorithms from scratch. It uses Linear Regression and Logistic Regression with L2 Regularization to perform predictive analysis on a dataset of NBA games. The primary goal was to build these models from the ground up using NumPy, showcasing a deep understanding of their internal mechanics rather than relying on pre-built libraries.

The project focuses on two predictive tasks:
1.  **Regression:** Predicting the final total points of a game.
2.  **Classification:** Predicting if the total score will be "Over" or "Under" a predefined line.

## Dataset

The analysis was performed on the "NBA Games" dataset from Kaggle, which contains detailed statistics for over 26,000 games from 2004 to 2020. The key features used include points, field goal percentages, assists, and rebounds for both home and away teams.

-   **Source:** [Kaggle: NBA Games](https://www.kaggle.com/datasets/nathanlauga/nba-games)

## Methodology

### 1. Data Preprocessing
The initial data preparation involved several key steps. Missing values were handled by imputing the mean of each respective column. The target variables, `total_points` and `over_under`, were engineered from the base stats. Finally, the data was split into an 80% training set and a 20% test set, and all features were scaled using `StandardScaler`. Care was taken to fit the scaler only on the training data to prevent data leakage.

### 2. "From Scratch" Model Implementation
The core of this project is the custom implementation of the learning algorithms. I wrote Python functions for all essential components, including a function for creating polynomial features to capture non-linear patterns. The cost functions—Mean Squared Error (MSE) for regression and Binary Cross-Entropy (BCE) with L2 regularization for classification—were also implemented from scratch, along with a custom Gradient Descent optimizer to train the models.

### 3. Model Evaluation
The models were trained on the training set and their performance was evaluated on the unseen test set using standard metrics. For the regression model, these included MSE, RMSE, and R-squared (R²). The classification model was evaluated using Accuracy, F1-score, and a Confusion Matrix. To ensure the results were robust, K-Fold Cross-Validation (with K=5) was also performed.

## Results

Both models demonstrated near-perfect performance, which served as a successful validation of the implementation.

The **Regression Model** achieved an **R² of 1.00** and an **RMSE of approximately 1.46 points**. The **Classification Model** achieved an **F1-score of 0.95** and **97% accuracy**.

**Important Note:** These high scores are a direct result of the experimental design, where the models had access to post-game statistics to predict an outcome derived from them. The primary goal was to **validate the correctness of the from-scratch algorithms**, not to create a real-world prediction tool.

## Technologies Used
-   Python
-   Pandas & NumPy
-   Matplotlib & Seaborn
-   Jupyter Notebook