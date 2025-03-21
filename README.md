# IMDb Movies India - Data Analysis & Prediction

## 📌 Project Overview
This project analyzes IMDb movie data from India to extract insights about directors, genres, and movie ratings. Additionally, a predictive model is built using **Linear Regression** and **Random Forest Regressor** to estimate movie ratings based on various features.

## 📂 Dataset
The dataset used is **IMDb Movies India.csv**, containing information about Indian movies, including:
- **Title**
- **Year**
- **Genre**
- **Director**
- **Actors**
- **Duration**
- **Votes**
- **Rating**

## 🔧 Approach
1. **Data Preprocessing**: Cleaning and transforming raw data.
2. **Feature Engineering**: Encoding categorical variables.
3. **Exploratory Data Analysis (EDA)**: Understanding trends in ratings based on directors and genres.
4. **Model Building**: Using **Linear Regression** and **Random Forest Regressor** to predict ratings.
5. **Model Evaluation**: Comparing performance using MSE, MAE, and R² Score.

---

## 📊 Data Preprocessing
The preprocessing steps include:

- **Handling Missing Values**:
  - Missing ratings replaced with the most common rating.
  - Missing duration values replaced with the average duration.
  - Missing votes replaced with the mean votes.

- **Cleaning & Transforming Columns**:
  - Extracted numeric part from `Year`.
  - Converted `Duration` from string to numeric format.
  - Converted `Votes` to numeric format (handling special cases like `$5.16M`).

- **Encoding Categorical Features**:
  - Actors, Directors, and Genre were encoded based on their mean ratings.
  
- **Feature Selection**:
  - `Year`, `Duration`, `Votes`, encoded actor/director/genre ratings were selected for model training.

---

## 📈 Exploratory Data Analysis (EDA)
EDA focused on understanding movie trends:
- **Director Success Rate:** The average ratings of movies directed by various filmmakers.
- **Genre-Based Ratings:** Identifying which genres receive the highest ratings.
- **Correlation Analysis:** Examining relationships between features using Spearman correlation.

---

## 🤖 Model Building & Evaluation
### 🎯 Models Used:
1. **Linear Regression**
2. **Random Forest Regressor**

### 📊 Evaluation Metrics:
The models were evaluated using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

### 📢 Results:
- **Random Forest Regressor** outperformed **Linear Regression** in predicting ratings with a higher R² score and lower error metrics.

---

## 🚀 How to Run the Project
### 📌 Prerequisites
- Python 3.x
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

### 🔧 Steps to Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/imdb-movie-analysis.git
   ```
2. Navigate to the project folder:
   ```bash
   cd imdb-movie-analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python analysis.py
   ```

---


📌 **Author:** Muhammad Aftab Khan  


