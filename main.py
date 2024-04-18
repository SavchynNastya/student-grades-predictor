import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class StudentDataVisualizer:
    @staticmethod
    def visualize_student_grades_by_parental_occupations(df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Mjob', y='G3', hue='Fjob', data=df)
        plt.title('Student Grades by Parental Occupations')
        plt.xlabel('Mother Occupation')
        plt.ylabel('Student Grade (G3)')
        plt.xticks(rotation=45)
        plt.legend(title='Father Occupation')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_student_grades_by_parental_occupations_and_sex(df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Mjob', y='G3', hue='sex', data=df)
        plt.title('Student Grades by Parental Occupations')
        plt.xlabel('Mother Occupation')
        plt.ylabel('Student Grade (G3)')
        plt.xticks(rotation=45)
        plt.legend(title='Sex')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def train_random_forest_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"Mean Squared Error": mse, "Mean Absolute Error": mae, "R-squared": r2}

    @staticmethod
    def visualize_predicted_vs_actual_grades(y_test: pd.Series, y_pred: Union[list, pd.Series]) -> None:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)  # Real vs. Predicted
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
        plt.xlabel('Real Grades')
        plt.ylabel('Predicted Grades')
        plt.title('Real vs. Predicted Grades')
        plt.show()


csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_data.csv')
df = pd.read_csv(csv_file_path)

# Remove rows with zero values in grade sections
df = df[(df['G1'] > 0) & (df['G2'] > 0) & (df['G3'] > 0)]

# Visualize student grades by parental occupations
StudentDataVisualizer.visualize_student_grades_by_parental_occupations(df)

# Visualize student grades by parental occupations and sex
StudentDataVisualizer.visualize_student_grades_by_parental_occupations_and_sex(df)

# Preprocess the data
# Fill missing values for numeric columns with median
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing values for categorical columns with mode
categorical_cols = df.select_dtypes(exclude='number').columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Split the dataset into features (X) and target variable (y)
X = df.drop('G3', axis=1)
y = df['G3']

# Train the model
model = StudentDataVisualizer.train_random_forest_model(X, y)

# Evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
evaluation_results = StudentDataVisualizer.evaluate_model(model, X_test, y_test)
for metric, value in evaluation_results.items():
    print(f"{metric}: {value}")

# Make predictions
y_pred = model.predict(X_test)

# Visualize the predicted vs. actual grades
StudentDataVisualizer.visualize_predicted_vs_actual_grades(y_test, y_pred)
