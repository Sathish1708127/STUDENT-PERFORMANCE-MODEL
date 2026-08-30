import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the dataset
df = pd.read_csv('StudentsPerformance.csv')

# 2. Define features (X) and target (y)
# We will predict 'math score' based on categorical features
X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
y = df['math score']

# 3. Create a preprocessing pipeline
# This automatically converts text categories (like "standard lunch") into numbers (0 or 1)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.columns)
    ])

# 4. Bundle preprocessing and model into a single pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 6. Save the trained pipeline to a file
joblib.dump(model, 'student_model.pkl')
print("Model trained and saved as student_model.pkl!")