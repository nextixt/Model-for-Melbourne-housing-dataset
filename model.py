import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# reading dataset
data = pd.read_csv('train.csv')

# defining X and y
X = data.drop('SalePrice', axis = 1)
y = data['SalePrice']

# defining X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# defining numerical_cols and categorical_cols for ColumnTransformer
# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train.columns if
                    X_train[cname].nunique() < 10 and 
                    X_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# defining pipeline
pipeline = Pipeline(steps = [
    ('transformer', preprocessor),
    ('model', XGBRegressor(n_estimators = 300))
    
])

# model fitting
pipeline.fit(X_train, y_train)

# make predictions
preds = pipeline.predict(X_test)

# counting MAE
MAE = mean_absolute_error(y_test, preds)

# printing results
print(f'MAE: {MAE}')
