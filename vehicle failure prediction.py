'''


# Business Problem Statement:

# Unexpected vehicle breakdowns are causing significant financial losses and operational disruptions,
affecting transportation schedules and increasing maintenance costs

# `CRISP-ML(Q)` process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''
# Objective(s): Minimize unexpected vehicle breakdowns
# Constraints: Minimize Expenses

'''Success Criteria'''

# Business Success Criteria: Reduce the Unexpected failure of vehicles by atleast 30%
# ML Success Criteria: Build a machine learning model with 95% accuracy in predicting potential vehicle breakdown
# Economic Success Criteria: Achieve a cost saving of atleast $2M per year due to reduction of unexpected breakdowns

'''
# ## Data Collection
#Data
The dataset contains 19535 rows and 7 columns
The dataset could include various features and measurements related to the engine health of vehicles, 
such as engine RPM, temperature, pressure, and other sensor data.
'''
# import required packages
#pip install lazypredict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from sqlalchemy import create_engine
#pip install numpy pandas scipy
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import seaborn as sns
# Load your dataset (replace 'your_dataset.csv' with the actual file path)
# load data
data = pd.read_csv(r"C:\Users\srini\Downloads\PROJECT DATASETS\engine_data.csv")
     
# Creating engine which connect to MySQL
user = 'root' # user name
pw = 'pavani' # password
db = 'Engine' # database

# creating engine to connect database
#pip install pymysql
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
data.to_sql('engine_data', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from engine_data'

datadf = pd.read_sql_query(sql, con = engine)

print(datadf)
# Data Preprocessing & EDA

data.info()

data.head()

data.describe()
# calculate moments for each column

df = pd.DataFrame(data)

# Calculate moments for each column
numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    print(f"Column: {col}")
    print(f"1st Moment (Mean): {np.mean(df[col]):.2f}")
    print(f"1st Moment (Median): {np.median(df[col]):.2f}")
    print(f"1st Moment (Mode): {stats.mode(df[col]).mode:.2f}")
    print(f"2nd Moment (Variance): {np.var(df[col]):.2f}")
    print(f"2nd Moment (Standard deviation): {np.std(df[col]):.2f}")
    print(f"3rd Moment (Skewness): {stats.skew(df[col]):.2f}")
    print(f"4th Moment (Kurtosis): {stats.kurtosis(df[col]):.2f}\n")
    
#Visualising the Distribution of all the Attributes

sns.pairplot(data, hue="Engine Condition")
sns.jointplot(data, x="Fuel pressure", y="Engine rpm", hue="Engine Condition")   
sns.displot(data, x=data["Engine rpm"] * data["lub oil temp"], hue="Engine Condition") 
sns.displot(data, x="Coolant temp", hue="Engine Condition")
sns.displot(data, x="Coolant pressure", hue="Engine Condition")
sns.displot(data, x=data["Coolant pressure"] * data["Coolant temp"], hue="Engine Condition")
sns.jointplot(data, x="Lub oil pressure", y="lub oil temp", hue="Engine Condition")

#auto eda
#pip install ydata-profiling
from pandas_profiling import ProfileReport

# Load your dataset 
data = pd.read_csv(r"C:\Users\srini\Downloads\PROJECT DATASETS\engine_data.csv")

# Generate the report
profile = ProfileReport(data)
profile.to_file("eda_report.html")

import sweetviz as sv

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv(r"C:\Users\srini\Downloads\PROJECT DATASETS\engine_data.csv")

# Generate the EDA report
report = sv.analyze(data)

# Show the report in an HTML format (opens in a new tab)
report.show_html('eda_report.html')
# autoviz


from autoviz.AutoViz_Class import AutoViz_Class

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('C:/Users/srini/Downloads/PROJECT DATASETS/engine_data.csv')

# Create an instance of AutoViz
av = AutoViz_Class()

# Generate data visualizations automatically
av.AutoViz(filename='C:/Users/srini/Downloads/PROJECT DATASETS/engine_data.csv', sep=',',
           depVar='', dfte=None, header=0, verbose=0, lowess=False,
           chart_format='svg', max_rows_analyzed=19535, max_cols_analyzed=30)

#dtale

import dtale

# Load your dataset
data = pd.read_csv('C:/Users/srini/Downloads/PROJECT DATASETS/engine_data.csv')

# Create a D-Tale instance
d = dtale.show(data)

# Open the D-Tale UI in a browser
d.open_browser()
#Find and remove duplicates
data.drop_duplicates(inplace=True)

# missing values
missing_values = df.isnull()
print(missing_values)

#Engine condition analysis

data['Engine Condition'].describe()

#Correlation

corr = data.corr()
corr
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(corr, annot=True)
plt.title('Correlation', fontsize=18);

# Separate features and target variable
X = data.drop(columns=['Engine Condition'])
y = data['Engine Condition']

# Define preprocessing steps
numeric_features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure',
                    'lub oil temp', 'Coolant temp']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
# Save the preprocessor (ColumnTransformer) pipeline
joblib.dump(preprocessor, 'preprocessor.pkl')

# Save the cleaned data using joblib
joblib.dump(data, 'cleaned_data.pkl')
print("Cleaned data saved successfully!")
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lazypredict
from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import train_test_split


import joblib
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load cleaned data (assuming you've already loaded it)
# Replace 'cleaned_data.pkl' with the actual path to your cleaned data
data = joblib.load('cleaned_data.pkl')

# Separate features and target variable
X = data.drop(columns=['Engine Condition'])
y = data['Engine Condition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

# Initialize the LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit the model
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
# Print the model performance
print(models)
# Print the model performance
for model_name, model in models.items():
    print(f"Model: {model_name}")
    if 'accuracy' in model:
        print(f"Accuracy: {model['accuracy']:.2f}")
    if 'balanced accuracy' in model:
        print(f"Balanced Accuracy: {model['balanced accuracy']:.2f}")
    if 'roc auc' in model:
        print(f"ROC AUC: {model['roc auc']:.2f}")
    if 'f1 score' in model:
        print(f"F1 Score: {model['f1 score']:.2f}\n")

    # Save the trained model with metrics using joblib
    joblib.dump(model, f'{model_name}.pkl')


###Models
#1)NearestCentroid
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize the Nearest Centroid model
model = NearestCentroid()

# Define hyperparameter distributions
param_dist = {
    'metric': ['euclidean', 'manhattan', 'minkowski'],
}

# Perform randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                   n_iter=3, cv=5, scoring='accuracy', random_state=123, n_jobs=-1)
random_search.fit(X_train, y_train.values.ravel())

# Get the best model
best_nc_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_nc_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {random_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_nc_model, 'best_nearest_centroid_model.joblib')

#2)LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize the Logistic Regression model
model = LogisticRegression()

# Define hyperparameter distributions
param_dist = {
    'C': uniform(loc=0.1, scale=10),  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga']  # Optimization algorithm
}

# Perform randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                   n_iter=3, cv=5, scoring='accuracy', random_state=123, n_jobs=-1)
random_search.fit(X_train, y_train.values.ravel())

# Get the best model
best_lr_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_lr_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {random_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_lr_model, 'best_logistic_regression_model.joblib')


#3) GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize Gaussian Naive Bayes model
gnb = GaussianNB()

# Define hyperparameter distributions
param_dist = {
    'var_smoothing': uniform(loc=1e-9, scale=1e-8)  # Smoothing parameter for variance
}

# Perform randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=gnb, param_distributions=param_dist,
                                   n_iter=3, cv=5, scoring='accuracy', random_state=123, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best model
best_gnb_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_gnb_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {random_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_gnb_model, 'best_gaussian_nb_model.joblib')

#4)AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize AdaBoost Classifier
adaboost = AdaBoostClassifier()

# Define hyperparameter distributions
param_dist = {
    'n_estimators': [50, 100, 200],  # Number of base estimators
    'learning_rate': [0.01, 0.1, 1.0]  # Learning rate
}

# Perform randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=adaboost, param_distributions=param_dist,
                                   n_iter=3, cv=5, scoring='accuracy', random_state=123, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best model
best_adaboost_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_adaboost_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {random_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_adaboost_model, 'best_adaboost_model.joblib')

#5)DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Define hyperparameter distributions
param_dist = {
    'criterion': ['gini', 'entropy'],  # Quality of split
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when splitting
}

# Perform randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=dt_classifier, param_distributions=param_dist,
                                   n_iter=3, cv=5, scoring='accuracy', random_state=123, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best model
best_dt_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_dt_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {random_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_dt_model, 'best_decision_tree_model.joblib')

#6)SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize SGDClassifier
sgd_classifier = SGDClassifier()

# Define hyperparameter distributions
param_dist = {
    'loss': ['hinge', 'log', 'modified_huber'],  # Loss function
    'penalty': ['l2', 'l1', 'elasticnet'],  # Regularization type
    'alpha': uniform(loc=1e-5, scale=1e-4)  # Regularization strength
}

# Perform randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=sgd_classifier, param_distributions=param_dist,
                                   n_iter=3, cv=5, scoring='accuracy', random_state=123, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best model
best_sgd_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_sgd_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {random_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_sgd_model, 'best_sgd_classifier_model.joblib')

#7)ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize Extra Trees Classifier
extra_trees = ExtraTreesClassifier()

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required to be at a leaf node
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=extra_trees, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_extra_trees_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_extra_trees_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_extra_trees_model, 'best_extra_trees_model.joblib')

#8)LinearSVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize LinearSVC
linear_svc = LinearSVC()

# Define hyperparameters for grid search
param_grid = {
    'penalty': ['l2'],  # Specifies the norm used in the penalization
    'loss': ['squared_hinge'],  # Specifies the loss function
    'tol': [1e-4],  # Tolerance for stopping criteria
    'C': [0.1, 1.0, 10.0]  # Regularization parameter
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=linear_svc, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_linear_svc_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_linear_svc_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_linear_svc_model, 'best_linear_svc_model.joblib')

#9) RidgeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize Ridge Classifier
ridge_classifier = RidgeClassifier()

# Define hyperparameters for grid search
param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Regularization strength
    'fit_intercept': [True, False],  # Whether to calculate the intercept
    'tol': [1e-4],  # Tolerance for stopping criteria
    'class_weight': [None, 'balanced']  # Weights associated with classes
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=ridge_classifier, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_ridge_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_ridge_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_ridge_model, 'best_ridge_classifier_model.joblib')

#10)KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Initialize KNN Classifier
knn_classifier = KNeighborsClassifier()

# Define hyperparameters for grid search
param_grid = {
    'n_neighbors': [1, 5, 10],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'p': [1, 2]  # Power parameter for the Minkowski metric
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_knn_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_knn_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the best model using Joblib
joblib.dump(best_knn_model, 'best_knn_classifier_model.joblib')
##google colab
#automl
#TPOT classifier
!pip install tpot
# Import necessary libraries
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('/content/engine_data.csv')

# Separate features and target variable
X = data.drop(columns=['Engine Condition'])
y = data['Engine Condition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

# Initialize and train TPOT
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=123)
tpot.fit(X_train, y_train)

# Evaluate performance on test data
accuracy = tpot.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

##2)H2o
# Install H2O
!pip install h2o

# Import necessary libraries
import h2o
from h2o.automl import H2OAutoML

# Load your data into H2O
h2o_df = h2o.H2OFrame(data)  # Assuming 'data' is your DataFrame

# Set predictors and response column
predictors = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]
response = "Engine Condition"

# Split data
train, valid = h2o_df.split_frame(ratios=[0.8], seed=1234)

# Initialize AutoML
aml = H2OAutoML(max_models=5, seed=1234)
aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

# Get the leaderboard
leaderboard = aml.leaderboard
print(leaderboard)

