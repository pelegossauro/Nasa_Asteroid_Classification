# Nasa_Asteroid_Classification
Overview
This project analyzes a dataset containing information about near-Earth objects (NEOs) provided by NASA. The dataset includes various attributes such as the size, velocity, orbital characteristics, and whether an object is considered hazardous. The goal is to clean, preprocess, and explore the data, ultimately performing a target variable analysis to predict the hazardous nature of NEOs.

Dataset Description
The dataset (nasa.csv) consists of 4687 rows and 40 columns, with each row representing a near-Earth object (NEO). The columns include both numerical and categorical data about the NEOs' physical and orbital characteristics, such as:

Absolute Magnitude: A measure of the brightness of the NEO.
Est Dia in KM(min), Est Dia in KM(max): Estimated diameter of the NEO in kilometers (min and max values).
Relative Velocity km per sec: The relative velocity of the NEO in kilometers per second.
Miss Dist.(Astronomical): The closest approach distance to Earth in astronomical units (AU).
Hazardous: A boolean indicating whether the NEO is considered hazardous.
The target variable for this analysis is the Hazardous column, which is encoded as a boolean (True or False).

Steps Involved
1. Data Loading
The dataset is loaded using Pandas read_csv function.

python
Copiar código
import pandas as pd
df = pd.read_csv('nasa.csv')
df.head()
2. Data Exploration and Basic Information
The dataset is explored using basic commands like df.info() to check for null values and the data types of each column.

python
Copiar código
df.info()
This provides an overview of the 4687 rows and 40 columns, with a mix of numerical, boolean, and object types.

3. Dropping Irrelevant Columns
The dataset contains columns that do not contribute to the analysis, such as Neo Reference ID, Name, Orbit ID, and columns related to time and identification (Close Approach Date, Orbit Determination Date). These are dropped to simplify the dataset.

python
Copiar código
df = df.drop(['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date', 'Epoch Date Close Approach', 'Orbit Determination Date'], axis=1, errors='ignore')
4. Encoding Categorical Variables
The target variable Hazardous is a boolean column. For machine learning tasks, this column is encoded using one-hot encoding into two columns: False and True.

python
Copiar código
hazardous_labels = pd.get_dummies(df['Hazardous'])
df = pd.concat([df, hazardous_labels], axis=1)
df = df.drop(['Hazardous'], axis=1)
5. Dropping Redundant Columns
After analyzing correlations, several columns with high correlation (such as duplicate columns with different units) are removed to reduce redundancy.

python
Copiar código
df = df.drop(['Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 
              'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 
              'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 
              'Miss Dist.(kilometers)', 'Miss Dist.(miles)'], axis=1)
6. Dropping Columns with Single Unique Values
The Orbiting Body and Equinox columns have only one unique value (Earth and J2000, respectively), so these are dropped.

python
Copiar código
df.drop(['Orbiting Body', 'Equinox'], axis=1, inplace=True)
7. Visualizing Correlations
A heatmap of the correlation matrix is plotted to visualize relationships between the numerical features. This helps in identifying redundant features and understanding the patterns in the data.

python
Copiar código
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
8. Final Dataframe
After cleaning and preprocessing, the dataset is ready for further analysis or machine learning tasks. The resulting dataset includes important features like Absolute Magnitude, Est Dia in KM(min), Relative Velocity km per sec, and more, with a one-hot encoded target column (False and True).

Conclusion
This project performs a comprehensive preprocessing of the NASA Near-Earth Object dataset, which includes data cleaning, feature engineering, and visualization. The dataset is now ready for further machine learning model training and evaluation to predict whether a NEO is hazardous based on its attributes.

Dependencies
To run this script, the following Python libraries are required:

pandas
numpy
matplotlib
seaborn
You can install the dependencies using pip:

bash
Copiar código
pip install pandas numpy matplotlib seaborn
Future Improvements
Implement machine learning models to predict the hazardous nature of NEOs.
Experiment with feature scaling and dimensionality reduction techniques.
Perform cross-validation to evaluate model performance.
License
This project is licensed under the MIT License - see the LICENSE file for details.



