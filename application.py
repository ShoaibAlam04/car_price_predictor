import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Load the data
@st.cache_data
def load_data():
    car = pd.read_csv('Cleaned_Car_data.csv')
    car = car[car['Price'] < 6000000]
    return car

car = load_data()

# Model training
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)


st.title('Car Price Prediction Dashboard')

# Upload own data
uploaded_file = st.file_uploader("Upload your own data", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

# Display dataset
if st.checkbox('Show Dataset'):
    st.write(car)

# Display statistics
if st.checkbox('Show Data Statistics'):
    st.write(car.describe())

# Display Correlation Matrix
if st.checkbox('Show Correlation Matrix'):
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    sns.heatmap(car.corr(), annot=True, ax=ax3)
    st.pyplot(fig3)  # Pass the figure explicitly

# Display visualizations
if st.checkbox('Show Visualizations'):
    st.subheader('Company vs Price')
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    sns.boxplot(x='company', y='Price', data=car, ax=ax1)
    st.pyplot(fig1)  # Pass the figure explicitly

    st.subheader('Year vs Price')
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    sns.swarmplot(x='year', y='Price', data=car, ax=ax2)
    st.pyplot(fig2)  # Pass the figure explicitly




# Model performance
st.subheader(f'Model R2 Score: {r2:.2f}')



# User input
company = st.selectbox('Select Car Company', options=['Select'] + sorted(list(car['company'].unique())))

if company != 'Select':
    models = car[car['company'] == company]['name'].unique()
else:
    models = ['Select']

name = st.selectbox('Select Car Name', options=models)

year = st.slider('Select Year', int(car['year'].min()), int(car['year'].max()), int(car['year'].mean()))
kms_driven = st.number_input('Enter KMs Driven', int(car['kms_driven'].min()), int(car['kms_driven'].max()), int(car['kms_driven'].mean()))
fuel_type = st.selectbox('Select Fuel Type', options=['Select'] + list(car['fuel_type'].unique()))



# Reset and Predict Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button('Predict Price'):
        if name != 'Select' and company != 'Select' and fuel_type != 'Select':
            input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
            pred = pipe.predict(input_df)
            st.success(f'Estimated Price is {pred[0]:.2f}')
        else:
            st.warning('Please complete all selections before predicting.')

with col2:
    if st.button('Reset'):
        name = 'Select'
        company = 'Select'
        year = int(car['year'].mean())
        kms_driven = int(car['kms_driven'].mean())
        fuel_type = 'Select'
# CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Prediction
if name != 'Select' and company != 'Select' and fuel_type != 'Select':
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    if st.button('Predict Price'):
        pred = pipe.predict(input_df)
        st.success(f'Estimated Price is {pred[0]:.2f}')
