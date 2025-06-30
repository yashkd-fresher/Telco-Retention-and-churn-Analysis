
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration with custom theme
st.set_page_config(
    page_title="📊 Telecom Customer Churn Analysis",
    page_icon="📈",
    layout="wide"
)

# Apply custom CSS styles
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #fafafa;
        }
        .header {
            font-size: 2em;
            font-weight: bold;
            color: #2a7d9f;
            margin-bottom: 20px;
        }
        .subheader {
            font-size: 1.5em;
            font-weight: bold;
            color: #4c8c8c;
            margin-top: 20px;
        }
        .section-header {
            font-size: 1.25em;
            font-weight: bold;
            color: #4b4b4b;
            margin-top: 30px;
        }
        .stButton button {
            background-color: #0073e6;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px 20px;
            margin-top: 20px;
        }
        .stButton button:hover {
            background-color: #005bb5;
        }
        .stDataFrame {
            background-color: #ffffff;
        }
        .stSelectbox select {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Page title and description
st.title("📊 Telecom Customer Churn Analysis Dashboard")
st.markdown("""
This dashboard analyzes customer churn patterns in a telecom dataset, helping to identify 
key factors that contribute to customer retention or attrition. Explore various sections to get deeper insights.
""")

# Functions for data processing and model training
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    df_processed = df.copy()
    df_processed.drop(columns=['customerID'], inplace=True, errors='ignore')
    if 'Churn' in df_processed.columns:
        df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols].values)
    
    return df_processed, label_encoders, scaler

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, accuracy, conf_matrix, report, X_test, y_test, y_pred

def get_feature_importance(model, X):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    return feature_importance

# Load the data
data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "EDA & Visualizations", "Model Training & Evaluation", "Churn Prediction"])

# Data Overview page
if page == "Data Overview":
    st.header("Dataset Overview")
    st.subheader("Sample Data")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rows:** {data.shape[0]}")
        st.write(f"**Columns:** {data.shape[1]}")
        st.write(f"**Churn Rate:** {data['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")
        
    with col2:
        st.write("**Missing Values:**")
        missing_values = data.isnull().sum()
        st.write(missing_values[missing_values > 0] if len(missing_values[missing_values > 0]) > 0 else "No missing values")
    
    st.subheader("Summary Statistics")
    st.dataframe(data.describe(), use_container_width=True)
    
    st.subheader("Column Information")
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.write(f"**Categorical Columns:** {', '.join(categorical_columns)}")
    st.write(f"**Numerical Columns:** {', '.join(numerical_columns)}")

# EDA & Visualizations page
elif page == "EDA & Visualizations":
    st.header("Exploratory Data Analysis & Visualizations")
    
    st.subheader("Churn Distribution")
    fig = px.pie(data, names='Churn', title='Churn Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig)
    
    st.subheader("Categorical Features vs Churn")
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    selected_cat_col = st.selectbox("Select Categorical Feature", categorical_cols)
    churn_percentage = data.groupby([selected_cat_col, 'Churn']).size().unstack().fillna(0)
    churn_percentage = churn_percentage.div(churn_percentage.sum(axis=1), axis=0) * 100
    
    fig = px.bar(
        churn_percentage.reset_index().melt(id_vars=selected_cat_col),
        x=selected_cat_col,
        y='value',
        color='Churn',
        barmode='group',
        title=f'{selected_cat_col} vs Churn Rate (%)',
        labels={'value': 'Percentage (%)'}
    )
    st.plotly_chart(fig)
    
    st.subheader("Numerical Features vs Churn")
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    selected_num_col = st.selectbox("Select Numerical Feature", numerical_cols)
    
    fig = px.histogram(data, x=selected_num_col, color='Churn', marginal='box', opacity=0.7, barmode='overlay',
                       title=f'Distribution of {selected_num_col} by Churn Status', labels={selected_num_col: selected_num_col})
    st.plotly_chart(fig)
    
    st.subheader("Correlation Matrix of Numerical Features")
    data_corr = data.copy()
    data_corr['Churn'] = data_corr['Churn'].map({'Yes': 1, 'No': 0})
    corr_matrix = data_corr[numerical_cols + ['Churn']].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Matrix of Numerical Features', aspect="auto")
    st.plotly_chart(fig)

# Model Training & Evaluation page
elif page == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    processed_data, label_encoders, scaler = preprocess_data(data)
    X = processed_data.drop('Churn', axis=1)
    y = processed_data['Churn']
    
    train_button = st.button("Train XGBoost Model")
    if train_button:
        with st.spinner('Training model...'):
            model, accuracy, conf_matrix, report, X_test, y_test, y_pred = train_model(X, y)
            model_objects = {'model': model, 'label_encoders': label_encoders, 'scaler': scaler, 'feature_names': X.columns.tolist()}
            with open('churn_model_objects.pkl', 'wb') as f:
                pickle.dump(model_objects, f)
            st.success('Model trained successfully and saved as "churn_model_objects.pkl"!')
            
            st.subheader("Confusion Matrix")
            conf_fig = px.imshow(
                conf_matrix,
                text_auto=True,
                x=['Predicted No Churn', 'Predicted Churn'],
                y=['Actual No Churn', 'Actual Churn'],
                color_continuous_scale='Blues',
                title='Confusion Matrix'
            )
            st.plotly_chart(conf_fig)
            
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            st.subheader("Feature Importance")
            feature_importance = get_feature_importance(model, X)
            fig = px.bar(feature_importance.head(15), x='Importance', y='Feature', orientation='h',
                         title='Top 15 Features by Importance', color='Importance')
            st.plotly_chart(fig)
    else:
        st.info("Click the button to train the XGBoost model")

# Churn Prediction page
elif page == "Churn Prediction":
    st.header("Customer Churn Prediction")
    
    try:
        with open('churn_model_objects.pkl', 'rb') as f:
            model_objects = pickle.load(f)
        
        model = model_objects['model']
        label_encoders = model_objects['label_encoders']
        scaler = model_objects['scaler']
        feature_names = model_objects['feature_names']
        
        st.success("Model loaded successfully!")
        
        st.subheader("Enter Customer Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
        
        with col2:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        col4, col5 = st.columns(2)
        with col4:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col5:
            monthly_charges = st.slider("Monthly Charges ($)", 10.0, 120.0, 50.0)
            total_charges = st.slider("Total Charges ($)", 10.0, 9000.0, monthly_charges * tenure)
        
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([input_data])
        
        if st.button("Predict Churn Probability"):
            input_processed = input_df.copy()
            for col, le in label_encoders.items():
                if col in input_processed.columns:
                    try:
                        input_processed[col] = le.transform(input_processed[col])
                    except ValueError:
                        input_processed[col] = le.transform([le.classes_[0]])[0]
            
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            input_processed[numerical_cols] = scaler.transform(input_processed[numerical_cols])
            input_processed = input_processed[feature_names]
            
            churn_probability = model.predict_proba(input_processed)[0, 1]
            churn_prediction = "Yes" if churn_probability > 0.5 else "No"
            
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_probability * 100,
                    title={'text': "Churn Probability"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 30], 'color': "green"}, {'range': [30, 70], 'color': "yellow"}, {'range': [70, 100], 'color': "red"}]},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                st.plotly_chart(fig)
            
            with col2:
                st.markdown(f"""
                ### Prediction Summary
                - **Churn Prediction:** {churn_prediction}
                - **Churn Probability:** {churn_probability:.2%}
                - **Customer Risk Level:** {'High' if churn_probability > 0.7 else 'Medium' if churn_probability > 0.3 else 'Low'}
                """)
            
            st.subheader("Features Influencing This Prediction")
            
            shap_values = model.get_booster().predict(xgb.DMatrix(input_processed), pred_contribs=True)
            shap_values = shap_values[0, :-1]
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(shap_values),
                'Impact': shap_values
            }).sort_values(by='Importance', ascending=False)
            
            st.dataframe(shap_df)
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
