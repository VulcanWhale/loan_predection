import streamlit as st
import pandas as pd
import numpy as np
import get_data
import wash_data
import matplotlib.pyplot as plt
from xgboost import XGBModel, XGBClassifier
from sklearn.model_selection import train_test_split



#This section is to wash the data, making it more convenient for EDA. And I use some build-in functions to pad the nan value of data we chose.
def is_graduate(x):
    if x=='Graduate':
        return 1
    else:
        return 0
def is_female(x):
    if x=='Female':
        return 1
    else:
        return 0
def is_married(x):
    if x=='Yes':
        return 1
    else:
        return 0
def is_urban(x):
    if x=='Urban':
        return 1
    else:
        return 0
def is_self_employed(x):
    if x=='Yes':
        return 1
    else:
        return 0    
def Loan_Status_(x):
    if x=='Y':
        return 1
    else:
        return 0
    
def wash_data():
    HomeLoansApproval = pd.read_csv(r"C:\Users\USER\Downloads\Home_Loan_prediction\loan_sanction_train.csv")
    # Drop duplicates
    HomeLoansApproval = HomeLoansApproval.drop_duplicates(subset=['Loan_ID'])

    # Handle missing values (consider using more sophisticated techniques if needed)
    HomeLoansApproval.dropna(axis=0, how='any', subset=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History'], inplace=True)
    HomeLoansApproval_mean = HomeLoansApproval['LoanAmount'].fillna(value=HomeLoansApproval['LoanAmount'].mean(), inplace=False)
    HomeLoansApproval_median = HomeLoansApproval['Loan_Amount_Term'].fillna(HomeLoansApproval['Loan_Amount_Term'].median(), inplace=False)

    # Update columns after filling missing values
    HomeLoansApproval = HomeLoansApproval.drop(labels=['LoanAmount', 'Loan_Amount_Term'], axis=1)
    HomeLoansApproval['LoanAmount'] = HomeLoansApproval_mean
    HomeLoansApproval['Loan_Amount_Term'] = HomeLoansApproval_median

    # Create new features from categorical columns
    HomeLoansApproval['Is_graduate'] = HomeLoansApproval['Education'].apply(lambda x: is_graduate(x))
    HomeLoansApproval['Is_Female'] = HomeLoansApproval['Gender'].apply(lambda x: is_female(x))
    HomeLoansApproval['Is_married'] = HomeLoansApproval['Married'].apply(lambda x: is_married(x))
    HomeLoansApproval['Is_urban'] = HomeLoansApproval['Property_Area'].apply(lambda x: is_urban(x))
    HomeLoansApproval['Is_self_employed'] = HomeLoansApproval['Self_Employed'].apply(lambda x: is_self_employed(x))
    HomeLoansApproval['Loan_Status_'] = HomeLoansApproval['Loan_Status'].apply(lambda x: Loan_Status_(x))

    Loan_Status = HomeLoansApproval['Loan_Status_']
    HomeLoansApproval = HomeLoansApproval.drop(['Education', 'Gender', 'Married', 'Property_Area', 'Self_Employed', 'Loan_Status', 'Loan_Status_'], axis=1)
    HomeLoansApproval['Loan_Status'] = Loan_Status

    # Handle 'Dependents' with categories
    HomeLoansApproval['Dependents'] = HomeLoansApproval['Dependents'].apply(lambda x: ((0 if x == '0' else 1) if x != '2' else 2) if x != '3+' else 3)

    return HomeLoansApproval




def prepare_data():
    df = wash_data()
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_xgb_model(X_train, y_train):
    X_train = X_train.drop('Loan_ID', axis=1)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, enable_categorical=True)
    model.fit(X_train, y_train)
    return model

    
#This section is to make a function that can connect the parameters of widgets of app to our dataset and get the selected data. 
def get_all_data():
    return wash_data()


def select_data(size=1,is_graduate=None,is_married=None,is_female=None,is_self_employed=None,is_urban=None,credit_history=None):
    df=get_all_data()
    df=df.head(int(len(df)*size))
    df=df[df.columns if is_graduate==None else df['Is_graduate']==is_graduate]
    df=df[df.columns if is_female==None else df['Is_Female']==is_female]
    df=df[df.columns if is_self_employed==None else df['Is_self_employed']==is_self_employed]
    df=df[df.columns if is_married==None else df['Is_married']==is_married]
    df=df[df.columns if is_urban==None else df['Is_urban']==is_urban]
    df=df[df.columns if credit_history==None else df['Credit_History']==credit_history]
    return df

def select_Loan_Status(x=None):
    df=get_all_data()
    df=df[df.columns if x==None else df['Loan_Status']==x]
    return df

def select_Loan_Status(x=None):
    df=get_all_data()
    df=df[df.columns if x==None else df['Loan_Status']==x]
    return df


df = pd.read_csv(r"C:\Users\USER\Downloads\Home_Loan_prediction\loan_sanction_train.csv")



def prepare_features_from_user_input(user_input):
    features = []
    features.append(1 if user_input["Education"] == "Graduate" else 0)
    features.append(1 if user_input["is_married"] == "Married" else 0)
    features.append(int(user_input["Dependents"]))  # Convert to int
    features.append(1 if user_input["Self_Employed"] == "Yes" else 0)
    features.append(user_input["ApplicantIncome"])
    features.append(user_input["CoapplicantIncome"])
    features.append(user_input["LoanAmount"])
    features.append(user_input["Loan_Amount_Term"])
    features.append(int(user_input["Credit_History"]))  # Convert to int
    features.append(1 if user_input["Property_Area"] == "Urban" else 0)
    features.append(1 if user_input["Gender"] == "Female" else 0)
   
    return features


def page_question1(df, model):
    st.title("Question 1")
    st.header("According to the database, what is the probability of a successful loan based on the user's situation?")
    st.markdown("Please choose your situation")

    user_input = {}  # Empty dictionary to store user input
    user_input["Education"] = st.selectbox("Education Level", options=['Graduate', 'Non-Graduate'])
    user_input["is_married"] = st.selectbox("Marital Status", options=["Married", "Single"])
    user_input["Dependents"] = st.selectbox("Enter Number of Dependents", options=['0', '1', '2', '3+'])
    user_input["Self_Employed"] = st.selectbox("Self Employed", options=['No', 'Yes'])
    user_input["ApplicantIncome"] = st.number_input("Applicant Income")
    user_input["CoapplicantIncome"] = st.number_input("Coapplicant Income")
    user_input["LoanAmount"] = st.number_input("Loan Amount")
    user_input["Loan_Amount_Term"] = st.number_input("Loan Amount Term")
    user_input["Credit_History"] = st.selectbox("Credit History", options=['0', '1'])  # Assuming binary credit history
    user_input["Property_Area"] = st.selectbox("Property Area", options=['Rural', 'Semiurban', 'Urban'])
    user_input["Gender"] = st.selectbox("Gender", options=['Male', 'Female'])

    # Prepare features from user input
    user_features_encoded = prepare_features_from_user_input(user_input)

    # Make prediction
    prediction = model.predict_proba(np.array([user_features_encoded]))[:, 1]  # Assuming model outputs probabilities
    loan_approval_probability = prediction[0] * 100

    st.header(f"The probability of your loan being approved is: {loan_approval_probability:.2f}%")

    st.markdown("""The probability algorithm is based on the proportion of the successful number of databases to all eligible quantities. Therefore, due to the limitations of database data and the limited amount of data, the calculated results have limitations. This is only a reference for whether the applicant can successfully apply. The probability of reality varies greatly, please consider more based on individual circumstances.""")

    if st.button('More Information'):
        st.markdown('''As is shown by the result, we can observe trends and patterns in the loan success rate based on these income ranges. It is possible to see that as the applicant's income range increases, there may be a higher likelihood of loan approval. Similarly, as the co-applicant's income range increases, it may also positively impact the loan success rate. Furthermore, considering the combined effect of both the applicant's and co-applicant's income ranges can provide additional insights. For instance, if the applicant's income range is low but the co-applicant's income range is high, it may increase the overall chances of loan approval. Not only that, different income ranges yielded different increases in loan success rate results. Therefore, it is recommended that different combinations''')

    return prediction



    
def main():
  page = 'Question1'
    # Train the model before using it for prediction
  X_train, X_test, y_train, y_test = prepare_data()
  trained_model = train_xgb_model(X_train, y_train)  # Train the model
  
  # Initialize session_state dictionary
  if 'page' not in st.session_state:
    st.session_state['page'] = 'Question1'
  page = st.sidebar.radio('Navigate', ['Question1'])

  # ... existing code ...
  if page == 'Question1':
    page_question1(df, trained_model)

main()