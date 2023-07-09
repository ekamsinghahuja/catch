import streamlit as st

# Collect user input using Streamlit components
import pandas as pd
import numpy as np
import pickle
from PIL import Image

st.title("Credit Ease")

image_path = "/Users/ekamsinghahuja/Desktop/catch/imgee.jpeg"



# Open the image using Pillow
image = Image.open(image_path)

# Resize the image to the desired width and height
resized_image = image 

# Display the resized image
st.image(resized_image, caption='credit ease', width=500)




ohe = pickle.load(open('ohe.pkl','rb'))
scaler_normal = pickle.load(open('scl.pkl','rb'))
model = pickle.load(open('model_creditease.pkl','rb'))
da = pickle.load(open('da.pkl','rb'))

merge_ohe_col = np.concatenate((ohe.categories_[0], 
                ohe.categories_[1],
                ohe.categories_[2],
                ohe.categories_[3],
                ohe.categories_[4],
                ))

def clean_data(data_point):
    data_point_as_frame = data_point
    
    #grouping data
    data_point_as_frame['age_group'] = pd.cut(data_point_as_frame['person_age'],bins=[20, 26, 36, 46, 56, 66],labels=['20-25', '26-35', '36-45', '46-55', '56-65'])
    data_point_as_frame['income_group'] = pd.cut(data_point_as_frame['person_income'],bins=[0, 25000, 50000, 75000, 100000, float('inf')],labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
    data_point_as_frame['loan_amount_group'] = pd.cut(data_point_as_frame['loan_amnt'],bins=[0, 5000, 10000, 15000, float('inf')],labels=['small', 'medium', 'large', 'very large'])
    
    #ratios
    # Create loan-to-income ratio
    data_point_as_frame['loan_to_income_ratio'] = data_point_as_frame['loan_amnt'] / data_point_as_frame['person_income']

    # Create loan-to-employment length ratio
    data_point_as_frame['loan_to_emp_length_ratio'] =  data_point_as_frame['person_emp_length']/ data_point_as_frame['loan_amnt'] 

    # Create interest rate-to-loan amount ratio
    data_point_as_frame['int_rate_to_loan_amt_ratio'] = data_point_as_frame['loan_int_rate'] / data_point_as_frame['loan_amnt']
    
    drop_colums = ['cb_person_cred_hist_length','cb_person_default_on_file','loan_grade']
    scale_cols = ['person_income','person_age','person_emp_length', 'loan_amnt','loan_int_rate','loan_percent_income','loan_to_income_ratio', 'loan_to_emp_length_ratio',
        'int_rate_to_loan_amt_ratio']
    ohe_colums = ['person_home_ownership','loan_intent','income_group','age_group','loan_amount_group',]
    
    
    col_list = ['person_age',#
    'person_income',#
    'person_home_ownership',#
    'person_emp_length',#
    'loan_intent', #
    'loan_grade',#
    'loan_amnt',#
    'loan_int_rate',#
    'loan_status',#
    'loan_percent_income',#
    'cb_person_default_on_file',#
    'cb_person_cred_hist_length',
    'age_group','income_group','loan_amount_group']
    
    # merge_ohe_col = np.concatenate((ohe.categories_[0], 
    #             ohe.categories_[1],
    #             ohe.categories_[2],
    #             ohe.categories_[3],
    #             ohe.categories_[4],
    #             ohe.categories_[5],
    #             ohe.categories_[6]))
    
    #drop
    # data_point_as_frame = data_point_as_frame.drop(drop_colums, axis=1)
    
    
    #one hot
    ohe_data = pd.DataFrame(ohe.transform(data_point_as_frame[ohe_colums]).toarray(), columns=merge_ohe_col)
    
    
    data_point_as_frame_new = pd.concat([ohe_data, data_point_as_frame], axis=1)
    data_point_as_frame_new = data_point_as_frame_new.drop(ohe_colums, axis=1)
    data_point_as_frame_new[scale_cols] = scaler_normal.transform(data_point_as_frame_new[scale_cols])
    
    data_point_as_frame_new.columns = data_point_as_frame_new.columns.astype(str)
    data_point_as_frame_new = data_point_as_frame_new.drop(['nan'], axis=1)
    return data_point_as_frame_new


import pandas as pd
from scipy.spatial.distance import cdist

def fun(d):
    
    df = pd.DataFrame(d, index=[0])
    df = clean_data(df)
    da2 = da[da['loan_status']==0]
    da2 = da2.drop(['loan_status'],axis=1)
    da2.reset_index(inplace=True)
    da2=da2.drop(['index'],axis=1)
    da3 = clean_data(da2)
    distances = cdist(da3 , df)
    nearest_index = distances.argmin()
    return da2.iloc[nearest_index]



def in_put(d):
    # types=['int64',
    # 'int64',
    # 'object',
    # 'float64',
    # 'object',
    # 'object',
    # 'int64',
    # 'float64',
    # 'int64',
    # 'float64',
    # 'object',
    # 'int64']
    
    df = pd.DataFrame(d, index=[0])
    # df = df.astype(dict(zip(df.columns, types)))
    
    
    df_new = clean_data(df)
    return model.predict(df_new),model.predict_proba(df_new),fun(d)

person_age = st.number_input("Person Age", min_value=0)
person_income = st.number_input("Person Income Monthly", min_value=1)
person_home_ownership = st.selectbox("Person Home Ownership", ['OWN', 'RENT', 'MORTGAGE','OTHER'])
person_emp_length = st.number_input("Person Employment Length", min_value=0)
loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION',
       'HOMEIMPROVEMENT'])
# loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.selectbox("Loan Interest Rate", [5,6,7,8,10,11,12,13,14,15,16,17,20])
# loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0)
# cb_person_default_on_file = st.selectbox("Person Default on File", ['Y', 'N'])
# cb_person_cred_hist_length = st.number_input("Person Credit History Length", min_value=0)

# Create a dictionary from the user input
user_input = {
    'person_age': person_age,
    'person_income': person_income,
    'person_home_ownership': person_home_ownership,
    'person_emp_length': person_emp_length,
    'loan_intent': loan_intent,
    # 'loan_grade': 'A',
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_amnt / person_income,
    # 'cb_person_default_on_file': 0,
    # 'cb_person_cred_hist_length': 0
}


# Display the user input dictionary
# st.write("User Input:", user_input)
submit_button = st.button("Submit")
if submit_button:
    try:
        val = in_put(user_input)
        # st.write("solved",val)
        prediction = val[0][0]
        zero = val[1][0][0]
        one = val[1][0][1]
        nei = val[2]
        
        if (prediction==0):
            st.header("Congo! Loan approved")
            st.metric(label="Loan-approval-probability", value= zero)
        else:
            
            st.header("Sorry! Loan not approved")
            st.metric(label="Loan-approval-probability", value= zero)
            st.header("Check nearest person who got approved")
            
            st.subheader(f'Person Age - {nei[0]}')
            st.subheader(f'Person Income Monthly - {nei[1]}')
            st.subheader(f'Person Home Ownership - {nei[2]}')
            st.subheader(f'Person Employment Length - {nei[3]}')
            st.subheader(f'Loan Intent - {nei[4]}')
            st.subheader(f'Loan Amount - {nei[5]}')
            st.subheader(f'Loan Interest Rate - {nei[6]}')
            # st.subheader(f'Loan Interest Rate - {nei[7]}')
        
    except Exception as e:
        # Exception handling code
        st.write("Please Refill")
        
    
    
    
