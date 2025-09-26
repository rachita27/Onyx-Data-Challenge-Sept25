import pandas as pd
import numpy as np
import streamlit as st
import os
import joblib

# st.markdown(
#     """
#     <h1 style='text-align: left; color: #2E86C1;'>
#         Nova Bank: Credit Risk Evaluation App
#     </h1>
#     """,
#     unsafe_allow_html=True
# ) 

# st.title("Nova Bank: Credit Risk Evaluation App")
import streamlit as st

col1, col2 = st.columns([1,4])
pth = "C:\\Users\\rachi\\OneDrive\\Desktop\\Python Learning\\onyx_sept\\Data_Code\\Models_output\\"

BASE_DIR = os.path.dirname(__file__)  
# MODEL_PATH = os.path.join(BASE_DIR, "models", "nova_logo.png")

col1, col2 = st.columns([1, 3])  # adjust ratio



with col1:
    st.image(os.path.join(BASE_DIR, "models", "nova_logo.png"), width=90)

with col2:
    st.markdown("<h4>Nova Bank: Credit Risk Evaluation App </h4>", unsafe_allow_html=True)

    # Custom CSS to prevent wrapping
    # st.markdown(
    #     """
    #     <style>
    #     .no-wrap-title {
    #         white-space: nowrap;
    #         font-size: 36px;
    #         font-weight: bold;
    #         color: #2E86C1;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    # # Apply custom class
    # st.markdown("<h3 class='no-wrap-title'; color: #2E86C1;>Nova Bank: Credit Risk Evaluation App </h3>", unsafe_allow_html=True)

# cwd = os.getcwd()
# st.write("Current working directory:", cwd)
# C:\Users\rachi\OneDrive\Desktop\Python Learning\onyx_sept\Data_Code\Clean_Credit.csv

@st.cache_data
def clean_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "models", "Clean_Credit.csv"))
    return df

df = clean_data()

col_nm = ['person_home_ownership',
 'loan_intent',
 'loan_grade',
 'cb_person_default_on_file',
 'gender',
 'marital_status',
 'education_level',
 'country',
 'state',
 'city',
 'employment_type',
 'loan_term_months',
 'past_delinquencies']

tab1, tab2, tab3 = st.tabs(["Data Required Info", "Data Upload","Risk Adjustment Tool"])

with tab1:
    st.text("Data Sample View:")
    st.write(df.sample(n =2))
    st.markdown("<h5> Please Make Sure Data is uploaded above shown format. </h5>", unsafe_allow_html=True)
    st.markdown("<h6> Select the Column & Check the Acceptable Value </h6>", unsafe_allow_html=True)
    st.markdown("<h7> (**Only for Categorical Columns) </h6>", unsafe_allow_html=True)
    optn1 = st.selectbox("Select the column name: ",col_nm)
    st.write(f"Column Name:{optn1} Unique Values: {df[optn1].unique().tolist()} ")

with tab2:
    

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    int_lngth = joblib.load(os.path.join(BASE_DIR, "models", "interest_rt.joblib"))
    st.markdown("<h6> Here you can Upload Your Customer Data With all necessary Columns </h6>", unsafe_allow_html=True)
    st.markdown("<h8> For Demo Purpose you can export the table from Tab1 </h8>", unsafe_allow_html=True)
    if uploaded_file is not None:
        # st.success("✅ File uploaded successfully!")
        df_f = pd.read_csv(uploaded_file) 
        X_train= df_f.loc[:,~df_f.columns.isin(['client_ID','loan_status','city_latitude','city_longitude'])]


        ##Null Treatment
        null_df = joblib.load(os.path.join(BASE_DIR, "models", "null_teat.joblib"))
        dct = {i: j for i, j in list(zip(null_df['Column_name'],null_df['treat'])) if i != 'loan_status'}
        X_train.fillna(dct,inplace = True)

        ##Scaling & Adjusting the variables
        person_home_ownership= pd.get_dummies(X_train['person_home_ownership'] , dtype=int) #,columns=['MORTGAGE','OTHER' ,'OWN',	'RENT'] 

        loan_intent = pd.get_dummies(X_train['loan_intent'], dtype=int) #.reindex(columns= ['DEBTCONSOLIDATION','EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'], fill_value=0)
        
        ##loan grade
        ln_gr = joblib.load(os.path.join(BASE_DIR, "models", "loan_grade.pkl"))
        loan_grade =pd.DataFrame(ln_gr.transform(X_train[['loan_grade']]).ravel(), columns=['loan_grade'])
        # st.write(person_home_ownership.head())

        cb_person_default_on_file = pd.get_dummies(X_train['cb_person_default_on_file'], dtype=int)

        ##gender
        gender = pd.get_dummies(X_train['gender'], dtype=int)
        
        ##marital_status
        marital_status = pd.get_dummies(X_train['marital_status'], dtype=int)

        ##educational_level
        edn_level = joblib.load(os.path.join(BASE_DIR, "models","edn_level.pkl"))
        # edn_level = OrdinalEncoder(categories= [['High School', 'Bachelor', 'Master', 'PhD']])
        education_level = pd.DataFrame(edn_level.transform(X_train[['education_level']]).ravel(), columns=['education_level'])

        ###Location variable, although as per EDA, it seems there aren't important	
        country = pd.get_dummies(X_train['country'], dtype=int)
        state = pd.get_dummies(X_train['state'], dtype=int)
        city = pd.get_dummies(X_train['city'],  dtype=int)

        ##loan_term_months
        ln_term_mnth = joblib.load(os.path.join(BASE_DIR, "models","ln_term_mnth.pkl"))
        loan_term = pd.DataFrame(ln_term_mnth.transform(X_train[['loan_term_months']]).ravel(), columns=['loan_term_months'])
        # st.write(loan_term.head())
        ##past_delinquencies
        past_delinquencies = pd.get_dummies(X_train['past_delinquencies'], dtype=int)
        past_delinquencies.columns = ["past_delinquencies_"+str(i) for i in past_delinquencies.columns]

        X_cat = pd.concat([person_home_ownership,loan_intent, loan_grade,cb_person_default_on_file,gender,marital_status,education_level,country,state,city,loan_term,past_delinquencies],axis = 1)
        # st.write(X_cat.head())
        X_cat = X_cat.reindex(columns= ['OTHER', 'OWN', 'RENT', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL',
       'PERSONAL', 'VENTURE', 'loan_grade', 'Y', 'Male', 'Married', 'Single',
       'Widowed', 'education_level', 'UK', 'USA', 'California', 'England',
       'New York', 'Ontario', 'Quebec', 'Scotland', 'Texas', 'Wales',
       'Cardiff', 'Dallas', 'Edinburgh', 'Glasgow', 'Houston', 'London',
       'Los Angeles', 'Manchester', 'Montreal', 'New York City', 'Ottawa',
       'Quebec City', 'San Francisco', 'Swansea', 'Toronto', 'Vancouver',
       'Victoria', 'loan_term_months', 'past_delinquencies_1',
       'past_delinquencies_2', 'past_delinquencies_3', 'past_delinquencies_4',
       'past_delinquencies_5', 'past_delinquencies_6'],fill_value=0)
        # st.write(X_cat.head())

        ###Numeric column
        person_age_T = joblib.load(os.path.join(BASE_DIR, "models","person_age_T.pkl"))
        age_t = person_age_T.transform(X_train[['person_age']]) #.ravel()

        person_income_T = joblib.load(os.path.join(BASE_DIR, "models","person_income_T.pkl"))
        inc_t = person_income_T.transform(X_train[['person_income']]) #.ravel()

        ##as it has zeros so box-cox can't use
        person_emp_length_T = joblib.load(os.path.join(BASE_DIR, "models","person_emp_length_T.pkl"))
        emp_length_t = person_emp_length_T.transform(X_train[['person_emp_length']]) #.ravel()

        loan_amnt_T = joblib.load(os.path.join(BASE_DIR, "models","loan_amnt_T.pkl"))
        loan_amt_t = loan_amnt_T.transform(X_train[['loan_amnt']]) #.ravel()

        loan_int_rate_T = joblib.load(os.path.join(BASE_DIR, "models","loan_int_rate_T.pkl"))
        loan_int_t = loan_int_rate_T.transform(X_train[['loan_int_rate']]) #.ravel()

        ##as it has zeros so box-cox can't use
        loan_percent_income_T = joblib.load(os.path.join(BASE_DIR, "models","loan_percent_income_T.pkl"))
        loan_income_t =loan_percent_income_T.transform(X_train[['loan_percent_income']]) #.ravel()

        cred_history_T = joblib.load(os.path.join(BASE_DIR, "models","cred_history_T.pkl"))
        cred_history_t = cred_history_T.transform(X_train[['cb_person_cred_hist_length']]) #.ravel()

        ##Ratio
        loan_to_income_ratio_T = joblib.load(os.path.join(BASE_DIR, "models","loan_to_income_ratio_T.pkl"))
        loan_to_income_ratio_t = loan_to_income_ratio_T.transform(X_train[['loan_to_income_ratio']]) #.ravel()

        other_debt_T = joblib.load(os.path.join(BASE_DIR, "models","other_debt_T.pkl"))
        other_debt_t = other_debt_T.transform(X_train[['other_debt']]) #.ravel()

        ##Ratio
        debt_to_income_ratio_T = joblib.load(os.path.join(BASE_DIR, "models","debt_to_income_ratio_T.pkl"))
        debt_to_income_ratio_t = debt_to_income_ratio_T.transform(X_train[['debt_to_income_ratio']]) #.ravel()

        open_accounts_T = joblib.load(os.path.join(BASE_DIR, "models","open_accounts_T.pkl"))
        open_accounts_t = open_accounts_T.transform(X_train[['open_accounts']]) #.ravel()

        ##Ratio
        credit_utilization_ratio_T = joblib.load(os.path.join(BASE_DIR, "models","credit_utilization_ratio_T.pkl"))
        credit_utilization_ratio_t = credit_utilization_ratio_T.transform(X_train[['credit_utilization_ratio']]) #.ravel()

        merged = np.hstack([age_t,inc_t,emp_length_t,loan_amt_t,loan_int_t, loan_income_t, cred_history_t,loan_to_income_ratio_t,other_debt_t,debt_to_income_ratio_t,open_accounts_t,credit_utilization_ratio_t])
        # merged.head()
        X_train_num = pd.DataFrame(merged, columns= ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length','loan_to_income_ratio','other_debt','debt_to_income_ratio','open_accounts','credit_utilization_ratio'])
        # X_train_num.head()

        ###############################
        ############################
        X_train_treat = pd.concat([X_train_num,X_cat], axis = 1)
        # st.write(X_train_treat.head())

        ##Model Prediction
        grid_search = joblib.load(os.path.join(BASE_DIR, "models","xg_model.pkl"))
        df_f['loan_status_predicted'] = grid_search.predict(X_train_treat).ravel().tolist()
        df_f['predicted_probability'] = grid_search.predict_proba(X_train_treat)[:,1].ravel().tolist()

        # st.write(df_f.head())
        # def style_special_cols(x):
        #     styles = pd.DataFrame("", index=x.index, columns=x.columns)
    
        #     # Highlight Pass and Grade with bold + colored font
        #     styles["loan_status_predicted"] = "background-color: #fef3bd; font-weight: bold; color: darkblue;"
        #     styles["predicted_probability"] = "background-color: #ffd6a5; font-weight: bold; color: darkred;"
            
        #     return styles

        # # Apply style
        # styled_df = df_f.style.apply(style_special_cols, axis=None)
        # st.dataframe(styled_df)

        special_cols = ["loan_status_predicted", "predicted_probability"]

        # Styling function
        def highlight_special_cols(row):
            styles = [""] * len(row)  # default = no style
            if row["loan_status_predicted"] == 1:
                for col in special_cols:
                    styles[row.index.get_loc(col)] = "background-color: lightcoral; color: white; font-weight: bold;"
            else:
                for col in special_cols:
                    styles[row.index.get_loc(col)] = "background-color: lightgreen; color: black;"
            return styles

        # Apply style
        styled_df = df_f.style.apply(highlight_special_cols, axis=1)
        st.session_state["base_df"] = df_f
        st.session_state["train_treat"] = X_train_treat

        # Display
        st.dataframe(styled_df)
        st.markdown("<h3> Please note that: </h3>", unsafe_allow_html=True)
        st.write("Scroll from left to right, in the end predicted probability & status is added.")
        st.write("Hover on top right corner of table you can export the table")


with tab3:
    
    if "base_df" in st.session_state:
        df_f = st.session_state["base_df"]   
        X_train_treat = st.session_state["train_treat"] 
        med = joblib.load(os.path.join(BASE_DIR, "models","null_teat.joblib"))

        if df_f[df_f['loan_status_predicted'] == 1].any().any():
            
            ## Slider Option

            st.markdown("<h7> Please select the customer to check whether we can reduce the risk of Default </h7> <br> </br>", unsafe_allow_html=True)
            optn2 = st.selectbox("Please Select the Customer who're predicted as Defaulter: ",df_f.loc[df_f['loan_status_predicted'] ==1, ['client_ID']]['client_ID'].tolist())
            st.markdown(f"<h7> Customer Selected: {optn2} <br> </br>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Current Actual Interest Rate: ", float(df_f.loc[df_f['client_ID'] == optn2, ['loan_int_rate']].iloc[0].iat[0]),"%")
                int_rate = st.slider("Adjust Interest Rate (%)", min_value=0.0, max_value=80.0, value= float(df_f.loc[df_f['client_ID'] == optn2, ['loan_int_rate']].iloc[0].iat[0]), step=0.0001)
            with col2:
                st.write("Current Actual Loan-to-Income Ratio: ", float(df_f.loc[df_f['client_ID'] == optn2, ['loan_to_income_ratio']].iloc[0].iat[0]))
                loan_to_income = st.slider("Adjust Loan-to-Income Ratio", min_value=0.0, max_value=1.0, value=float(df_f.loc[df_f['client_ID'] == optn2, ['loan_to_income_ratio']].iloc[0].iat[0]), step=0.0001)
            with col3:
                st.write("Current Actual Credit Utilization Rate: ", float(df_f.loc[df_f['client_ID'] == optn2, ['credit_utilization_ratio']].iloc[0].iat[0]))
                credit_utilization_ratio = st.slider("Adjust Credit Utilization Rate", min_value=0.0, max_value=1.0, value=float(df_f.loc[df_f['client_ID'] == optn2, ['credit_utilization_ratio']].iloc[0].iat[0]), step=0.0001)
    
            
            ##Making the amendments to check low Risk Profile
            
            loan_to_income_ratio_T = joblib.load(os.path.join(BASE_DIR, "models","loan_to_income_ratio_T.pkl"))
            loan_to_incom_rt = loan_to_income_ratio_T.transform(pd.DataFrame({'loan_to_income_ratio':[loan_to_income]})) #.ravel()
            
            treat_X = pd.concat([X_train_treat,df_f['client_ID']],axis =1)
            

            # 
            treat_X = treat_X.loc[df_f['client_ID'] == optn2,:]
            # st.write(treat_X)
            # st.write(loan_to_incom_rt.ravel())
            treat_X['loan_to_income_ratio'] = loan_to_incom_rt.ravel()

            loan_int_rate_T = joblib.load(os.path.join(BASE_DIR, "models","loan_int_rate_T.pkl"))
            loan_int_ = loan_int_rate_T.transform(pd.DataFrame({'loan_int_rate':[int_rate]}) )

            # st.write(loan_int_.ravel())
            treat_X['loan_int_rate'] = loan_int_.ravel()
            # st.write(treat_X)

            credit_utilization_ratio_T = joblib.load(os.path.join(BASE_DIR, "models","credit_utilization_ratio_T.pkl"))
            credit_utilization_ratio_ = credit_utilization_ratio_T.transform(pd.DataFrame({'credit_utilization_ratio':[credit_utilization_ratio]}))

            # st.write(credit_utilization_ratio_.ravel())
            treat_X['credit_utilization_ratio'] = credit_utilization_ratio_.ravel()
            # st.write(treat_X)
            
            grid_search = joblib.load(os.path.join(BASE_DIR, "models","xg_model.pkl"))
            # st.write(grid_search.predict(treat_X.drop(columns='client_ID',axis=1)).ravel())
            if (grid_search.predict(treat_X.drop(columns='client_ID',axis=1))) == 1:
                st.markdown("<h5> As per the New Variable Selections Customer is Still DEFAULTER </h5>", unsafe_allow_html=True)
                st.write("The PROBABILITY OF DEFAULT IS: ",round(float(grid_search.predict_proba(treat_X.drop(columns='client_ID',axis=1))[:,1].ravel()),4))
            
            else:

                st.markdown("<h5> As per the New Variable Selections Customer is NOT DEFAULTER anymore.</h5>", unsafe_allow_html=True)
                st.write("The PROBABILITY OF DEFAULT IS Reduced to: ",round(float(grid_search.predict_proba(treat_X.drop(columns='client_ID',axis=1))[:,1].ravel()),4))
            

            
        else:
            st.markdown("<h3> No Customer Predicted as Defaulter: </h3>", unsafe_allow_html=True)



        # st.dataframe(df_f.head())
    else:
        st.warning("⚠️ Please upload data in Tab 2 first.")



