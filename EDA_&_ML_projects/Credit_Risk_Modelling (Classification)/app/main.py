# Import the Streamlit library for creating web apps
import streamlit as st
# Import the custom prediction function from the helper file
from prediction_helper import predict

# Layout: 4-column grid — col1, col2 for content, col3, col4 for social links
col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

# LinkedIn profile link button — opens in new tab
with col3:
    st.link_button("LinkedIn", "https://linkedin.com/in/codebywolf", use_container_width=True)

# GitHub profile link button — opens in new tab
with col4:
    st.link_button("GitHub", "https://github.com/codebywolf", use_container_width=True)

# Set the title of the Streamlit web application
st.title('XCash Finance: Credit Card Modeling')

# Create rows of columns to organize the input fields in a grid layout
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

# --- Input fields for user data ---

# Use 'with' to place the input field in the first column of the first row
with row1[0]:
    # Create a number input for Age
    age = st.number_input("Age",min_value=18,max_value=70,step=1)

with row1[1]:
    # Create a number input for Income
    income = st.number_input("Income",min_value=0,value=50000)

with row1[2]:
    # Create a number input for Loan Amount
    loan_amount = st.number_input("Loan Amount", min_value=0, value=500000, step=1000)


# Calculate the Loan to Income Ratio, handling the case where income is zero
loan_to_income_ratio = loan_amount/income if income > 0 else 0

with row2[0]:
    # Create a number input for Loan Tenure in months
    loan_tenure_months = st.number_input("Loan Tenure (Months)",min_value=0,max_value=59,step=1, value=36)

with row2[1]:
    # Display the calculated Loan to Income Ratio as a disabled field
    st.number_input(
        "Loan to Income Ratio",
        value=round(loan_to_income_ratio, 2),
        disabled=True,
        help='Calculated as (Loan Amount / Income)'
    )

with row2[2]:
    # Create a number input for Average Days Past Due
    avg_dpd_per_delinquent = st.number_input("Avg DPD",min_value=0,step=1, value=5)


with row3[0]:
    # Create a number input for Delinquent Ratio
    delinquent_ratio = st.number_input("Delinquent Ratio",min_value=0,max_value=100,step=1, value=30)

with row3[1]:
    # Create a number input for Credit Utilization Ratio
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio",min_value=0,max_value=99,step=1, value=30)

with row3[2]:
    # Create a number input for the number of open loan accounts
    number_of_open_accounts = st.number_input("Open Loan Accounts",min_value=1,max_value=4,step=1, value=2)


with row4[0]:
    # Create a dropdown select box for Residence Type
    residence_type = st.selectbox("Residence Type",['Owned', 'Rented', 'Mortgage'])

with row4[1]:
    # Create a dropdown select box for Loan Purpose
    loan_purpose = st.selectbox("Loan Purpose",['Education', 'Home', 'Auto', 'Personal'])

with row4[2]:
    # Create a dropdown select box for Loan Type
    loan_type = st.selectbox("Loan Type",['Unsecured', 'Secured'])


# Create a button that, when clicked, will trigger the prediction
if st.button('Calculate Risk'):
    # Call the predict function with all the user-provided inputs
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months,
                                                avg_dpd_per_delinquent, delinquent_ratio, credit_utilization_ratio,
                                                number_of_open_accounts, residence_type, loan_purpose, loan_type )

    # Display the prediction results
    st.write('Risk Probability: ', f'{probability:.2%}')
    st.write('Credit Score: ', credit_score)
    st.write('Rating: ', rating)

# Print the input values to the console for debugging purposes
print(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquent, delinquent_ratio, credit_utilization_ratio, number_of_open_accounts, residence_type, loan_purpose, loan_type)
