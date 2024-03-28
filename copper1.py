import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st 
import re 

st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
     <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

# Define the possible values for the dropdown menus 
status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'off']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 2]
product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
           '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
           '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
           '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
           '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331']

# Define the widgets for user input 
with st.form("my_form"):
    col1, col2, col3 = st.columns([5, 2, 5])
    with col1:
        st.write(' ')
        status = st.selectbox("Status", status_options, key=1) 
        item_type = st.selectbox("Item Type", item_type_options, key=2)
        country = st.selectbox("Country", sorted(country_options), key=3)
        application = st.selectbox("Application", sorted(application_options), key=4)
        product_ref = st.selectbox("Product Reference", product, key=5)
    with col3:
        st.write('<h5 style ="color:rgb(0,153,153,0.4);">NOTE: Min & Max given for reference,</h5>', unsafe_allow_html=True)
        quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
        thickness = st.text_input("Enter Thickness (Min:0.18 & Max:1722207579)")
        width = st.text_input("Enter Width (Min:1, Max:2990)")
        customer = st.text_input("Customer ID (Min:12458, Max:30408185)")
        submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
        st.markdown("""
            <style>
            div.stButton > button:first-child{
                background-color: #009999;
                color:white;
                width:100%;
            }
            </style>
        """, unsafe_allow_html=True)

    cflag = 0 
    pattern = "(?:\d+|\d*\.\d+)$"
    for i in [quantity_tons, thickness, width, customer]:
        if re.match(pattern, i):
            pass
        else:
            cflag = 1
            break      

    if submit_button and cflag == 1:
        if len(i) == 0:
            st.write("Please enter a valid number, spaces not allowed")
        else:
            st.write("You have entered an invalid value:", i)

    if submit_button and cflag == 0:
        import pickle 
        with open("C:/Users/PRIYAN/Desktop/project/copper modelling/cop_cmodel.pkl", 'rb') as f:
            cmodel_loaded = pickle.load(f)

        with open("C:/Users/PRIYAN/Desktop/project/copper modelling/cop_cscalar.pkl", 'rb') as f:
            cscalar_loaded = pickle.load(f)

        with open("C:/Users/PRIYAN/Desktop/project/copper modelling/cop_ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)

        # Predict the selling price for a new sample 
        # 'quantity tons_log', 'setting_price_log', 'application', 'thickness_log', 'width', 'country'
        new_sample = np.array([[np.log(float(quantity_tons)), np.log(float(thickness)), application, np.log(float(width)), country]])
        new_sample = ct_loaded.transform(new_sample)
        new_sample = cscalar_loaded.transform(new_sample)
        new_pred = cmodel_loaded.predict(new_sample)
        st.write(f"The predicted selling price is: {new_pred}")
