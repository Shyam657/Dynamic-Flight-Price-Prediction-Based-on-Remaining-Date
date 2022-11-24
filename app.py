import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as MSE
import warnings
import math
import seaborn as sns
import matplotlib.pyplot as plt




import streamlit as st
import datetime
from datetime import date ,timedelta


st.set_page_config(
    page_title="Dynamic Flight Price Prediction Based on Remaining Date ",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://twitter.com/Shyam_311',
        'Report a bug': "https://twitter.com/Shyam_311",
        'About': "This is an app which can be used to attract customers to onboard a flight with discount in remaining last few days of booking !!!"
    }
)

#image_url="https://cdn.pixabay.com/photo/2016/04/30/08/35/aircraft-1362586__340.jpg"



st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 70%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
    .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2013/08/06/19/13/plane-170272__340.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
    
</style>
""",
    unsafe_allow_html=True,
)




days = 30
total_slots = 250
journey = 50
train_val_prop = 0.75
lst_days = list(range(1,days+1))
wastage_pct_min_range = 15
wastage_pct_max_range = 20
emptiness_threshold = 0.02
optimisation_day_bfr_jouney = 10
journey_id = random.randint(1,journey)


today=date.today()
d1=timedelta(days=optimisation_day_bfr_jouney)
d3,d4=today,today+d1

st.write(' Flights starting with a minimum of price 4000 and maximum of price 14000')
col1, col2, col3 = st.columns([1,1,1])
with col1:
  d = st.date_input(
  "Enter the date you want to onboard !!",
  today+timedelta(days=5),min_value=d3,max_value=d4)
  st.write('Your onboarding Date is :', d)
with col2:
  min_price = st.number_input('Insert Your  minimum flight price',value=5500,step=100,min_value=4000)
with col3:
  max_price = st.number_input('Insert Your maximum  flight price',value=9000,step=300,max_value=14100)






#min_price = 4000
#max_price = 14000




lst_df = []
for j in range(journey):
    lst_price = []
    prev_price = min_price
    for i in range(days):
        cur_price = min(prev_price + random.randint(0,500), max_price)
        prev_price = cur_price
        lst_price.append(cur_price)

    slots_filled =  round((1 - random.randint(wastage_pct_min_range,wastage_pct_max_range)/100.0) * total_slots,0)
    lst_slots = [] 
    weights = []
    for time_,p in enumerate(lst_price):
        
        # More bookings comes with time but also reduces as price increases
        weights.append(((time_ + 1)* random.uniform(1,1.2))/math.pow(p,5.0))

    msum = sum(weights)
    weights = [w/msum for w in weights]


    for w in weights:
        lst_slots.append(round(w * slots_filled, 0) )


    lst_df.append(pd.DataFrame( {'journey_id' : [j+1] * days, 'day':lst_days, 'price' : lst_price, 'slots' : lst_slots} ))
    

df = pd.concat(lst_df)

#Let's train on x% of journeys and validate learn't model performance on the remaining (1-x)%.

df_train = pd.concat(lst_df[:int(train_val_prop * len(lst_df)) + 1])
df_val = pd.concat(lst_df[int(train_val_prop * len(lst_df)) + 1:])





from xgboost import XGBRegressor
from numpy import asarray
model = XGBRegressor()
model.fit(df_train[['price','day']], np.array(df_train.slots))







mp = {}
lst = []
for p in range(min_price, max_price+1):
    for d in range(days - optimisation_day_bfr_jouney , days +  1):
        lst.append( [p,d] )
        
pred = model.predict( np.array(lst))

for i in range(len(lst)):
    mp[lst[i][0],lst[i][1]] = int(pred[i])




def m_feasible(price_points, available_slots, emptiness_threshold):
    tmp = 0
    for p in price_points:
#         tmp = tmp + int(model.predict(np.asarray([[p[0],p[1]]]))[0])
        tmp = tmp + mp[p[0],p[1]]

    if tmp <= available_slots and total_slots * emptiness_threshold <= (available_slots - tmp):
        return True
    return False





def m_revenue(price_points):
    rev = 0
    slots = []
    for p in price_points:
        
        s_filled = mp[p[0],p[1]]
        slots.append(s_filled)
        rev = rev + (s_filled * p[0])

    return rev,slots





def optimise(df, journey_id):

    df_tmp = df[(df.journey_id == journey_id) & (df.day > (days - optimisation_day_bfr_jouney))].reset_index(drop = True)

    slots_filled = df[ (df.journey_id == journey_id) & (df.day < (days - optimisation_day_bfr_jouney)) ].slots.sum()

    available_slots = (total_slots - slots_filled) 
    
    # random search
    times = 50000
    ans = 0
    solution = []
    for j in range(times):
        price_points = []
        prev_price = min_price
        for i in range(optimisation_day_bfr_jouney):
            cur_price = random.randint(prev_price,int(prev_price * 1.15))
            if cur_price > max_price:
                break
            prev_price = cur_price
            price_points.append((cur_price,(days - optimisation_day_bfr_jouney + 1)))
        if m_feasible(price_points, available_slots, emptiness_threshold):
            if m_revenue(price_points)[0] > ans:
                ans, slots = m_revenue(price_points)
                solution = [p[0] for p in price_points]
                
#     df_tmp = df[df.journey_id == 1].loc[days - optimisation_day_bfr_jouney: ].reset_index(drop = True)
    df_tmp['proposed_price'] = solution
    df_tmp['forecasted_slots'] = slots
    
    orig = np.sum(df_tmp['price'] * df_tmp['slots'])
    proposed = np.sum(df_tmp['proposed_price'] * df_tmp['forecasted_slots'])
    revenue_gain = round(proposed-orig, 2)
    revenue_gain_pct = round((proposed-orig)/orig * 100.0, 2)
    
    slots_extra_gain = round(df_tmp['forecasted_slots'].sum() - df_tmp.slots.sum() , 0)
    slots_extra_gain_pct = round( slots_extra_gain/df_tmp.slots.sum() *100, 2)
    
#     display(df_tmp)
  
    return df_tmp[['day','proposed_price']]




df2=optimise(df,journey_id)


st.write(df2)
