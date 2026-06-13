import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Hotel Cancellation Predictor', layout='wide')

@st.cache_resource
def load_model():
    return joblib.load('hotel_pipeline.pkl')

@st.cache_data
def load_cleaned():
    return pd.read_csv('cleaned_df.csv')

model = load_model()
df_clean = load_cleaned()

st.sidebar.title('Hotel Booking Pipeline')
page = st.sidebar.radio('Go to', ['EDA Dashboard', 'Prediction', 'Model Performance', 'About'])

if page == 'EDA Dashboard':
    st.title('Exploratory Data Analysis')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Bookings', f'{df_clean.shape[0]:,}')
    col2.metric('Canceled %', f'{df_clean["is_canceled"].mean()*100:.1f}%')
    col3.metric('Features', df_clean.shape[1])

    tab1, tab2 = st.tabs(['Numerical', 'Categorical'])
    with tab1:
        num = df_clean.select_dtypes(include='number').drop('is_canceled', axis=1)
        col = st.selectbox('Feature', num.columns)
        fig = px.histogram(df_clean, x=col, color='is_canceled', barmode='overlay',
                          color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)
        corr = num.corr()
        fig2 = px.imshow(corr, text_auto='.2f', color_continuous_scale='Picnic', aspect='auto')
        st.plotly_chart(fig2, use_container_width=True)
    with tab2:
        cat = df_clean.select_dtypes(include='object')
        col = st.selectbox('Category', cat.columns)
        ct = pd.crosstab(df_clean[col], df_clean['is_canceled'], normalize='index').reset_index().melt(id_vars=[col])
        fig = px.bar(ct, x=col, y='value', color='is_canceled', barmode='group',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, labels={'value': 'Rate'})
        st.plotly_chart(fig, use_container_width=True)

elif page == 'Prediction':
    st.title('Predict Cancellation')
    st.markdown('Fill details and click Predict')

    with st.form('form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            hotel = st.selectbox('Hotel', ['City Hotel', 'Resort Hotel'])
            lead = st.number_input('Lead Time', 0, 1000, 50)
            weekend = st.number_input('Weekend Nights', 0, 30, 1)
            week = st.number_input('Week Nights', 0, 50, 3)
            adults = st.number_input('Adults', 0, 10, 2)
            children = st.number_input('Children', 0, 10, 0)
            babies = st.number_input('Babies', 0, 10, 0)
        with col2:
            meal = st.selectbox('Meal', ['BB', 'FB', 'HB', 'SC', 'Undefined'])
            country = st.text_input('Country', 'PRT')
            segment = st.selectbox('Market Segment', ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Aviation'])
            channel = st.selectbox('Channel', ['Direct', 'Corporate', 'TA/TO', 'GDS'])
            repeated = st.selectbox('Repeated Guest', [0, 1])
            prev_cancel = st.number_input('Prev Cancellations', 0, 50, 0)
            prev_not = st.number_input('Prev Not Canceled', 0, 100, 0)
        with col3:
            room = st.selectbox('Reserved Room', [chr(c) for c in range(ord('A'), ord('P')+1)])
            assigned = st.selectbox('Assigned Room', [chr(c) for c in range(ord('A'), ord('P')+1)])
            changes = st.number_input('Changes', 0, 20, 0)
            deposit = st.selectbox('Deposit', ['No Deposit', 'Refundable', 'Non Refund'])
            waiting = st.number_input('Waiting List', 0, 1000, 0)
            cust = st.selectbox('Customer Type', ['Transient', 'Contract', 'Transient-Party', 'Group'])
            parking = st.number_input('Parking Spaces', 0, 10, 0)
            requests = st.number_input('Special Requests', 0, 20, 0)
            adr = st.number_input('ADR', 0.0, 1000.0, 100.0)

        submitted = st.form_submit_button('Predict', type='primary')

    if submitted:
        total_stay_val = weekend + week
        total_people_val = adults + children + babies
        data = {
            'hotel': hotel,
            'lead_time': lead,
            'stays_in_weekend_nights': weekend,
            'stays_in_week_nights': week,
            'adults': adults,
            'children': children,
            'babies': babies,
            'meal': meal,
            'country': country,
            'market_segment': segment,
            'distribution_channel': channel,
            'is_repeated_guest': repeated,
            'previous_cancellations': prev_cancel,
            'previous_bookings_not_canceled': prev_not,
            'reserved_room_type': room,
            'assigned_room_type': assigned,
            'booking_changes': changes,
            'deposit_type': deposit,
            'days_in_waiting_list': waiting,
            'customer_type': cust,
            'adr': adr,
            'required_car_parking_spaces': parking,
            'total_of_special_requests': requests,
            'total_stay': total_stay_val,
            'total_people': total_people_val,
            'arrival_month_num': 6,
            'arrival_season': 'Summer'
        }
        inp = pd.DataFrame([data])

        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('Prediction', 'Cancel' if pred == 1 else 'Not Cancel')
        with c2:
            st.metric('Cancel Probability', f'{prob[1]:.1%}')
        with c3:
            st.metric('Not Cancel Probability', f'{prob[0]:.1%}')

        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Not Canceled', 'Canceled'], y=prob,
                           marker_color=['#2ecc71', '#e74c3c'],
                           text=[f'{p:.1%}' for p in prob], textposition='auto'))
        fig.update_layout(title='Probabilities', yaxis_range=[0, 1], height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == 'Model Performance':
    st.title('Model Performance')
    st.markdown('6-model 5-fold CV (70/30 train/test split on cleaned_df.csv)')

    col1, col2, col3 = st.columns(3)
    col1.metric('Best Model (CV)', 'XGBoost')
    col2.metric('CV F1 Score', '0.6867')
    col3.metric('CV Accuracy', '0.8323')

    st.markdown('**Hold-out Test Set Results (30% unseen data):**')
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric('Test F1', '0.6887')
    tc2.metric('Test Accuracy', '0.8302')
    tc3.metric('Precision (Canceled)', '0.69')
    tc4.metric('Recall (Canceled)', '0.69')

    data = {
        'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree'],
        'F1 Train': [0.7311, 0.6848, 0.6604, 0.6153, 0.7365, 0.9968],
        'F1 Test': [0.6867, 0.6714, 0.6456, 0.6144, 0.5921, 0.5829],
        'Acc Train': [0.8556, 0.8255, 0.7731, 0.7363, 0.8160, 0.9983],
        'Acc Test': [0.8323, 0.8183, 0.7639, 0.7355, 0.7074, 0.7620]
    }
    df_perf = pd.DataFrame(data)

    tab1, tab2 = st.tabs(['F1 Score', 'Accuracy'])
    with tab1:
        fig = go.Figure()
        for m in ['F1 Train', 'F1 Test']:
            fig.add_trace(go.Bar(name=m, x=df_perf['Model'], y=df_perf[m], text=df_perf[m], textposition='auto'))
        fig.update_layout(barmode='group', yaxis_range=[0, 1], height=450)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig2 = go.Figure()
        for m in ['Acc Train', 'Acc Test']:
            fig2.add_trace(go.Bar(name=m, x=df_perf['Model'], y=df_perf[m], text=df_perf[m], textposition='auto'))
        fig2.update_layout(barmode='group', yaxis_range=[0, 1], height=450)
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander('Full Results Table'):
        st.dataframe(df_perf.style.format({c: '{:.4f}' for c in df_perf.columns if c != 'Model'}))
        st.markdown('**Note:** Models ranked by CV F1 Test score. Final model: XGBoost (lr=0.1, max_depth=6, n_estimators=200)')

else:
    st.title('About')
    st.markdown('Hotel Booking Cancellation Prediction using ML.')

