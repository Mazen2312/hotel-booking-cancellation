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
        data = {
            'hotel': hotel, 'lead_time': lead,
            'stays_in_weekend_nights': weekend, 'stays_in_week_nights': week,
            'adults': adults, 'children': children, 'babies': babies,
            'meal': meal, 'country': country,
            'market_segment': segment, 'distribution_channel': channel,
            'is_repeated_guest': repeated,
            'previous_cancellations': prev_cancel,
            'previous_bookings_not_canceled': prev_not,
            'reserved_room_type': room, 'assigned_room_type': assigned,
            'booking_changes': changes, 'deposit_type': deposit,
            'days_in_waiting_list': waiting, 'customer_type': cust,
            'adr': adr, 'required_car_parking_spaces': parking,
            'total_of_special_requests': requests,
            'company': 0,
            'arrival_date_year': 2017, 'arrival_date_month': 'June',
            'arrival_date_week_number': 25, 'arrival_date_day_of_month': 15
        }
        inp = pd.DataFrame([data])
        inp['total_stay'] = inp['stays_in_weekend_nights'] + inp['stays_in_week_nights']
        inp['total_people'] = inp['adults'] + inp['children'] + inp['babies']
        inp['arrival_month_num'] = 6
        inp['arrival_season'] = 'Summer'
        drop_cols = ['arrival_date_year', 'arrival_date_month', 'arrival_date_week_number',
                     'arrival_date_day_of_month', 'company', 'reservation_status', 'reservation_status_date']
        inp.drop(columns=[c for c in drop_cols if c in inp.columns], inplace=True)

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
    st.markdown('6-model 5-fold CV (no hold-out split, CV is the sole evaluation)')

    col1, col2, col3 = st.columns(3)
    col1.metric('Best Model (Tuned)', 'XGBoost')
    col2.metric('Tuned CV F1', '0.5902')
    col3.metric('CV Accuracy', '0.7126')

    st.markdown('**GridSearchCV Results (Top 4 Models Tuned):**')
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric('XGBoost (Best)', '0.5902')
    tc2.metric('LightGBM', '0.5880')
    tc3.metric('Logistic Regression', '0.5708')
    tc4.metric('KNN', '0.4722')

    data = {
        'Model': ['Logistic Regression', 'LightGBM', 'K-Nearest Neighbors', 'XGBoost', 'Random Forest', 'Decision Tree'],
        'CV F1': [0.5708, 0.4891, 0.4579, 0.4547, 0.4438, 0.4167],
        'CV Accuracy': [0.6896, 0.7312, 0.5949, 0.7126, 0.7165, 0.6685]
    }
    df_perf = pd.DataFrame(data)

    tab1, tab2 = st.tabs(['CV F1 Score', 'CV Accuracy'])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='CV F1', x=df_perf['Model'], y=df_perf['CV F1'], text=df_perf['CV F1'], textposition='auto', marker_color='#3498db'))
        fig.update_layout(yaxis_range=[0, 1], height=450, title='5-Fold Cross-Validation F1 Score')
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='CV Accuracy', x=df_perf['Model'], y=df_perf['CV Accuracy'], text=df_perf['CV Accuracy'], textposition='auto', marker_color='#2ecc71'))
        fig2.update_layout(yaxis_range=[0, 1], height=450, title='5-Fold Cross-Validation Accuracy')
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander('Tuning Details'):
        tuned_data = {
            'Model': ['XGBoost', 'LightGBM', 'Logistic Regression', 'K-Nearest Neighbors'],
            'Best Params': ['lr=0.01, max_depth=3, n_est=200', 'lr=0.01, max_depth=3, n_est=200', 'C=10', 'n_neighbors=11, weights=uniform'],
            'Tuned CV F1': [0.5902, 0.5880, 0.5708, 0.4722]
        }
        st.dataframe(pd.DataFrame(tuned_data).style.format({'Tuned CV F1': '{:.4f}'}))
        st.markdown('**Final Model:** XGBoost (lr=0.01, max_depth=3, n_estimators=200) trained on full dataset')
        st.markdown('**Full Training F1:** 0.6103 | **Full Training Accuracy:** 0.7045')

else:
    st.title('About')
    st.markdown('Hotel Booking Cancellation Prediction using ML.')



