import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from prophet import Prophet

USER_CREDENTIALS = {"admin": "password123", "user": "userpass"}

def check_login(username, password):
    return USER_CREDENTIALS.get(username) == password

def send_password_reset_email(username):
    reset_token = f"reset_token_for_{username}"
    reset_token[username] = reset_token
    return reset_token

def reset_user_password(username, new_password):
    if username in USER_CREDENTIALS:
        USER_CREDENTIALS[username] = new_password
        return True
    return False

def train_test_split(data,train_ratio=0.8):
    split_index=int(len(data)*train_ratio)
    train_data=data.iloc[:split_index]
    test_data=data=data.iloc[split_index:]
    return train_data,test_data

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username=""
if "password" not in st.session_state:
    st.session_state.password=""

def logout():
    st.session_state.logged_in=False
    st.session_state.username=""
    st.session_state.password=""

if not st.session_state.logged_in:
    st.title("Login Page")
    st.subheader("Please log in to access the dashboard")

    with st.form("login_form",clear_on_submit=True):
        username=st.text_input("Username",value=st.session_state.username)
        password=st.text_input("Password",type="password",value=st.session_state.password)
        submit_button=st.form_submit_button("Login")

        if submit_button:
            if check_login(username,password):
                st.session_state.logged_in = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
    if st.button("Forgot Password?"):
        reset_username = st.text_input("Enter your username to reset password:")
        reset_button = st.button("Send Reset Link")
        
        if reset_button:
            if reset_username in USER_CREDENTIALS:
                reset_token = send_password_reset_email(reset_username)
                st.success(f"Reset link sent to {reset_username}. Token: {reset_token}")
                reset_password = st.text_input("Enter New Password:",type="password")
                confirm_button = st.button("Confirm Password Reset")
                
                if confirm_button:
                    if reset_user_password(reset_username, reset_password):
                        st.success("Password reset successful! You can now log in with your new password.")
                    else:
                        st.error("Failed to reset password. Please try again.")
            else:
                st.error("Username not found. Please check and try again.")
else:
    st.sidebar.button("Logout",on_click=logout)

    data = pd.read_csv("AEP_hourly.csv")
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)
    st.title("Energy Consumption Dashboard")
    st.sidebar.header("User Options")
    prediction_days = st.sidebar.slider("Days to forecast:", 1, 80, 30)
    tab1,tab2 = st.tabs(["Visualization", "Forecasting"])
    with tab1:
        st.subheader("Raw Data")
        st.write(data.head())

        st.subheader("Energy Consumption Over Time")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data['AEP_MW'], label='AEP_MW', color='blue')
        ax.set_title("Hourly Energy Usage", fontsize=16)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Energy (MW)", fontsize=12)
        ax.grid()
        st.pyplot(fig)

        st.subheader("Daily Energy Usage")
        daily_data = data.resample('D').mean()
        st.line_chart(daily_data['AEP_MW'])

        st.subheader("Hourly Distribution")
        hourly_data = data.groupby(data.index.hour).mean()
        st.bar_chart(hourly_data['AEP_MW'])
    with tab2:
        st.subheader("Forecasting")
        df_prophet=data.resample('D').mean().reset_index()
        df_prophet= df_prophet.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})
        train_data = df_prophet[:-prediction_days]
        test_data = df_prophet[-prediction_days:]
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=prediction_days)
        forecast = model.predict(future)
        test_forecast = forecast[-prediction_days:]
        test_data['yhat'] = test_forecast['yhat'].values
        mae = (test_data['y'] - test_data['yhat']).abs().mean()
        mape = ((test_data['y'] - test_data['yhat']).abs() / test_data['y']).mean() * 100
        rmse = ((test_data['y'] - test_data['yhat']) * 2).mean() * 0.5
        r2 = 1 - sum((test_data['y'] - test_data['yhat']) * 2) / sum((test_data['y'] - test_data['y'].mean()) * 2)
        st.write("*Forecast Evaluation Metrics:*")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        st.write("Forecast Results")
        st.write(test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        fig = model.plot(forecast)
        st.pyplot(fig)







