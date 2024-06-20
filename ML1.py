import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR
import DS1
import CM2
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def load_data(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    return df


def calculate_error(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100


def parse_month_column(data):
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    data['Month'] = data['Month'].map(month_map)
    data['Month'] = pd.to_datetime(data['Month'], format='%m').dt.strftime('2023-%m-%d')
    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')
    return data


def home_page():
    st.title("Welcome to the Multi-Page Forecasting and Analysis Application")
    st.write("""
        This application provides tools for forecasting and analyzing cashflows for passenger and freight aircraft.

        **Pages:**

        - **Home**: Overview of the application.
        - **Correlation**: Upload an Excel file and visualize the correlation matrix of numeric columns.
        - **ML Algorithms**: Use various machine learning algorithms to predict or forecast values from uploaded data.
        - **Descriptive Statistics**: Generate and display descriptive statistics for your data.

        Use the sidebar to navigate between the pages.
    """)


def correlation_page():
    st.title("Correlation Matrix - Cashflow Prediction")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Read the sheet names
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names

            # Check for the required sheets
            if "Passenger Aircraft" not in sheet_names or "Freight Aircraft" not in sheet_names:
                st.error("The Excel file must contain 'Passenger Aircraft' and 'Freight Aircraft' sheets.")
            else:
                # Select the type of aircraft
                aircraft_type = st.selectbox("Select Aircraft Type", ["Passenger Aircraft", "Freight Aircraft"])

                # Read the selected sheet
                df = pd.read_excel(uploaded_file, sheet_name=aircraft_type)

                st.write(f"DataFrame ({aircraft_type}):")
                st.write(df)

                # Select only numeric columns
                numeric_df = df.select_dtypes(include=['number'])

                if numeric_df.empty:
                    st.error("The uploaded file does not contain any numeric data.")
                else:
                    # Calculate the correlation matrix
                    corr_matrix = numeric_df.corr()

                    st.write("Correlation Matrix:")
                    st.write(corr_matrix)

                    # Handle large number of columns by allowing the user to select a subset of columns
                    columns_to_display = st.multiselect(
                        'Select columns to display in correlation matrix',
                        numeric_df.columns,
                        default=list(numeric_df.columns[:10])  # Convert to list for default selection
                    )

                    if len(columns_to_display) > 0:
                        subset_corr_matrix = corr_matrix.loc[columns_to_display, columns_to_display]

                        # Plot the correlation matrix
                        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed
                        sns.heatmap(subset_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("Please select at least one column to display the correlation matrix.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


def ml_algorithms_page():
    st.title("ML Algorithms")

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        sheet_selection = st.radio(
            "Select the sheet to load data from:",
            ("Passenger Aircraft", "Freight Aircraft")
        )

        data = load_data(uploaded_file, sheet_selection)
        st.write("Data Preview:", data.head())

        if 'Month' in data.columns:
            try:
                data = parse_month_column(data)
                if data['Month'].isna().any():
                    st.error(
                        "Some dates in 'Month' column could not be parsed. Please ensure all dates are in the correct format.")
                else:
                    data.set_index('Month', inplace=True)

                    value_columns = st.multiselect("Select the value columns to forecast",
                                                   [col for col in data.columns if col != 'Month'])

                    additional_columns = [
                        'Landings(ETD Forecast)', 'Seats per Dep(ETD Forecast)', 'Seats(ETD Forecast)',
                        'Avg Stage Length (km)(ETD Forecast)', 'ASKs (m)(ETD Forecast)', 'Block Hours(ETD Forecast)',
                        'Passenger Numbers(ETD Forecast)', 'RPKs (m)(ETD Forecast)',
                        'Leg Tonnage (Mini Freighter & Belly)(ETD Forecast)',
                        'ATKs (m)(ETD Forecast)', 'FTKs (m)(ETD Forecast)', 'Passenger LF%(ETD Forecast)',
                        'Cargo Belly LF%(ETD Forecast)',
                        'P2P %(ETD Forecast)', 'Passenger Numbers( Non Rev pax)(ETD Forecast)',
                        'Total Passenger Numbers( Rev+ Non Rev pax)(ETD Forecast)'
                    ]

                    additional_columns_selected = st.multiselect("Select additional columns to add to the table",
                                                                 [col for col in data.columns if
                                                                  col != 'Month'])

                    # New functionality to calculate and display percentages
                    st.subheader("Percentage of Income Sources Compared to Total(In)")

                    income_columns = [
                        "Credit Card Companies(In)", "BSP clearing bodies(In)", "CASS clearing bodies(In)",
                        "Corporates(In)", "Agency(In)", "Others(In)", "Government/Public sector(In)",
                        "Airline(In)", "Payment Return(In)", "Cash Sales(In)", "VAT Refund(In)",
                        "Industry clearing bodies(In)", "Interest Income(In)"
                    ]

                    if "Total(In)" in data.columns:
                        percentage_data = {}

                        for col in income_columns:
                            if col in data.columns:
                                percentage_data[col] = (data[col] / data["Total(In)"]) * 100

                        percentage_df = pd.DataFrame(percentage_data)
                        st.write(percentage_df)
                    else:
                        st.error("The column 'Total(In)' is not found in the data.")
                        # New functionality to calculate and display percentages of Outgoing
                        st.subheader("Outgoing Cashflow - Percentage (Each Parameter)")

                    outgoing_columns = [
                        "Oil & Gas/ Petroleum(Out)", "DOC Vendor(Out)", "Aircraft Related Supplier(Out)",
                        "Technical(Out)", "Advertising/PR/Marketing(Out)", "Intercompany(Out)",
                        "Government/Public sector(Out)", "IATA(Out)", "Information Technology(Out)",
                        "Tech support(Out)", "Related Parties(Out)", "Real Estate / Property(Out)",
                        "Service charges: CC(Out)", "Insurance(Out)", "Others(Out)", "Utilities(Out)",
                        "Incentives and Commission(Out)", "Inflight(Out)",
                        "Consultancy Services (Legal & Professional)(Out)",
                        "Shipping/ Cargo(Out)", "Holidays Suppliers(Out)", "Food & Beverages(Out)",
                        "Fuel Hedges(Out)", "Etihad Guest Reward Shop(Out)",
                        "Stationary/Office Supplies/Publishing(Out)",
                        "Security and Safety Companies(Out)", "Transport Industry(Out)", "Staff Expense(Out)",
                        "Consolidated Payment(Out)", "Medical(Out)", "Cargo/Guest/Baggage Claims(Out)",
                        "Uniform(Out)", "Customer Security Deposit(Out)", "PASSENGER TICKET REFUND CUSTOMER(Out)",
                        "Bank Charge(Out)", "Industry clearing bodies(Out)", "ICH NON DOC/ICH DOC(Out)",
                        "KUMPULAN WANG SIMPANAN PEKERJA(Out)", "Payment Return(Out)"
                    ]

                    if "Total(Out)" in data.columns:
                        percentage_data = {}

                        for col in outgoing_columns:
                            if col in data.columns:
                                percentage_data[col] = (data[col] / data["Total(Out)"]) * 100

                        percentage_df = pd.DataFrame(percentage_data)
                        st.write(percentage_df)
                    else:
                        st.error("The column 'Total(Out)' is not found in the data.")
                    if not value_columns:
                        st.error("Please select at least one value column to forecast.")
                    else:
                        forecast_type = st.sidebar.selectbox("Select Prediction or Forecast",
                                                             ["Prediction", "Forecast"])
                        algorithm = st.sidebar.radio("Select an algorithm", [
                            "Linear Regression", "Polynomial Regression", "ARIMA", "SARIMA",
                            "Support Vector Regression (SVR)", "Random Forest Regressor", "XGBoost Regressor",
                            "Gradient Boosting Regressor", "K-Nearest Neighbors Regression (KNN)",
                            "Exponential Smoothing (ETS)", "Vector Autoregression (VAR)"
                        ])

                        for value_column in value_columns:
                            st.write(f"### {forecast_type} for {value_column}")

                            X = np.arange(len(data)).reshape(-1, 1)
                            y = data[value_column].values
                            z = data[additional_columns_selected].values
                            if algorithm == "Linear Regression":
                                model = LinearRegression()
                                model.fit(X, y)
                                y_pred = model.predict(X)

                            elif algorithm == "Polynomial Regression":
                                degree = st.slider("Select degree of polynomial", 1, 10, 2,
                                                   key=f"degree_{value_column}")
                                poly = PolynomialFeatures(degree=degree)
                                X_poly = poly.fit_transform(X)
                                model = LinearRegression()
                                model.fit(X_poly, y)
                                y_pred = model.predict(X_poly)

                            elif algorithm == "ARIMA":
                                p = st.number_input("p", min_value=0, value=1, key=f"p_{value_column}")
                                d = st.number_input("d", min_value=0, value=1, key=f"d_{value_column}")
                                q = st.number_input("q", min_value=0, value=1, key=f"q_{value_column}")
                                model = ARIMA(y, order=(p, d, q))
                                model_fit = model.fit()
                                y_pred = model_fit.predict(start=0, end=len(y) - 1)

                            elif algorithm == "SARIMA":
                                p = st.number_input("p", min_value=0, value=1, key=f"p_{value_column}")
                                d = st.number_input("d", min_value=0, value=1, key=f"d_{value_column}")
                                q = st.number_input("q", min_value=0, value=1, key=f"q_{value_column}")
                                P = st.number_input("P", min_value=0, value=1, key=f"P_{value_column}")
                                D = st.number_input("D", min_value=0, value=1, key=f"D_{value_column}")
                                Q = st.number_input("Q", min_value=0, value=1, key=f"Q_{value_column}")
                                s = st.number_input("Seasonal period (s)", min_value=1, value=12,
                                                    key=f"s_{value_column}")
                                model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
                                model_fit = model.fit()
                                y_pred = model_fit.predict(start=0, end=len(y) - 1)

                            elif algorithm == "Support Vector Regression (SVR)":
                                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"],
                                                      key=f"kernel_{value_column}")
                                C = st.number_input("C", min_value=0.01, value=1.0, key=f"C_{value_column}")
                                model = SVR(kernel=kernel, C=C)
                                model.fit(X, y)
                                y_pred = model.predict(X)

                            elif algorithm == "Random Forest Regressor":
                                n_estimators = st.number_input("Number of trees", min_value=1, value=100,
                                                               key=f"n_estimators_{value_column}")
                                model = RandomForestRegressor(n_estimators=n_estimators)
                                model.fit(X, y)
                                y_pred = model.predict(X)

                            elif algorithm == "XGBoost Regressor":
                                n_estimators = st.number_input("Number of trees", min_value=1, value=100,
                                                               key=f"xgb_n_estimators_{value_column}")
                                model = XGBRegressor(n_estimators=n_estimators)
                                model.fit(X, y)
                                y_pred = model.predict(X)

                            elif algorithm == "Gradient Boosting Regressor":
                                n_estimators = st.number_input("Number of trees", min_value=1, value=100,
                                                               key=f"gbr_n_estimators_{value_column}")
                                model = GradientBoostingRegressor(n_estimators=n_estimators)
                                model.fit(X, y)
                                y_pred = model.predict(X)

                            elif algorithm == "K-Nearest Neighbors Regression (KNN)":
                                n_neighbors = st.number_input("Number of neighbors", min_value=1, value=5,
                                                              key=f"n_neighbors_{value_column}")
                                model = KNeighborsRegressor(n_neighbors=n_neighbors)
                                model.fit(X, y)
                                y_pred = model.predict(X)

                            elif algorithm == "Exponential Smoothing (ETS)":
                                trend = st.selectbox("Trend component", [None, "add", "mul"],
                                                     key=f"trend_{value_column}")
                                seasonal = st.selectbox("Seasonal component", [None, "add", "mul"],
                                                        key=f"seasonal_{value_column}")
                                seasonal_periods = st.number_input("Seasonal periods", min_value=1, value=12,
                                                                   key=f"seasonal_periods_{value_column}")
                                model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal,
                                                             seasonal_periods=seasonal_periods)
                                model_fit = model.fit()
                                y_pred = model_fit.fittedvalues

                            elif algorithm == "Vector Autoregression (VAR)":
                                maxlags = st.number_input("Max lags", min_value=1, value=12,
                                                          key=f"maxlags_{value_column}")
                                model = VAR(data[value_columns])
                                model_fit = model.fit(maxlags=maxlags)
                                y_pred = model_fit.fittedvalues[value_column]

                            if forecast_type == "Prediction":
                                error_percentage = calculate_error(y, y_pred)

                                summary_table = pd.DataFrame({
                                    'Month': data.index,
                                    'Actual': y,
                                    'Forecast': y_pred,
                                    'Error %': np.abs((y - y_pred) / y) * 100
                                })

                                for col in additional_columns_selected:
                                    if col in data.columns:
                                        summary_table[col] = z
                                        error_col = f"Error % ({col})"
                                        summary_table[error_col] = np.abs(
                                            (summary_table[col] - y) / summary_table[col]) * 100

                                st.write(f"Error Percentage for {value_column}: {error_percentage:.2f}%")
                                st.write("## Actual vs Forecast for 2023")
                                st.write(summary_table)

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=data.index, y=y, mode='lines', name='Actual'))
                                fig.add_trace(go.Scatter(x=data.index, y=y_pred, mode='lines', name='Forecast'))

                                selected_columns = st.multiselect("Select columns to display in the chart",
                                                                  additional_columns_selected)

                                for col in selected_columns:
                                    if col in data.columns:
                                        fig.add_trace(
                                            go.Scatter(x=data.index, y=data[col], mode='lines', name=f'{col}'))

                                fig.update_layout(
                                    title=f'Actual vs Forecast for {value_column}',
                                    xaxis_title='Month',
                                    yaxis_title=value_column,
                                    template='plotly_dark'
                                )

                                st.plotly_chart(fig)

                            elif forecast_type == "Forecast":
                                forecast_period = pd.date_range(start='2024-01-01', periods=12, freq='M')
                                if algorithm in ["ARIMA", "SARIMA", "Exponential Smoothing (ETS)"]:
                                    forecast = model_fit.forecast(steps=12)
                                elif algorithm == "Vector Autoregression (VAR)":
                                    forecast = model_fit.forecast(model_fit.y, steps=12)
                                    forecast = forecast[:, list(data.columns).index(value_column)]
                                else:
                                    X_future = np.arange(len(data), len(data) + len(forecast_period)).reshape(-1, 1)
                                    if algorithm == "Polynomial Regression":
                                        X_future_poly = poly.transform(X_future)
                                        forecast = model.predict(X_future_poly)
                                    else:
                                        forecast = model.predict(X_future)

                                forecast_table = pd.DataFrame({
                                    'Month': forecast_period,
                                    'Forecast': forecast
                                })
                                st.write("## Forecast for 2024")
                                st.write(forecast_table)

                                fig = go.Figure()
                                show_actual = st.sidebar.checkbox("Show Actual", value=True)
                                show_prediction = st.sidebar.checkbox("Show Prediction", value=True)
                                show_forecast = st.sidebar.checkbox("Show Forecast", value=True)

                                if show_actual:
                                    fig.add_trace(go.Scatter(x=data.index, y=y, mode='lines', name='Actual'))

                                if show_prediction:
                                    fig.add_trace(go.Scatter(x=data.index, y=y_pred, mode='lines', name='Prediction'))

                                if show_forecast:
                                    fig.add_trace(
                                        go.Scatter(x=forecast_period, y=forecast, mode='lines', name='Forecast',
                                                   line=dict(dash='dash')))

                                fig.update_layout(
                                    title=f'Forecast for {value_column} - 2024',
                                    xaxis_title='Month',
                                    yaxis_title=value_column,
                                    template='plotly_dark'
                                )

                                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error processing 'Month' column: {e}")
        else:
            st.error("The dataset must contain a 'Month' column.")


def descriptive_statistics_page():
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers

    # Title of the Streamlit app
    st.title('Descriptive Statistics Analysis App')

    # File uploader to upload the dataset
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Try to read the file as a CSV
            df = pd.read_csv(uploaded_file)
        except Exception as e_csv:
            try:
                # If CSV reading fails, try to read as Excel
                df = pd.read_excel(uploaded_file)
            except Exception as e_xlsx:
                st.error(f"Error reading file: {e_csv} {e_xlsx}")
                df = None

        if df is not None:
            # Show the dataframe
            st.write("DataFrame:")
            st.dataframe(df)

            # Select columns for analysis
            columns = st.multiselect("Select columns for analysis", options=df.columns)

            if columns:
                for col in columns:
                    if np.issubdtype(df[col].dtype, np.number):
                        st.write(f"### Analysis for column: {col}")

                        # Calculate and display statistics
                        mean_val = df[col].mean()
                        median_val = df[col].median()
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else np.nan
                        std_val = df[col].std()
                        var_val = df[col].var()
                        cv_val = (std_val / mean_val) * 100
                        outliers = detect_outliers(df, col)

                        st.write(f"Mean: {mean_val}")
                        st.write(f"Median: {median_val}")
                        st.write(f"Mode: {mode_val}")
                        st.write(f"Standard Deviation: {std_val}")
                        st.write(f"Variance: {var_val}")
                        st.write(f"Coefficient of Variation: {cv_val}")

                        # Display outliers
                        if not outliers.empty:
                            st.write("Outliers detected:")
                            st.dataframe(outliers)
                        else:
                            st.write("No outliers detected")

                        # Plotting the boxplot
                        fig, ax = plt.subplots()
                        ax.boxplot(df[col].dropna())
                        ax.set_title(f"Boxplot for {col}")
                        ax.set_ylabel(col)
                        st.pyplot(fig)
                    else:
                        st.write(f"Column {col} is not numerical and was skipped.")
            else:
                st.write("Please select at least one column for analysis.")
    else:
        st.write("Please upload a file to proceed.")

    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers


def Trend_Insights_page():
    # Set the title of the Streamlit app
    st.title("Excel File Analytics with Line Chart")

    # Upload the Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        # Load the Excel file
        excel_data = pd.ExcelFile(uploaded_file)

        # Load each sheet into a dictionary
        sheets_dict = {sheet_name: excel_data.parse(sheet_name) for sheet_name in excel_data.sheet_names}

        # Display the sheet names
        st.write("Sheets available in the Excel file:", list(sheets_dict.keys()))

        # Multiselect for columns from each sheet
        selected_columns = []
        for sheet_name, df in sheets_dict.items():
            columns = df.columns
            selected = st.multiselect(f"Select columns from {sheet_name}", columns)
            selected_columns.extend([(sheet_name, col) for col in selected])

        if selected_columns:
            # List to store data for plotting
            plot_data = {}

            # Iterate over the selected columns and extract data
            for (sheet_name, col) in selected_columns:
                if 'Month' not in sheets_dict[sheet_name].columns:
                    st.error(f"'Month' column not found in {sheet_name}")
                    continue
                plot_data[f"{sheet_name} - {col}"] = sheets_dict[sheet_name].set_index('Month')[col]

            # Create a DataFrame for plotting
            plot_df = pd.DataFrame(plot_data)

            # Normalize the data to show trends
            scaler = StandardScaler()
            plot_df_normalized = pd.DataFrame(scaler.fit_transform(plot_df), index=plot_df.index,
                                              columns=plot_df.columns)

            # Plot the line chart with normalized data using Plotly
            fig = px.line(plot_df_normalized, title='Trend Analysis',
                          labels={'value': 'Normalized Value', 'index': 'Month'})
            fig.update_layout(xaxis_title='Month', yaxis_title='Normalized Value')
            st.plotly_chart(fig)

            # Option to download the normalized plot data
            st.download_button(
                label="Download normalized data as CSV",
                data=plot_df_normalized.to_csv().encode('utf-8'),
                file_name='normalized_plot_data.csv',
                mime='text/csv'
            )


# Page navigation
page = st.sidebar.radio("Select a page",
                        ["Home", "Correlation", "ML Algorithms", "Descriptive Statistics", "Trend Insights"])

if page == "Home":
    home_page()
elif page == "Correlation":
    correlation_page()
elif page == "ML Algorithms":
    ml_algorithms_page()
elif page == "Descriptive Statistics":
    descriptive_statistics_page()
elif page == "Trend Insights":
    Trend_Insights_page()