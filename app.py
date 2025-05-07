from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import calendar
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///electricity_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model for database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    registered_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Hardcoded admin credentials
ADMIN_USERNAME = 'admin@gmail.com'
ADMIN_PASSWORD = 'admin'

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'mohankumarb498@gmail.com'
app.config['MAIL_PASSWORD'] = 'yhkd vdiz jmir nnlw'
app.config['MAIL_DEFAULT_SENDER'] = 'abienterpriseabi@gmail.com'

mail = Mail(app)

# Create the database tables
# with app.app_context():
#     db.create_all()

# Load synthetic data (with error handling if file doesn't exist yet)
try:
    data = pd.read_csv('synthetic_data.csv', parse_dates=['Timestamp'])
except FileNotFoundError:
    # Create empty DataFrame with expected columns if file doesn't exist
    data = pd.DataFrame(columns=['Timestamp', 'Region', 'Load'])

# Route: Home
@app.route('/')
def home():
    return render_template('index.html')

# Route: User Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please login instead.', 'warning')
            return redirect(url_for('login'))
            
        otp = random.randint(1000, 9999)
        session['otp'] = otp
        session['email'] = email
        session['password'] = password

        # Send OTP via Flask-Mail
        try:
            msg = Message('OTP Verification', recipients=[email])
            msg.body = f'Your OTP is: {otp}'
            mail.send(msg)
            flash('OTP sent to your email. Please verify.', 'info')
            return redirect(url_for('verify_otp'))
        except Exception as e:
            flash(f'Error sending OTP: {str(e)}', 'danger')
            return redirect(url_for('signup'))

    return render_template('signup.html')

# Route: Verify OTP
@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        user_otp = request.form['otp']
        if 'otp' in session and int(user_otp) == session['otp']:
            email = session.pop('email')
            password = session.pop('password')
            
            # Hash the password and store the user in the database
            hashed_password = generate_password_hash(password)
            new_user = User(email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            
            flash('Signup successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid OTP. Please try again.', 'danger')

    return render_template('verify_otp.html')

# Route: User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if email == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user'] = email
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')

# Route: User Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user' not in session and 'admin' not in session:
        flash('Please login to access this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# Route: Admin Dashboard
@app.route('/admin')
def admin_dashboard():
    if 'admin' in session:
        users = User.query.all()
        return render_template('admin_dashboard.html', users=users)
    else:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

# Route: Delete User (for admin)
@app.route('/delete-user', methods=['POST'])
def delete_user():
    if 'admin' in session:
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            db.session.delete(user)
            db.session.commit()
            flash(f'User {email} deleted successfully.', 'success')
        else:
            flash('User not found.', 'danger')
        return redirect(url_for('admin_dashboard'))
    else:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

# Route: Visualize
@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if 'user' not in session and 'admin' not in session:
        flash('Please login to access this page.', 'warning')
        return redirect(url_for('login'))
        
    # Get available regions from data
    available_regions = sorted(data['Region'].unique()) if not data.empty else ['No data available']
    selected_region = request.args.get('region', available_regions[0] if available_regions else 'No data')
    
    if data.empty or selected_region == 'No data available':
        # Handle case where no data is available
        return render_template('visualize.html', plot_url=None, region=selected_region, 
                              regions=available_regions, error="No data available for visualization")
    
    filtered_data = data[data['Region'] == selected_region]
    
    # Check if we have enough data for decomposition
    if len(filtered_data) < 48:  # Need at least 2 periods (24*2)
        return render_template('visualize.html', plot_url=None, region=selected_region, 
                              regions=available_regions, error="Insufficient data for seasonal decomposition")
    
    # Create a copy of the filtered data for enhancement
    filtered_data_enhanced = filtered_data.copy()
    
    # Generate synthetic data for all seasons if we only have one season
    # Get the range of months in our dataset
    if filtered_data_enhanced['Timestamp'].dt.month.nunique() < 4:
        # Create base entries to derive seasonal patterns from
        base_winter_entries = filtered_data_enhanced.copy().reset_index(drop=True)
        
        # Create entries for Spring (March-May)
        spring_entries = base_winter_entries.copy()
        spring_entries['Timestamp'] = spring_entries['Timestamp'].apply(
            lambda x: x.replace(month=random.choice([3, 4, 5]), day=random.randint(1, 28))
        )
        # Adjust load values for spring (slightly lower than winter)
        spring_entries['Load'] = spring_entries['Load'] * np.random.uniform(0.8, 0.95, len(spring_entries))
        
        # Create entries for Summer (June-August)
        summer_entries = base_winter_entries.copy()
        summer_entries['Timestamp'] = summer_entries['Timestamp'].apply(
            lambda x: x.replace(month=random.choice([6, 7, 8]), day=random.randint(1, 28))
        )
        # Adjust load values for summer (much higher due to cooling)
        summer_entries['Load'] = summer_entries['Load'] * np.random.uniform(1.2, 1.4, len(summer_entries))
        
        # Create entries for Rain (September-November)
        rain_entries = base_winter_entries.copy()
        rain_entries['Timestamp'] = rain_entries['Timestamp'].apply(
            lambda x: x.replace(month=random.choice([9, 10, 11]), day=random.randint(1, 28))
        )
        # Adjust load values for rain (moderate)
        rain_entries['Load'] = rain_entries['Load'] * np.random.uniform(0.9, 1.1, len(rain_entries))
        
        # Combine all seasonal data
        filtered_data_enhanced = pd.concat([
            filtered_data_enhanced,  # Original winter data
            spring_entries,
            summer_entries,
            rain_entries
        ]).reset_index(drop=True)
    
    # Add month information to the data for seasonal analysis
    filtered_data_enhanced['Month'] = filtered_data_enhanced['Timestamp'].dt.month
    
    # Define seasons based on month
    def get_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Rain'
        else:  # month in [12, 1, 2]
            return 'Winter'
    
    # Define rain season (assuming monsoon months - can be adjusted based on region)
    def get_rain_status(month):
        if month in [6, 7, 8, 9]:  # Monsoon months
            return 'Rainy Season'
        else:
            return 'Non-Rainy Season'
    
    filtered_data_enhanced['Season'] = filtered_data_enhanced['Month'].apply(get_season)
    filtered_data_enhanced['Rain'] = filtered_data_enhanced['Month'].apply(get_rain_status)
    
    # Add time features for Gradient Boosting prediction
    filtered_data_enhanced['Hour'] = filtered_data_enhanced['Timestamp'].dt.hour
    filtered_data_enhanced['Day'] = filtered_data_enhanced['Timestamp'].dt.day
    filtered_data_enhanced['DayOfWeek'] = filtered_data_enhanced['Timestamp'].dt.dayofweek
    filtered_data_enhanced['Is_Weekend'] = (filtered_data_enhanced['DayOfWeek'] >= 5).astype(int)

    # Prepare data for Gradient Boosting model
    features = ['Hour', 'Day', 'DayOfWeek', 'Is_Weekend', 'Month']
    X = filtered_data_enhanced[features]
    y = filtered_data_enhanced['Load']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = gb_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Generate data for next month prediction
    last_date = filtered_data_enhanced['Timestamp'].max()
    next_month_start = last_date + timedelta(days=1)
    next_month_end = last_date + timedelta(days=30)
    
    next_month_dates = pd.date_range(start=next_month_start, end=next_month_end, freq='H')
    next_month_data = pd.DataFrame({
        'Timestamp': next_month_dates,
        'Hour': next_month_dates.hour,
        'Day': next_month_dates.day,
        'DayOfWeek': next_month_dates.dayofweek,
        'Is_Weekend': (next_month_dates.dayofweek >= 5).astype(int),
        'Month': next_month_dates.month
    })
    
    # Predict the load for the next month
    next_month_predictions = gb_model.predict(next_month_data[features])
    next_month_data['Predicted_Load'] = next_month_predictions
    
    # Generate visualization for next month prediction
    fig_prediction = px.line(
        next_month_data,
        x='Timestamp',
        y='Predicted_Load',
        title=f'Next Month Load Prediction for {selected_region}',
        labels={'Predicted_Load': 'Predicted Load (kW)', 'Timestamp': 'Date'}
    )
    
    # Add shaded area for prediction uncertainty (±10% for visualization)
    fig_prediction.add_traces([
        go.Scatter(
            name='Upper Bound',
            x=next_month_data['Timestamp'],
            y=next_month_predictions * 1.1,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=next_month_data['Timestamp'],
            y=next_month_predictions * 0.9,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    
    # Create a comparison chart of actual vs predicted values
    comparison_data = pd.DataFrame({
        'Timestamp': filtered_data_enhanced['Timestamp'].iloc[-48:],  # Last 2 days of actual data
        'Actual': filtered_data_enhanced['Load'].iloc[-48:],
        'Predicted': gb_model.predict(filtered_data_enhanced[features].iloc[-48:])
    })
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(
        x=comparison_data['Timestamp'],
        y=comparison_data['Actual'],
        mode='lines+markers',
        name='Actual Load'
    ))
    fig_comparison.add_trace(go.Scatter(
        x=comparison_data['Timestamp'],
        y=comparison_data['Predicted'],
        mode='lines+markers',
        name='Predicted Load'
    ))
    fig_comparison.update_layout(
        title=f'Actual vs Predicted Load for {selected_region}',
        xaxis_title='Timestamp',
        yaxis_title='Load (kW)'
    )
    
    # Create a seasonal forecast chart
    # Group data by month and calculate average load
    monthly_avg = filtered_data_enhanced.groupby('Month')['Load'].mean().reset_index()
    
    # Create a dataframe with all 12 months
    all_months = pd.DataFrame({'Month': range(1, 13)})
    seasonal_forecast = all_months.merge(monthly_avg, on='Month', how='left')
    
    # For missing months, interpolate values
    seasonal_forecast['Load'] = seasonal_forecast['Load'].interpolate(method='polynomial', order=3)
    seasonal_forecast['Month_Name'] = seasonal_forecast['Month'].apply(lambda m: calendar.month_abbr[m])
    
    # Create a seasonal forecast chart
    fig_seasonal = px.line(
        seasonal_forecast,
        x='Month_Name',
        y='Load',
        title=f'Seasonal Load Forecast for {selected_region}',
        labels={'Load': 'Average Load (kW)', 'Month_Name': 'Month'},
        markers=True
    )
    
    # Create a bar chart showing model performance metrics
    metrics_data = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R² Score'],
        'Value': [rmse, mae, r2]
    })
    
    fig_metrics = px.bar(
        metrics_data,
        x='Metric',
        y='Value',
        title='Model Performance Metrics',
        text_auto='.3f',
        color='Metric'
    )
    
    # Create a feature importance chart
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance,
        x='Feature',
        y='Importance',
        title='Feature Importance for Load Prediction',
        color='Importance'
    )
    
    # Decompose time series
    try:
        decomposition = seasonal_decompose(filtered_data['Load'], model='additive', period=24)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Plot
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        ax[0].plot(filtered_data['Timestamp'], filtered_data['Load'], label='Original', color='#4e73df')
        ax[0].set_title('Original Load')
        ax[1].plot(filtered_data['Timestamp'], trend, label='Trend', color='#1cc88a')
        ax[1].set_title('Trend')
        ax[2].plot(filtered_data['Timestamp'], seasonal, label='Seasonal', color='#f6c23e')
        ax[2].set_title('Seasonal')
        ax[3].plot(filtered_data['Timestamp'], residual, label='Residual', color='#e74a3b')
        ax[3].set_title('Residual')

        for a in ax:
            a.legend()
            a.grid(True, linestyle='--', alpha=0.7)

        # Save plot to a string buffer
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        buf.close()
        
        # Bar chart: Average Load by Region
        region_avg = data.groupby('Region')['Load'].mean().reset_index()
        fig_bar = px.bar(
            region_avg,
            x='Region', 
            y='Load',
            color='Region',
            title='Average Load by Region'
        )

        # Stem chart: Peak Load Analysis
        peak_loads = data.groupby('Timestamp')['Load'].sum().reset_index()
        peak_times = peak_loads[peak_loads['Load'] > 400]  # Filter for peak loads above 400 kW

        fig_stem = go.Figure()
        fig_stem.add_trace(go.Scatter(
            x=peak_loads['Timestamp'], 
            y=peak_loads['Load'], 
            mode='markers+lines',
            marker=dict(color='red', size=8),
            name='Load'
        ))
        fig_stem.update_layout(
            title="Total Load Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Total Load (kW)",
            showlegend=True
        )

        # Additional Graphs
        # 1. Line chart: Load over time for the selected region
        fig_line = px.line(
            filtered_data,
            x='Timestamp',
            y='Load',
            title=f'Load Over Time for {selected_region}',
            labels={'Load': 'Load (kW)', 'Timestamp': 'Time'}
        )

        # 2. Pie chart: Load distribution by region
        region_sum = data.groupby('Region')['Load'].sum().reset_index()
        fig_pie = px.pie(
            region_sum,
            names='Region',
            values='Load',
            title='Load Distribution by Region'
        )

        # 3. Histogram: Load frequency distribution
        fig_hist = px.histogram(
            filtered_data,
            x='Load',
            nbins=20,
            title=f'Load Frequency Distribution for {selected_region}',
            labels={'Load': 'Load (kW)'}
        )
        
        # Seasonal Analysis Charts
        # 1. Box plot of load by season
        fig_season_box = px.box(
            filtered_data_enhanced, 
            x='Season', 
            y='Load', 
            color='Season',
            title=f'Load Distribution by Season for {selected_region}',
            labels={'Load': 'Load (kW)', 'Season': 'Season'},
            category_orders={"Season": ["Spring", "Summer", "Rain", "Winter"]}
        )
        
        # 2. Bar chart of average load by season
        season_avg = filtered_data_enhanced.groupby('Season')['Load'].mean().reset_index()
        fig_season_bar = px.bar(
            season_avg, 
            x='Season', 
            y='Load', 
            color='Season',
            title=f'Average Load by Season for {selected_region}',
            labels={'Load': 'Average Load (kW)', 'Season': 'Season'},
            category_orders={"Season": ["Spring", "Summer", "Rain", "Winter"]}
        )
        
        # 3. Bar chart of peak load by season
        season_max = filtered_data_enhanced.groupby('Season')['Load'].max().reset_index()
        fig_season_peak = px.bar(
            season_max, 
            x='Season', 
            y='Load', 
            color='Season',
            title=f'Peak Load by Season for {selected_region}',
            labels={'Load': 'Peak Load (kW)', 'Season': 'Season'},
            category_orders={"Season": ["Spring", "Summer", "Rain", "Winter"]}
        )
        
        # 4. Bar chart of minimum load by season
        season_min = filtered_data_enhanced.groupby('Season')['Load'].min().reset_index()
        fig_season_min = px.bar(
            season_min, 
            x='Season', 
            y='Load', 
            color='Season',
            title=f'Minimum Load by Season for {selected_region}',
            labels={'Load': 'Minimum Load (kW)', 'Season': 'Season'},
            category_orders={"Season": ["Spring", "Summer", "Rain", "Winter"]}
        )
        
        # 5. Line chart comparing rainy vs non-rainy season
        rain_avg = filtered_data_enhanced.groupby(['Rain', filtered_data_enhanced['Timestamp'].dt.hour])['Load'].mean().reset_index()
        rain_avg.columns = ['Rain', 'Hour', 'Average Load']
        
        fig_rain = px.line(
            rain_avg, 
            x='Hour', 
            y='Average Load', 
            color='Rain',
            title=f'Average Daily Load Profile: Rainy vs Non-Rainy Season for {selected_region}',
            labels={'Average Load': 'Average Load (kW)', 'Hour': 'Hour of Day'}
        )
        
        # 6. Monthly average load
        monthly_avg = filtered_data_enhanced.groupby('Month')['Load'].mean().reset_index()
        monthly_avg['Month Name'] = monthly_avg['Month'].apply(lambda x: calendar.month_name[x])
        monthly_avg = monthly_avg.sort_values('Month')
        
        fig_monthly = px.line(
            monthly_avg,
            x='Month Name',
            y='Load',
            title=f'Monthly Average Load for {selected_region}',
            labels={'Load': 'Average Load (kW)', 'Month Name': 'Month'},
            markers=True
        )
        
        # 7. Bar chart comparing weekday vs weekend by season
        filtered_data_enhanced['Is Weekend'] = filtered_data_enhanced['Timestamp'].dt.dayofweek >= 5
        filtered_data_enhanced['Day Type'] = filtered_data_enhanced['Is Weekend'].map({True: 'Weekend', False: 'Weekday'})
        
        season_day_type = filtered_data_enhanced.groupby(['Season', 'Day Type'])['Load'].mean().reset_index()
        fig_season_day_type = px.bar(
            season_day_type,
            x='Season',
            y='Load',
            color='Day Type',
            barmode='group',
            title=f'Average Load by Season and Day Type for {selected_region}',
            labels={'Load': 'Average Load (kW)', 'Season': 'Season', 'Day Type': 'Day Type'},
            category_orders={"Season": ["Spring", "Summer", "Rain", "Winter"]}
        )

        # 8. Bar chart of hourly peaks by season
        hour_season_peak = filtered_data_enhanced.groupby(['Season', filtered_data_enhanced['Timestamp'].dt.hour])['Load'].max().reset_index()
        hour_season_peak.columns = ['Season', 'Hour', 'Peak Load']
        
        peak_hours = {}
        for season in hour_season_peak['Season'].unique():
            season_data = hour_season_peak[hour_season_peak['Season'] == season]
            peak_hour = season_data.loc[season_data['Peak Load'].idxmax()]['Hour']
            peak_hours[season] = int(peak_hour)
        
        fig_hour_peaks = px.bar(
            hour_season_peak.pivot(index='Hour', columns='Season', values='Peak Load').reset_index(),
            x='Hour',
            y=["Spring", "Summer", "Rain", "Winter"],
            title=f'Hourly Peak Load by Season for {selected_region}',
            labels={'value': 'Peak Load (kW)', 'Hour': 'Hour of Day'},
            barmode='group'
        )

        # Suggestion for peak load time
        if not peak_times.empty:
            # Create a list of suggestions
            suggestions = [
                f"⚠️ Peak load at {row['Timestamp']} with a total load of {row['Load']:.2f} kW"
                for _, row in peak_times.iterrows()
            ]
        else:
            suggestions = ["No peak load above 400 kW detected."]

        # Get seasonal statistics for insights
        seasonal_stats = filtered_data_enhanced.groupby('Season')['Load'].agg(['mean', 'max', 'min']).reset_index()
        seasonal_stats = seasonal_stats.round(2)
        
        seasonal_insights = []
        for _, row in seasonal_stats.iterrows():
            seasonal_insights.append(f"{row['Season']} - Average: {row['mean']:.2f} kW, Peak: {row['max']:.2f} kW")
            
        # Add insights about peak hours
        peak_hour_insights = []
        for season, hour in peak_hours.items():
            ampm = "AM" if hour < 12 else "PM"
            display_hour = hour if hour <= 12 else hour - 12
            if display_hour == 0:
                display_hour = 12
            peak_hour_insights.append(f"{season}: Peak typically occurs at {display_hour} {ampm}")

        return render_template('visualize.html', 
                              plot_url=plot_url, 
                              region=selected_region, 
                              regions=available_regions, 
                              error=None, 
                              # New prediction charts (at the top)
                              prediction_chart=fig_prediction.to_html(full_html=False),
                              comparison_chart=fig_comparison.to_html(full_html=False),
                              seasonal_forecast_chart=fig_seasonal.to_html(full_html=False),
                              metrics_chart=fig_metrics.to_html(full_html=False),
                              feature_importance_chart=fig_importance.to_html(full_html=False),
                              gb_metrics={
                                  'rmse': f"{rmse:.2f}",
                                  'mae': f"{mae:.2f}",
                                  'r2': f"{r2:.2f}"
                              },
                              # Original charts
                              bar_chart=fig_bar.to_html(full_html=False), 
                              stem_chart=fig_stem.to_html(full_html=False), 
                              line_chart=fig_line.to_html(full_html=False),
                              pie_chart=fig_pie.to_html(full_html=False),
                              hist_chart=fig_hist.to_html(full_html=False),
                              season_box_chart=fig_season_box.to_html(full_html=False),
                              season_bar_chart=fig_season_bar.to_html(full_html=False),
                              season_peak_chart=fig_season_peak.to_html(full_html=False),
                              season_min_chart=fig_season_min.to_html(full_html=False),
                              rain_chart=fig_rain.to_html(full_html=False),
                              monthly_chart=fig_monthly.to_html(full_html=False),
                              season_day_type_chart=fig_season_day_type.to_html(full_html=False),
                              hour_peaks_chart=fig_hour_peaks.to_html(full_html=False),
                              suggestions=suggestions,
                              seasonal_insights=seasonal_insights,
                              peak_hour_insights=peak_hour_insights,
                              has_synthetic_seasons=filtered_data['Timestamp'].dt.month.nunique() < 4)
                              
    except Exception as e:
        return render_template('visualize.html', plot_url=None, region=selected_region, 
                              regions=available_regions, error=f"Error generating visualization: {str(e)}")

# Route: Upload Dataset
@app.route('/upload', methods=['GET', 'POST'])
def upload_dataset():
    if 'user' not in session and 'admin' not in session:
        flash('Please login to access this page.', 'warning')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            try:
                # Save the uploaded file
                file_path = 'uploaded_dataset.csv'
                file.save(file_path)
                
                # Validate the file format
                df = pd.read_csv(file_path)
                required_columns = ['Timestamp', 'Region', 'Load']
                if not all(col in df.columns for col in required_columns):
                    flash('CSV file must contain Timestamp, Region, and Load columns.', 'danger')
                    return redirect(request.url)
                
                # If valid, use it as the current dataset
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.to_csv('synthetic_data.csv', index=False)
                
                # Update the global data variable
                global data
                data = df
                
                flash('File uploaded and validated successfully!', 'success')
                return redirect(url_for('dashboard'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Only CSV files are allowed.', 'danger')
            return redirect(request.url)

    return render_template('upload.html')

# Route: Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)