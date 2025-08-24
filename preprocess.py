import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import random 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load NetCDF file
weather_ds = xr.open_dataset("data/merged_weather.nc")
print(weather_ds)

# Extract variables
u10 = weather_ds['u10']      # 10m u-component wind
v10 = weather_ds['v10']      # 10m v-component wind
t2m = weather_ds['t2m']      # 2m temperature
d2m = weather_ds['d2m']      # 2m dewpoint temperature
sp  = weather_ds['sp']       # surface pressure
tp  = weather_ds['tp']       # total precipitation
time = weather_ds['valid_time']
lat = weather_ds['latitude']
lon = weather_ds['longitude']

# Calculate wind speed and direction
wind_speed = np.sqrt(u10**2 + v10**2)
wind_dir = np.arctan2(v10, u10)

# Relative Humidity (approx formula using T & dewpoint)
def calc_relative_humidity(temp, dewpoint):
    return 100 * (np.exp((17.625 * dewpoint) / (243.04 + dewpoint)) /
                  np.exp((17.625 * temp) / (243.04 + temp)))

rh = calc_relative_humidity(t2m-273.15, d2m-273.15)  # Convert Kelvin → °C

print("Wind speed (min, max):", float(wind_speed.min()), float(wind_speed.max()))
print("Wind direction (min, max radians):", float(wind_dir.min()), float(wind_dir.max()))
print("Relative humidity (min, max %):", float(rh.min()), float(rh.max()))

# Load fire data
fire_df = pd.read_csv("data/fire_archive_SV-C2_xxxx.csv")
confidence_map = {"l": 0, "n": 50, "h": 100}
fire_df['confidence_num'] = fire_df['confidence'].map(confidence_map)

# Keep medium+high
fire_df = fire_df[fire_df['confidence_num'] >= 50]

fire_df['fire_label'] = 1


# Convert fire date/time to datetime
fire_df['datetime'] = pd.to_datetime(fire_df['acq_date'] + ' ' + fire_df['acq_time'].astype(str).str.zfill(4),
                                     format='%Y-%m-%d %H%M')


# Function to extract nearest weather data for a fire point
def get_weather_at_point(lat, lon, time):
    point_data = weather_ds.sel(latitude=lat, longitude=lon, valid_time=time, method="nearest")
    return {
        'u10': float(point_data['u10'].values),
        'v10': float(point_data['v10'].values),
        't2m': float(point_data['t2m'].values),
        'd2m': float(point_data['d2m'].values),
        'sp': float(point_data['sp'].values),
        'tp': float(point_data['tp'].values),
    }


# Apply to a small subset first (for speed)
sample_fires = fire_df.sample(100)  # take 100 fires for testing

weather_features = []
for _, row in sample_fires.iterrows():
    try:
        weather_features.append(get_weather_at_point(row['latitude'], row['longitude'], row['datetime']))
    except Exception as e:
        print("Skipping due to error:", e)

weather_df = pd.DataFrame(weather_features)
final_df = pd.concat([sample_fires.reset_index(drop=True), weather_df], axis=1)

print("this is final df",final_df.head())


def generate_no_fire_samples(n_samples=100):
    no_fire_records = []
    times = weather_ds['valid_time'].values
    lats = weather_ds['latitude'].values
    lons = weather_ds['longitude'].values

    for _ in range(n_samples):
        # Random time, lat, lon
        t = pd.to_datetime(str(random.choice(times)))
        lat = float(random.choice(lats))
        lon = float(random.choice(lons))

        # Check if there is a fire nearby within 0.1 degree and same day
        nearby_fires = fire_df[
            (abs(fire_df['latitude'] - lat) < 0.1) &
            (abs(fire_df['longitude'] - lon) < 0.1) &
            (fire_df['datetime'].dt.date == t.date())
        ]

        if len(nearby_fires) == 0:  # No fire here
            try:
                weather_point = get_weather_at_point(lat, lon, t)
                weather_point['latitude'] = lat
                weather_point['longitude'] = lon
                weather_point['datetime'] = t
                weather_point['fire_label'] = 0
                no_fire_records.append(weather_point)
            except:
                pass
    
    return pd.DataFrame(no_fire_records)

# Generate 500 no-fire samples
no_fire_df = generate_no_fire_samples(500)

print("success ",no_fire_df.head())

# Keep only features + label
fire_data = final_df[['u10','v10','t2m','d2m','sp','tp']]
fire_data['fire_label'] = 1

no_fire_data = no_fire_df[['u10','v10','t2m','d2m','sp','tp','fire_label']]

# Merge datasets
full_dataset = pd.concat([fire_data, no_fire_data], axis=0).dropna()

print(full_dataset['fire_label'].value_counts())
print(full_dataset.head())

X = full_dataset[['u10','v10','t2m','d2m','sp','tp']]
y = full_dataset['fire_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict probabilities
y_prob = model.predict_proba(X_test)[:,1]  # Probability of fire

def classify_risk(p):
    if p > 0.7:
        return "High"
    elif p > 0.4:
        return "Medium"
    else:
        return "Low"

risk_classes = [classify_risk(p) for p in y_prob]

# Add results to test set
results = X_test.copy()
results['true_label'] = y_test.values
results['fire_probability'] = y_prob
results['risk_class'] = risk_classes

print(results.head())
joblib.dump(model, "models/fire_model.pkl")