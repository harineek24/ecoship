"""
Feature Engineering Pipeline
Creates per-driver, per-week features from all data tables.
This is the shared foundation that feeds carbon, safety, and HR models.
"""

import pandas as pd
import numpy as np


def build_weekly_features(trips_df, events_df, fuel_df,
                          activity_df, drivers_df):
    """
    Create per-driver, per-week feature matrix.

    Returns DataFrame with ~150 drivers x ~26 weeks = ~3,900 rows
    """
    # Add week/month to all tables
    for df in [trips_df, events_df, fuel_df, activity_df]:
        df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week.astype(int)
        df['month'] = pd.to_datetime(df['date']).dt.month

    features = []

    for driver_id in drivers_df['driver_id']:
        driver_trips = trips_df[trips_df['driver_id'] == driver_id]
        driver_events = events_df[events_df['driver_id'] == driver_id]
        driver_fuel = fuel_df[fuel_df['driver_id'] == driver_id]
        driver_activity = activity_df[activity_df['driver_id'] == driver_id]
        driver_info = drivers_df[drivers_df['driver_id'] == driver_id].iloc[0]

        weeks = sorted(driver_trips['week'].unique())

        for week in weeks:
            w_trips = driver_trips[driver_trips['week'] == week]
            w_events = driver_events[driver_events['week'] == week]
            w_fuel = driver_fuel[driver_fuel['week'] == week]
            w_activity = driver_activity[driver_activity['week'] == week]

            total_km = w_trips['distance_km'].sum()
            total_hours = w_trips['duration_minutes'].sum() / 60

            if total_km < 1:
                continue

            # === DRIVING BEHAVIOR ===
            event_counts = w_events['event_type'].value_counts()

            hard_brakes_per_100km = event_counts.get('hard_brake', 0) / total_km * 100
            hard_accels_per_100km = event_counts.get('hard_accel', 0) / total_km * 100
            speeding_per_100km = event_counts.get('speeding', 0) / total_km * 100
            sharp_turns_per_100km = event_counts.get('sharp_turn', 0) / total_km * 100
            phone_use_per_100km = event_counts.get('phone_use', 0) / total_km * 100
            high_severity_events = (w_events['severity'] == 'high').sum()
            total_events_per_100km = len(w_events) / total_km * 100

            # === FUEL / CARBON ===
            total_fuel = w_fuel['fuel_consumed_liters'].sum()
            total_energy = w_fuel['energy_consumed_kwh'].sum()
            total_idle_minutes = w_fuel['idle_time_minutes'].sum()
            total_idle_fuel = w_fuel['idle_fuel_liters'].sum()

            fuel_per_100km = (total_fuel / total_km * 100) if total_fuel > 0 else 0
            energy_per_100km = (total_energy / total_km * 100) if total_energy > 0 else 0
            idle_time_pct = (total_idle_minutes / (total_hours * 60) * 100) if total_hours > 0 else 0
            idle_fuel_pct = (total_idle_fuel / total_fuel * 100) if total_fuel > 0 else 0

            co2_kg = total_fuel * 2.31
            co2_per_km = (co2_kg / total_km) if total_km > 0 else 0

            # === PHYSICAL WORKLOAD ===
            total_steps = w_activity['steps'].sum() if len(w_activity) > 0 else 0
            total_floors = 0
            avg_package_weight = 5.0
            total_walk_time = 0
            deliveries_with_idle = 0
            total_lifting = 0

            if len(w_activity) > 0:
                total_floors = (
                    w_activity['stairs_up_floors'].sum() +
                    w_activity['stairs_down_floors'].sum()
                )
                total_lifting = w_activity['lifting_event'].sum()
                avg_package_weight = w_activity['package_weight_kg'].mean()
                total_walk_time = (
                    w_activity['walk_to_door_seconds'].sum() +
                    w_activity['walk_back_seconds'].sum()
                ) / 60
                deliveries_with_idle = w_activity['vehicle_idle_during_delivery'].sum()

            idle_during_delivery_pct = (
                deliveries_with_idle / len(w_activity) * 100
                if len(w_activity) > 0 else 0
            )

            n_working_days = w_trips['date'].nunique()

            # === WORKLOAD ===
            daily_hours = total_hours / max(n_working_days, 1)
            daily_km = total_km / max(n_working_days, 1)
            trips_per_day = len(w_trips) / max(n_working_days, 1)

            night_trips = w_trips[
                w_trips['start_time'].apply(
                    lambda x: int(x.split(':')[0]) >= 18 or
                              int(x.split(':')[0]) < 6
                )
            ]
            night_pct = len(night_trips) / max(len(w_trips), 1) * 100

            # === DRIVER STATIC FEATURES ===
            age = driver_info['age']
            experience = driver_info['experience_years']
            shift = driver_info['shift_preference']
            region = driver_info['region']
            has_eco_training = 'eco_driving' in str(driver_info['training_completed'])
            has_safety_training = 'defensive_driving' in str(driver_info['training_completed'])

            features.append({
                'driver_id': driver_id,
                'week': week,
                'month': int(w_trips['month'].mode().iloc[0]),

                # Driving behavior
                'hard_brakes_per_100km': round(hard_brakes_per_100km, 2),
                'hard_accels_per_100km': round(hard_accels_per_100km, 2),
                'speeding_per_100km': round(speeding_per_100km, 2),
                'sharp_turns_per_100km': round(sharp_turns_per_100km, 2),
                'phone_use_per_100km': round(phone_use_per_100km, 2),
                'high_severity_events': high_severity_events,
                'total_events_per_100km': round(total_events_per_100km, 2),

                # Carbon / fuel
                'fuel_per_100km': round(fuel_per_100km, 2),
                'co2_kg': round(co2_kg, 2),
                'co2_per_km': round(co2_per_km, 4),
                'idle_time_pct': round(idle_time_pct, 1),
                'idle_fuel_pct': round(idle_fuel_pct, 1),
                'idle_during_delivery_pct': round(idle_during_delivery_pct, 1),

                # Physical workload
                'daily_steps': int(total_steps / max(n_working_days, 1)),
                'daily_floors': round(total_floors / max(n_working_days, 1), 1),
                'daily_lifting_events': round(total_lifting / max(n_working_days, 1), 1),
                'avg_package_weight_kg': round(avg_package_weight, 1),
                'daily_walk_minutes': round(total_walk_time / max(n_working_days, 1), 1),

                # Workload
                'total_km': round(total_km, 1),
                'total_hours': round(total_hours, 1),
                'daily_hours': round(daily_hours, 1),
                'daily_km': round(daily_km, 1),
                'trips_per_day': round(trips_per_day, 1),
                'night_driving_pct': round(night_pct, 1),
                'n_working_days': n_working_days,

                # Static
                'age': age,
                'experience_years': experience,
                'shift': shift,
                'region': region,
                'has_eco_training': has_eco_training,
                'has_safety_training': has_safety_training,
            })

    features_df = pd.DataFrame(features)

    # === ADD TREND FEATURES (rolling 4-week deltas) ===
    trend_cols = ['total_events_per_100km', 'fuel_per_100km',
                  'daily_hours', 'daily_steps', 'co2_per_km']

    for col in trend_cols:
        features_df[f'{col}_trend'] = (
            features_df.groupby('driver_id')[col]
            .transform(lambda x: x.diff(4))
        )

    return features_df


if __name__ == '__main__':
    print("Loading data...")
    trips_df = pd.read_csv('data/trips.csv')
    events_df = pd.read_csv('data/events.csv')
    fuel_df = pd.read_csv('data/fuel.csv')
    activity_df = pd.read_csv('data/activity.csv')
    drivers_df = pd.read_csv('data/drivers_with_style.csv')

    print("Building weekly features...")
    features_df = build_weekly_features(trips_df, events_df, fuel_df,
                                         activity_df, drivers_df)
    print(f"Feature matrix shape: {features_df.shape}")
    print(f"\nColumns: {list(features_df.columns)}")
    print(f"\nDescriptive statistics:")
    print(features_df.describe().round(2))

    features_df.to_csv('data/weekly_features.csv', index=False)
    print(f"\nSaved to data/weekly_features.csv ({features_df.shape[0]} rows x {features_df.shape[1]} columns)")
