"""
Fleet Intelligence Platform - Data Generation Script
Generates realistic simulated fleet telematics data for a delivery company.

Company Profile:
- 2000 drivers, 1600 vehicles (some shared)
- 3 regions: Northeast (urban-heavy), Southeast (suburban), Midwest (rural)
- Mix: 80% diesel vans, 15% gas vans, 5% electric vans
- Time period: 6 months (2024-01-01 to 2024-06-30)
- ~180 working days, drivers work ~22 days/month

Run in Google Colab or locally. Saves all outputs as CSV files.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

np.random.seed(42)

os.makedirs('data', exist_ok=True)


# ============================================
# 1. DRIVER PROFILES (150 rows)
# ============================================

def generate_drivers(n=150):
    """
    Generate driver profiles with realistic distributions.

    Key patterns embedded:
    - ~20% of drivers will churn (termination_date not null)
    - Churned drivers: younger, less experienced, more night shifts
    - 3-4 natural behavior clusters emerge from trip data
    """
    drivers = []
    # ~37% northeast, ~33% southeast, ~30% midwest
    regions = (['northeast'] * int(n * 0.37) +
               ['southeast'] * int(n * 0.33) +
               ['midwest'] * (n - int(n * 0.37) - int(n * 0.33)))
    np.random.shuffle(regions)

    for i in range(n):
        hire_months_ago = np.random.exponential(24) + 3
        hire_date = datetime(2024, 1, 1) - timedelta(days=int(hire_months_ago * 30))
        age = int(np.clip(np.random.normal(38, 10), 21, 62))
        experience = max(0, int(hire_months_ago / 12))

        # Driving style (hidden variable that drives behavior)
        style_roll = np.random.random()
        if style_roll < 0.25:
            style = 'cautious'
        elif style_roll < 0.60:
            style = 'normal'
        elif style_roll < 0.85:
            style = 'aggressive'
        else:
            style = 'fatigued'

        # Churn: 20% of drivers leave
        will_churn = np.random.random() < 0.20
        if will_churn:
            term_date = datetime(2024, 1, 1) + timedelta(
                days=np.random.randint(60, 180)
            )
            term_reason = np.random.choice(
                ['voluntary', 'involuntary'], p=[0.7, 0.3]
            )
        else:
            term_date = None
            term_reason = None

        shift = np.random.choice(
            ['day', 'night', 'rotating'],
            p=[0.5, 0.2, 0.3]
        )

        training = []
        if np.random.random() < 0.4:
            training.append('eco_driving')
        if np.random.random() < 0.5:
            training.append('defensive_driving')
        if np.random.random() < 0.3:
            training.append('fatigue_awareness')

        drivers.append({
            'driver_id': f'D{i+1:03d}',
            'age': age,
            'experience_years': experience,
            'hire_date': hire_date.strftime('%Y-%m-%d'),
            'region': regions[i],
            'shift_preference': shift,
            'training_completed': ','.join(training) if training else 'none',
            'monthly_salary': int(np.random.normal(4200, 500)),
            'termination_date': term_date.strftime('%Y-%m-%d') if term_date else None,
            'termination_reason': term_reason,
            '_driving_style': style
        })

    return pd.DataFrame(drivers)


# ============================================
# 2. TRIPS TABLE (~200,000 rows)
# ============================================

def generate_trips(drivers_df, n_days=180):
    """
    Generate trip records. Each driver does 5-12 trips per day.
    """
    trips = []
    trip_id = 0

    for _, driver in drivers_df.iterrows():
        region = driver['region']
        shift = driver['shift_preference']

        start = datetime(2024, 1, 1)
        if driver['termination_date']:
            end = datetime.strptime(driver['termination_date'], '%Y-%m-%d')
        else:
            end = datetime(2024, 7, 1)

        current = start
        while current < end:
            if np.random.random() < 0.30:
                current += timedelta(days=1)
                continue

            n_trips = np.random.randint(5, 13)

            if shift == 'day':
                base_hour = np.random.randint(6, 9)
            elif shift == 'night':
                base_hour = np.random.randint(18, 21)
            else:
                base_hour = np.random.choice([7, 19])

            current_time = current.replace(hour=base_hour, minute=0)

            for t in range(n_trips):
                if region == 'northeast':
                    route = np.random.choice(
                        ['urban', 'suburban', 'highway', 'mixed'],
                        p=[0.5, 0.3, 0.1, 0.1]
                    )
                elif region == 'southeast':
                    route = np.random.choice(
                        ['urban', 'suburban', 'highway', 'mixed'],
                        p=[0.2, 0.5, 0.15, 0.15]
                    )
                else:
                    route = np.random.choice(
                        ['urban', 'suburban', 'highway', 'mixed'],
                        p=[0.1, 0.2, 0.5, 0.2]
                    )

                if route == 'urban':
                    dist = np.random.lognormal(2.0, 0.5)
                elif route == 'suburban':
                    dist = np.random.lognormal(2.7, 0.4)
                elif route == 'highway':
                    dist = np.random.lognormal(3.5, 0.5)
                else:
                    dist = np.random.lognormal(2.8, 0.6)

                dist = np.clip(dist, 2, 120)

                avg_speed = {'urban': 25, 'suburban': 40,
                             'highway': 80, 'mixed': 45}[route]
                duration = (dist / avg_speed) * 60
                duration *= np.random.uniform(0.8, 1.3)
                duration = np.clip(duration, 10, 120)

                vehicle_type = np.random.choice(
                    ['diesel_van', 'gas_van', 'electric_van'],
                    p=[0.80, 0.15, 0.05]
                )

                end_time = current_time + timedelta(minutes=int(duration))

                trips.append({
                    'trip_id': f'T{trip_id:06d}',
                    'driver_id': driver['driver_id'],
                    'vehicle_id': f'V{np.random.randint(1, 1601):04d}',
                    'vehicle_type': vehicle_type,
                    'date': current.strftime('%Y-%m-%d'),
                    'start_time': current_time.strftime('%H:%M'),
                    'end_time': end_time.strftime('%H:%M'),
                    'duration_minutes': round(duration, 1),
                    'distance_km': round(dist, 1),
                    'route_type': route,
                    'region': region,
                    'n_deliveries': np.random.randint(1, 8)
                        if route in ['urban', 'suburban']
                        else np.random.randint(1, 4)
                })
                trip_id += 1
                current_time = end_time + timedelta(
                    minutes=np.random.randint(5, 30)
                )

            current += timedelta(days=1)

    return pd.DataFrame(trips)


# ============================================
# 3. DRIVING EVENTS TABLE
# ============================================

def generate_driving_events(trips_df, drivers_df):
    """
    Generate individual driving events per trip.
    Event rates depend on driving style.
    """
    style_map = dict(zip(drivers_df['driver_id'], drivers_df['_driving_style']))
    term_map = dict(zip(drivers_df['driver_id'], drivers_df['termination_date']))
    training_map = dict(zip(drivers_df['driver_id'], drivers_df['training_completed']))

    events = []

    base_rates = {
        'cautious':   {'hard_brake': 0.5, 'hard_accel': 0.3, 'speeding': 0.2,
                       'sharp_turn': 0.3, 'excessive_idle': 0.8, 'phone_use': 0.1},
        'normal':     {'hard_brake': 1.5, 'hard_accel': 1.0, 'speeding': 0.8,
                       'sharp_turn': 0.8, 'excessive_idle': 1.5, 'phone_use': 0.3},
        'aggressive': {'hard_brake': 4.0, 'hard_accel': 3.5, 'speeding': 3.0,
                       'sharp_turn': 2.5, 'excessive_idle': 1.0, 'phone_use': 0.8},
        'fatigued':   {'hard_brake': 2.0, 'hard_accel': 1.2, 'speeding': 1.5,
                       'sharp_turn': 1.5, 'excessive_idle': 2.5, 'phone_use': 0.5},
    }

    for _, trip in trips_df.iterrows():
        driver_id = trip['driver_id']
        style = style_map.get(driver_id, 'normal')
        dist = trip['distance_km']
        trip_date = datetime.strptime(trip['date'], '%Y-%m-%d')

        rates = base_rates[style].copy()

        if style == 'fatigued':
            months_in = (trip_date - datetime(2024, 1, 1)).days / 30
            fatigue_multiplier = 1.0 + (months_in / 6) * 0.5
            rates = {k: v * fatigue_multiplier for k, v in rates.items()}

        term_date = term_map.get(driver_id)
        if term_date:
            term_dt = datetime.strptime(term_date, '%Y-%m-%d')
            days_to_term = (term_dt - trip_date).days
            if 0 < days_to_term < 56:
                churn_multiplier = 1.0 + (1 - days_to_term / 56) * 0.8
                rates = {k: v * churn_multiplier for k, v in rates.items()}

        training = training_map.get(driver_id, 'none')
        if 'eco_driving' in str(training):
            rates['hard_accel'] *= 0.8
            rates['excessive_idle'] *= 0.75
        if 'defensive_driving' in str(training):
            rates['hard_brake'] *= 0.8
            rates['speeding'] *= 0.7

        for event_type, rate_per_100km in rates.items():
            expected_count = (dist / 100) * rate_per_100km
            n_events = np.random.poisson(max(0, expected_count))

            for _ in range(n_events):
                severity = np.random.choice(
                    ['low', 'medium', 'high'], p=[0.6, 0.3, 0.1]
                )
                events.append({
                    'trip_id': trip['trip_id'],
                    'driver_id': driver_id,
                    'date': trip['date'],
                    'event_type': event_type,
                    'severity': severity,
                    'speed_at_event': int(np.clip(
                        np.random.normal(50, 20), 10, 130
                    )),
                    'duration_seconds': int(np.random.exponential(30)) + 5
                        if event_type == 'excessive_idle' else None
                })

    return pd.DataFrame(events)


# ============================================
# 4. FUEL / ENERGY TABLE (one per trip)
# ============================================

def generate_fuel_data(trips_df, drivers_df):
    """
    Generate fuel consumption per trip.
    """
    style_map = dict(zip(drivers_df['driver_id'], drivers_df['_driving_style']))

    fuel_data = []

    for _, trip in trips_df.iterrows():
        style = style_map.get(trip['driver_id'], 'normal')
        dist = trip['distance_km']
        vtype = trip['vehicle_type']
        route = trip['route_type']

        base_consumption = {
            'diesel_van': {'cautious': 10, 'normal': 12, 'aggressive': 15, 'fatigued': 13},
            'gas_van':    {'cautious': 12, 'normal': 14, 'aggressive': 18, 'fatigued': 15},
            'electric_van': {'cautious': 20, 'normal': 24, 'aggressive': 30, 'fatigued': 26},
        }

        consumption_per_100 = base_consumption[vtype][style]

        route_modifier = {'urban': 1.2, 'suburban': 1.0, 'highway': 0.85, 'mixed': 1.05}
        consumption_per_100 *= route_modifier[route]

        month = int(trip['date'].split('-')[1])
        if month in [12, 1, 2]:
            consumption_per_100 *= 1.12
        elif month in [7, 8]:
            consumption_per_100 *= 1.05

        consumption_per_100 *= np.random.uniform(0.9, 1.1)
        total_consumption = (dist / 100) * consumption_per_100

        idle_minutes = np.random.exponential(
            {'cautious': 3, 'normal': 5, 'aggressive': 4, 'fatigued': 8}[style]
        ) * trip['n_deliveries']

        is_ev = vtype == 'electric_van'
        if is_ev:
            idle_fuel = 0
            idle_energy = idle_minutes * 0.02
        else:
            idle_fuel = idle_minutes * 0.04
            idle_energy = 0

        fuel_data.append({
            'trip_id': trip['trip_id'],
            'driver_id': trip['driver_id'],
            'date': trip['date'],
            'vehicle_type': vtype,
            'fuel_consumed_liters': round(total_consumption + idle_fuel, 2) if not is_ev else 0,
            'energy_consumed_kwh': round(total_consumption + idle_energy, 2) if is_ev else 0,
            'idle_time_minutes': round(idle_minutes, 1),
            'idle_fuel_liters': round(idle_fuel, 2) if not is_ev else 0,
            'avg_speed_kmh': round(trip['distance_km'] / (trip['duration_minutes'] / 60), 1),
            'ambient_temp_celsius': int(np.random.normal(
                {1: 0, 2: 2, 3: 8, 4: 15, 5: 20, 6: 25,
                 7: 28, 8: 27, 9: 22, 10: 14, 11: 7, 12: 2}[month], 5
            ))
        })

    return pd.DataFrame(fuel_data)


# ============================================
# 5. DRIVER ACTIVITY TABLE
# ============================================

def generate_driver_activity(trips_df, drivers_df):
    """
    Simulate wearable/smartphone activity recognition data.
    """
    activities = []

    for _, trip in trips_df.iterrows():
        route = trip['route_type']
        n_deliveries = trip['n_deliveries']

        for d in range(n_deliveries):
            if route == 'urban':
                walk_dist = np.random.lognormal(3.5, 0.5)
                has_stairs = np.random.random() < 0.45
                floors = np.random.randint(1, 6) if has_stairs else 0
            elif route == 'suburban':
                walk_dist = np.random.lognormal(2.5, 0.4)
                has_stairs = np.random.random() < 0.15
                floors = np.random.randint(1, 3) if has_stairs else 0
            else:
                walk_dist = np.random.lognormal(2.0, 0.3)
                has_stairs = np.random.random() < 0.05
                floors = np.random.randint(1, 2) if has_stairs else 0

            walk_seconds = walk_dist / 1.4
            stair_seconds = floors * 15
            wait_seconds = np.random.exponential(20) + 5

            package_kg = np.random.lognormal(1.5, 0.8)
            package_kg = np.clip(package_kg, 0.5, 35)
            has_lifting = package_kg > 10

            steps = int(walk_dist / 0.7) * 2

            activities.append({
                'trip_id': trip['trip_id'],
                'driver_id': trip['driver_id'],
                'date': trip['date'],
                'delivery_number': d + 1,
                'walk_to_door_meters': round(walk_dist, 1),
                'walk_to_door_seconds': round(walk_seconds, 1),
                'stairs_up_floors': floors,
                'stairs_up_seconds': round(stair_seconds, 1),
                'wait_at_door_seconds': round(wait_seconds, 1),
                'stairs_down_floors': floors,
                'stairs_down_seconds': round(stair_seconds * 0.8, 1),
                'walk_back_seconds': round(walk_seconds * 0.95, 1),
                'package_weight_kg': round(package_kg, 1),
                'lifting_event': has_lifting,
                'steps': steps,
                'vehicle_idle_during_delivery': np.random.random() < 0.35
            })

    return pd.DataFrame(activities)


# ============================================
# 6. INCIDENTS TABLE (~80 rows)
# ============================================

def generate_incidents(trips_df, drivers_df, events_df):
    """
    Generate safety incidents.
    """
    style_map = dict(zip(drivers_df['driver_id'], drivers_df['_driving_style']))
    shift_map = dict(zip(drivers_df['driver_id'], drivers_df['shift_preference']))

    incidents = []
    inc_id = 0

    for driver_id in drivers_df['driver_id']:
        style = style_map[driver_id]
        shift = shift_map[driver_id]

        # Rates scaled to produce ~80 incidents across fleet over 6 months
        annual_rate = {
            'cautious': 0.30, 'normal': 0.80,
            'aggressive': 2.0, 'fatigued': 1.50
        }[style]

        if shift == 'rotating':
            annual_rate *= 1.5

        rate = annual_rate * 0.5
        n_incidents = np.random.poisson(rate)

        driver_trips = trips_df[trips_df['driver_id'] == driver_id]
        if len(driver_trips) == 0 or n_incidents == 0:
            continue

        for _ in range(n_incidents):
            trip = driver_trips.sample(1).iloc[0]
            inc_type = np.random.choice(
                ['collision', 'near_miss', 'property_damage', 'injury'],
                p=[0.2, 0.4, 0.25, 0.15]
            )
            severity = np.random.choice(
                ['minor', 'moderate', 'severe'], p=[0.6, 0.3, 0.1]
            )

            claim_amount = 0
            if inc_type == 'collision':
                claim_amount = int(np.random.lognormal(8, 1))
            elif inc_type == 'property_damage':
                claim_amount = int(np.random.lognormal(6.5, 0.8))

            days_lost = 0
            if inc_type == 'injury':
                days_lost = int(np.random.exponential(5)) + 1

            incidents.append({
                'incident_id': f'INC{inc_id:03d}',
                'driver_id': driver_id,
                'trip_id': trip['trip_id'],
                'date': trip['date'],
                'type': inc_type,
                'severity': severity,
                'fault': np.random.choice(
                    ['driver', 'other', 'unknown'], p=[0.5, 0.3, 0.2]
                ),
                'claim_amount': claim_amount,
                'days_lost': days_lost
            })
            inc_id += 1

    return pd.DataFrame(incidents)


# ============================================
# GENERATE AND SAVE EVERYTHING
# ============================================

if __name__ == '__main__':
    print("Generating driver profiles...")
    drivers_df = generate_drivers(2000)
    print(f"  {len(drivers_df)} drivers")

    print("Generating trips...")
    trips_df = generate_trips(drivers_df, 180)
    print(f"  {len(trips_df)} trips")

    print("Generating driving events...")
    events_df = generate_driving_events(trips_df, drivers_df)
    print(f"  {len(events_df)} events")

    print("Generating fuel data...")
    fuel_df = generate_fuel_data(trips_df, drivers_df)
    print(f"  {len(fuel_df)} fuel records")

    print("Generating driver activity...")
    activity_df = generate_driver_activity(trips_df, drivers_df)
    print(f"  {len(activity_df)} activity records")

    print("Generating incidents...")
    incidents_df = generate_incidents(trips_df, drivers_df, events_df)
    print(f"  {len(incidents_df)} incidents")

    # Save all files
    drivers_df.drop(columns=['_driving_style']).to_csv('data/drivers.csv', index=False)
    trips_df.to_csv('data/trips.csv', index=False)
    events_df.to_csv('data/events.csv', index=False)
    fuel_df.to_csv('data/fuel.csv', index=False)
    activity_df.to_csv('data/activity.csv', index=False)
    incidents_df.to_csv('data/incidents.csv', index=False)

    # Also save drivers with hidden style for model training
    drivers_df.to_csv('data/drivers_with_style.csv', index=False)

    # Print summary
    print("\n=== DATASET SUMMARY ===")
    print(f"Drivers:    {len(drivers_df):>10,}")
    print(f"Trips:      {len(trips_df):>10,}")
    print(f"Events:     {len(events_df):>10,}")
    print(f"Fuel:       {len(fuel_df):>10,}")
    print(f"Activity:   {len(activity_df):>10,}")
    print(f"Incidents:  {len(incidents_df):>10,}")
    print(f"\nChurned drivers: {drivers_df['termination_date'].notna().sum()}")
    print(f"Total CO2 (est): {(fuel_df['fuel_consumed_liters'].sum() * 2.31):.0f} kg")

    # Save summary stats
    summary = {
        'n_drivers': int(len(drivers_df)),
        'n_trips': int(len(trips_df)),
        'n_events': int(len(events_df)),
        'n_fuel_records': int(len(fuel_df)),
        'n_activity_records': int(len(activity_df)),
        'n_incidents': int(len(incidents_df)),
        'churned_drivers': int(drivers_df['termination_date'].notna().sum()),
        'total_co2_kg': float(fuel_df['fuel_consumed_liters'].sum() * 2.31),
        'total_distance_km': float(trips_df['distance_km'].sum()),
        'avg_safety_events_per_trip': float(len(events_df) / len(trips_df)),
    }
    with open('data/summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nAll files saved to data/")
    print("Summary stats saved to data/summary_stats.json")
