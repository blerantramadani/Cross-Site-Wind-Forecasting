"""
Digital Twin Engine v2.1 — Dynamic Air Density
================================================
Upgrade: Temperature now physically affects power output through
air density variation (ideal gas law: rho = P / (R*T)).

This fixes the ablation anomaly where removing temperature
paradoxically improved performance. Now temperature has a genuine
causal relationship with power output through atmospheric physics.

Authors: Blerant Ramadani, Vangel Fustic
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TURBINE PARAMETERS — PVE Bogdanci 2.3 MW Class
# ============================================================
RATED_POWER_KW = 2300
CUT_IN_SPEED = 3.0
RATED_SPEED = 12.0   # boundary for the cubic/flat split; saturation itself is enforced
                     # by min(P_cubic, P_rated) and engages near 10.4 m/s at standard density
CUT_OUT_SPEED = 25.0
ROTOR_DIAMETER = 101.0
ROTOR_AREA = np.pi * (ROTOR_DIAMETER / 2) ** 2
CP_EFFECTIVE = 0.42

# Standard pressure constant for ideal gas law
# rho(T) = P_atm / (R_specific * T_kelvin), lumped constant ≈ 353.0 (kg·K/m³)
# with P_atm = 101325 Pa
GAS_CONSTANT = 353.049


def generate_wind_speed_ou(n_samples, mu=6.5, phi=0.97, sigma=0.4,
                           alpha=1.5, seed=42, autocorrelated=True):
    """
    Generate wind speed using Ornstein-Uhlenbeck process.
    v(t) = mu + phi*(v(t-1) - mu) + sigma*eps + alpha*sin(2*pi*t/24)
    """
    np.random.seed(seed)
    epsilon = np.random.normal(0, 1, n_samples)

    if not autocorrelated:
        wind_speed = np.abs(mu + sigma * 5 * epsilon)
        return np.clip(wind_speed, 0, 30)

    wind_speed = np.zeros(n_samples)
    wind_speed[0] = mu

    for t in range(1, n_samples):
        diurnal = alpha * np.sin(2 * np.pi * t / 24)
        wind_speed[t] = mu + phi * (wind_speed[t - 1] - mu) + sigma * epsilon[t] + diurnal

    wind_speed = np.clip(wind_speed, 0, 30)
    return wind_speed


def generate_temperature(n_samples, dates, seed=42):
    """
    Generate ambient temperature with seasonal variation.
    Continental climate: ~5C in January, ~25C in July.
    """
    np.random.seed(seed + 100)
    day_of_year = dates.dayofyear
    temp_trend = 15 - 10 * np.cos(2 * np.pi * day_of_year / 365)
    temperature = temp_trend + np.random.normal(0, 2, n_samples)
    return np.round(temperature, 2)


def compute_power_output(wind_speed, temperature, add_noise=True, seed=42):
    """
    Compute power output with DYNAMIC air density.

    Key upgrade from v2.0: air density is no longer constant at 1.225.
    Instead, it varies with temperature using the ideal gas law:
        rho(T) = 353.049 / (T_celsius + 273.15)

    This means:
        - Cold air (0C) -> rho = 1.292 -> MORE power
        - Standard (15C) -> rho = 1.225 -> baseline
        - Hot air (35C) -> rho = 1.146 -> LESS power

    Temperature now has a genuine physical effect on power output,
    fixing the ablation anomaly in v2.0.
    """
    np.random.seed(seed + 200)
    power = np.zeros(len(wind_speed))

    for i in range(len(wind_speed)):
        v = wind_speed[i]
        t_celsius = temperature[i]

        if v < CUT_IN_SPEED or v >= CUT_OUT_SPEED:
            power[i] = 0.0
        elif v < RATED_SPEED:
            # Dynamic air density from ideal gas law
            t_kelvin = t_celsius + 273.15
            rho = GAS_CONSTANT / t_kelvin

            # Physical cubic power equation with dynamic density
            p = 0.5 * rho * ROTOR_AREA * CP_EFFECTIVE * (v ** 3)
            power[i] = min(p, RATED_POWER_KW)
        else:
            power[i] = RATED_POWER_KW

    if add_noise:
        noise = np.random.uniform(0.995, 1.005, len(power))
        power = power * noise
        power = np.clip(power, 0, RATED_POWER_KW)

    return np.round(power, 2)


def generate_dataset(n_years=1, start_date='2020-01-01', seed=42,
                     autocorrelated=True, mu=6.5, phi=0.97, sigma=0.4, alpha=1.5):
    """Generate complete synthetic dataset with dynamic air density."""
    start = pd.Timestamp(start_date)
    end = start + pd.DateOffset(years=n_years) - pd.Timedelta(hours=1)
    dates = pd.date_range(start=start, end=end, freq='h')
    n_samples = len(dates)

    wind_speed = generate_wind_speed_ou(
        n_samples, mu=mu, phi=phi, sigma=sigma,
        alpha=alpha, seed=seed, autocorrelated=autocorrelated
    )

    temperature = generate_temperature(n_samples, dates, seed=seed)

    # Power now depends on BOTH wind speed AND temperature
    power_output = compute_power_output(wind_speed, temperature,
                                        add_noise=True, seed=seed)

    df = pd.DataFrame({
        'Timestamp': dates,
        'WindSpeed_m_s': np.round(wind_speed, 2),
        'Temperature_C': temperature,
        'ActivePower_kW': power_output
    })

    return df


def plot_wind_profile(df, hours=200, title=None):
    """Plot wind speed profile."""
    plt.figure(figsize=(14, 4))
    plt.plot(df['Timestamp'][:hours], df['WindSpeed_m_s'][:hours],
             color='#0D9488', linewidth=0.8)
    plt.title(title or "Simulated Wind Speed Profile (Ornstein-Uhlenbeck)", fontsize=13)
    plt.ylabel("Wind Speed (m/s)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wind_profile_ou.png', dpi=150)
    plt.show()


def plot_power_curve(temperature_c=15.0):
    """Plot power curve at a given temperature."""
    wind_speeds = np.linspace(0, 27, 500)
    temp_array = np.full(500, temperature_c)
    power = compute_power_output(wind_speeds, temp_array, add_noise=False)

    plt.figure(figsize=(8, 5))
    plt.plot(wind_speeds, power, color='#0D9488', linewidth=2)
    plt.axvline(x=CUT_IN_SPEED, color='red', linestyle='--', alpha=0.6,
                label=f'Cut-in ({CUT_IN_SPEED} m/s)')
    v_rated_actual = (RATED_POWER_KW * 1000 / (0.5 * 1.225 * ROTOR_AREA * CP_EFFECTIVE)) ** (1/3)
    plt.axvline(x=v_rated_actual, color='orange', linestyle='--', alpha=0.6,
                label=f'Rated (~{v_rated_actual:.1f} m/s)')
    plt.axvline(x=CUT_OUT_SPEED, color='red', linestyle='--', alpha=0.6,
                label=f'Cut-out ({CUT_OUT_SPEED} m/s)')
    plt.title(f"Aerodynamic Power Curve — PVE Bogdanci (T={temperature_c}°C)", fontsize=13)
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Active Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('power_curve.png', dpi=150)
    plt.show()


def plot_density_effect():
    """Show how temperature affects power output at the same wind speed."""
    temps = np.linspace(-10, 40, 100)
    densities = GAS_CONSTANT / (temps + 273.15)

    # Power at 10 m/s for each temperature
    v = 10.0
    powers = 0.5 * densities * ROTOR_AREA * CP_EFFECTIVE * (v ** 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(temps, densities, color='#0D9488', linewidth=2)
    ax1.set_title("Air Density vs Temperature", fontsize=12)
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Air Density (kg/m³)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.225, color='gray', linestyle='--', alpha=0.5, label='Standard (1.225)')
    ax1.legend()

    ax2.plot(temps, powers, color='#EF4444', linewidth=2)
    ax2.set_title("Power Output vs Temperature (at v=10 m/s)", fontsize=12)
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Active Power (kW)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('density_effect.png', dpi=150)
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    print("=" * 60)
    print("Digital Twin Engine v2.1 — Dynamic Air Density")
    print("=" * 60)

    # Show the density effect
    print("\nDensity variation examples:")
    for t in [0, 15, 30]:
        rho = GAS_CONSTANT / (t + 273.15)
        print(f"  T={t:3d}°C -> rho={rho:.3f} kg/m³")

    # Dataset 1: Physics-consistent autocorrelated data
    print("\n[1/2] Generating autocorrelated dataset (OU + dynamic density)...")
    df_auto = generate_dataset(n_years=1, seed=42, autocorrelated=True)
    df_auto.to_csv('data/Synthetic_Bogdanci_OU.csv', index=False)
    print(f"  Saved: data/Synthetic_Bogdanci_OU.csv ({len(df_auto)} samples)")
    print(f"  Wind: mean={df_auto['WindSpeed_m_s'].mean():.1f}, "
          f"std={df_auto['WindSpeed_m_s'].std():.1f}")
    print(f"  Temp: mean={df_auto['Temperature_C'].mean():.1f}, "
          f"std={df_auto['Temperature_C'].std():.1f}")
    print(f"  Power: mean={df_auto['ActivePower_kW'].mean():.0f}, "
          f"max={df_auto['ActivePower_kW'].max():.0f}")

    # Dataset 2: Uncorrelated noise
    print("\n[2/2] Generating uncorrelated noise dataset...")
    df_noise = generate_dataset(n_years=1, seed=42, autocorrelated=False)
    df_noise.to_csv('data/Synthetic_Bogdanci_Noise.csv', index=False)
    print(f"  Saved: data/Synthetic_Bogdanci_Noise.csv ({len(df_noise)} samples)")

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_wind_profile(df_auto)
    plot_power_curve()
    plot_density_effect()

    print("\n✅ Digital Twin Engine v2.1 complete.")
    print("   Temperature now physically affects power through air density.")
