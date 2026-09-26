"""
Physics-Informed Synthetic Data Generator for Baghewala Heavy Oil Wells
Generates data with realistic engineering relationships between CSS, wellbore, and SRP parameters
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class BaghewalaWellDataGenerator:
    """Generate physics-informed synthetic data for Baghewala field"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
        # Baghewala field constants
        self.initial_reservoir_temp = 46.0  # °C
        self.initial_reservoir_pressure = 15.0  # bar
        self.initial_oil_api = 18.0
        self.initial_viscosity = 950.0  # cP at 46°C
        
    def generate_css_parameters(self, n_cycles=100):
        """Generate CSS (Cyclic Steam Stimulation) parameters"""
        css_data = []
        
        for cycle in range(1, n_cycles + 1):
            steam_vol = np.random.uniform(60, 100)  # tons
            steam_pressure = np.random.uniform(22, 28)  # bar
            injection_duration = np.random.uniform(18, 30)  # hours
            soak_time = np.random.uniform(24, 48)  # hours
            production_cutoff = np.random.uniform(0.5, 2.0)  # tonnes/day
            
            css_data.append({
                'cycle_id': cycle,
                'steam_volume': steam_vol,
                'steam_pressure': steam_pressure,
                'injection_duration': injection_duration,
                'soak_time': soak_time,
                'production_cutoff': production_cutoff
            })
        
        return pd.DataFrame(css_data)
    
    def generate_srp_parameters(self, n_cycles=100):
        """Generate SRP (Sucker Rod Pump) parameters"""
        srp_data = []
        
        for cycle in range(1, n_cycles + 1):
            spm = np.random.uniform(3.0, 6.0)  # Strokes Per Minute
            stroke_length = np.random.uniform(70, 100)  # inches
            vfd_frequency = np.random.uniform(35, 55)  # Hz
            
            # Rod load is dependent on stroke and SPM
            base_rod_load = (stroke_length / 100) * (spm / 5) * 50  # tons
            rod_load = base_rod_load
            
            motor_power = (spm * stroke_length / 1000) * 25  # kW (simplified)
            
            srp_data.append({
                'cycle_id': cycle,
                'spm': spm,
                'stroke_length': stroke_length,
                'vfd_frequency': vfd_frequency,
                'rod_load': max(0, rod_load),
                'motor_power': motor_power
            })
        
        return pd.DataFrame(srp_data)
    
    def calculate_reservoir_response(self, css_df):
        """
        Calculate reservoir temperature and viscosity response to steam injection
        Based on physics: More steam → Higher temperature → Lower viscosity
        """
        reservoir_data = []
        
        current_temp = self.initial_reservoir_temp
        
        for idx, row in css_df.iterrows():
            cycle = row['cycle_id']
            steam_vol = row['steam_volume']
            steam_pressure = row['steam_pressure']
            injection_duration = row['injection_duration']
            soak_time = row['soak_time']
            
            # Temperature increase from steam injection (simplified thermodynamic model)
            # More steam volume → more heating
            temp_increase = (
                (steam_vol - 60) * 0.15
                + (steam_pressure - 22) * 0.25
                + (injection_duration - 18) * 0.05
            )
            
            # Peak temperature during cycle
            peak_temp = current_temp + temp_increase
            soak_retention = 1.0 - 0.08 * np.exp(-soak_time / 24.0)
            peak_temp = self.initial_reservoir_temp + (peak_temp - self.initial_reservoir_temp) * soak_retention
            current_temp = peak_temp
            
            # Viscosity inversely related to temperature (empirical correlation)
            # For heavy oil: higher T → lower viscosity
            viscosity = 1100 * np.exp(-0.015 * current_temp)
            viscosity = max(100, viscosity)  # Minimum 100 cP
            
            # Production decreases due to cooling over cycles
            production_factor = 1.0 - (cycle * 0.01)  # 1% decrease per cycle
            production_factor = max(0.3, production_factor)
            
            reservoir_data.append({
                'cycle_id': cycle,
                'reservoir_temperature': current_temp,
                'reservoir_pressure': 20.0,
                'oil_viscosity': viscosity,
                'oil_mobility': 1.0 / viscosity,  # Inverse of viscosity
                'production_potential': production_factor
            })
            
            current_temp = self.initial_reservoir_temp + (
                current_temp - self.initial_reservoir_temp
            ) * np.exp(-(soak_time / 24.0) / 10.0)
        
        return pd.DataFrame(reservoir_data)
    
    def calculate_wellbore_conditions(self, reservoir_df, css_df):
        """
        Calculate wellbore pressure drop and flow conditions
        Physics: Higher viscosity → Higher pressure drop
        """
        wellbore_data = []
        
        for idx, row in reservoir_df.iterrows():
            cycle = row['cycle_id']
            temp = row['reservoir_temperature']
            viscosity = row['oil_viscosity']
            
            # Flow rate (related to oil mobility and pressure gradient)
            mobility = row['oil_mobility']
            flow_rate = mobility * 150  # bbl/day equivalent
            flow_rate = max(0, flow_rate)

            # Wellbore pressure drop increases with viscosity and flow.
            base_pressure_drop = 8.0  # bar
            pressure_drop = base_pressure_drop + (viscosity / 500) * 2 + (flow_rate / 200) * 1.5

            # Wellbore temperature (slightly lower than reservoir)
            wellbore_temp = temp - 3.0
            
            wellbore_data.append({
                'cycle_id': cycle,
                'wellbore_temperature': wellbore_temp,
                'wellbore_pressure_drop': pressure_drop,
                'fluid_mobility': mobility,
                'estimated_flow_rate': flow_rate,
                'pump_inlet_pressure': max(0, 20.0 - pressure_drop)
            })
        
        return pd.DataFrame(wellbore_data)
    
    def calculate_srp_performance(self, srp_df, wellbore_df, reservoir_df, css_df):
        """
        Calculate SRP performance metrics
        Physics: Higher viscosity & load → Lower efficiency, higher energy
        """
        srp_perf_data = []
        
        for idx, row in srp_df.iterrows():
            cycle = row['cycle_id']
            spm = row['spm']
            stroke = row['stroke_length']
            vfd = row['vfd_frequency']
            rod_load = row['rod_load']
            
            # Get corresponding wellbore/reservoir conditions
            res_row = reservoir_df[reservoir_df['cycle_id'] == cycle].iloc[0]
            
            viscosity = res_row['oil_viscosity']
            cutoff = css_df.loc[css_df['cycle_id'] == cycle, 'production_cutoff'].iloc[0]
            steam_volume = css_df.loc[css_df['cycle_id'] == cycle, 'steam_volume'].iloc[0]
            
            # Pump displacement = SPM * Stroke * Area (simplified)
            pump_disp = (spm / 5) * (stroke / 100) * 100 * 0.4
            
            # Theoretical production without losses
            theoretical_production = pump_disp * spm  # bbl/day
            
            # Efficiency decreases with:
            # 1. Increasing viscosity
            # 2. Increasing rod load (slippage, friction)
            viscosity_factor = 1.0 - (viscosity / 2000)  # Viscosity reduces efficiency
            viscosity_factor = max(0.3, viscosity_factor)
            
            load_factor = 1.0 - (rod_load / 200)  # Load reduces efficiency
            load_factor = max(0.4, load_factor)
            
            vfd_factor = max(0.9, 1.0 - abs(vfd - 45) / 50 * 0.1)
            pump_efficiency = 0.75 * viscosity_factor * load_factor * vfd_factor
            pump_efficiency = np.clip(pump_efficiency, 0.2, 0.9)
            
            # Actual production = theoretical * efficiency
            actual_production = theoretical_production * pump_efficiency
            actual_production = max(0, actual_production)
            
            # Energy consumption increases with rod load and viscosity
            base_energy = 8.0  # kW baseline
            load_energy = (rod_load / 100) * 3
            viscosity_energy = (viscosity / 500) * 2
            spm_energy = (spm / 5) * 1
            total_energy = base_energy + load_energy + viscosity_energy + spm_energy
            
            # SOR (Steam-Oil Ratio)
            cutoff_factor = np.clip(0.97 + 0.03 * cutoff, 0.95, 1.03)
            actual_production *= cutoff_factor
            sor = steam_volume / max(actual_production * 0.5, 1.0)
            
            # Failure risk based on rod load and viscosity
            failure_risk = (rod_load / 150) * 0.5 + (viscosity / 1000) * 0.3
            failure_risk = np.clip(failure_risk, 0.0, 1.0)
            
            # Anomaly detection (rod floating occurs at high viscosity + low load)
            is_anomalous = 0
            if viscosity > 800 and rod_load < 50:
                is_anomalous = 1
            
            srp_perf_data.append({
                'cycle_id': cycle,
                'actual_production': actual_production,
                'pump_efficiency': pump_efficiency,
                'energy_consumption': total_energy,
                'sor': sor,
                'rod_load_actual': rod_load,
                'failure_risk': failure_risk,
                'is_anomalous': is_anomalous,
                'motor_current': (total_energy / 11) * 150  # Approximation: 11kV supply
            })
        
        return pd.DataFrame(srp_perf_data)
    
    def generate_complete_dataset(self, n_cycles=100, n_wells=5):
        """Generate complete synthetic dataset for multiple wells"""
        all_data = []
        
        for well_id in range(1, n_wells + 1):
            print(f"Generating data for well BW-{well_id:03d}...")
            
            # Generate base parameters
            css_df = self.generate_css_parameters(n_cycles)
            srp_df = self.generate_srp_parameters(n_cycles)
            
            # Calculate responses
            reservoir_df = self.calculate_reservoir_response(css_df)
            wellbore_df = self.calculate_wellbore_conditions(reservoir_df, css_df)
            srp_perf_df = self.calculate_srp_performance(srp_df, wellbore_df, reservoir_df, css_df)
            
            # Combine all data
            combined = pd.concat([
                css_df.set_index('cycle_id'),
                srp_df.set_index('cycle_id'),
                reservoir_df.set_index('cycle_id'),
                wellbore_df.set_index('cycle_id'),
                srp_perf_df.set_index('cycle_id')
            ], axis=1)
            
            combined['well_id'] = f'BW-{well_id:03d}'
            combined['timestamp'] = [datetime.now() + timedelta(days=i*10) for i in range(len(combined))]
            combined['cycle_id'] = combined.index
            
            all_data.append(combined)
        
        final_df = pd.concat(all_data, ignore_index=False)
        final_df = final_df.reset_index(drop=True)
        
        return final_df
    
    def save_dataset(self, df, filename='synthetic_data.csv'):
        """Save dataset to CSV"""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nData types:\n{df.dtypes}")
        return filepath


if __name__ == "__main__":
    generator = BaghewalaWellDataGenerator(seed=42)
    
    # Generate synthetic dataset
    synthetic_df = generator.generate_complete_dataset(n_cycles=100, n_wells=5)
    
    # Save to file
    filepath = generator.save_dataset(synthetic_df)
    
    # Display statistics
    print("\n" + "="*80)
    print("SYNTHETIC DATA STATISTICS")
    print("="*80)
    print(synthetic_df.describe())
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns
    print(synthetic_df[numeric_cols].corr().round(3))
