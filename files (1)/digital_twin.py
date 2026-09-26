"""
Digital Twin Core Models
Reservoir Twin, Wellbore Twin, and SRP Twin integrated into one system
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class WellState:
    """Represents current well state across all domains"""
    reservoir_temperature: float
    reservoir_pressure: float
    oil_viscosity: float
    oil_mobility: float
    wellbore_temp: float
    wellbore_pressure_drop: float
    pump_inlet_pressure: float
    current_production: float
    current_efficiency: float
    current_energy: float
    current_sor: float
    rod_load: float
    failure_risk: float
    is_anomalous: int
    anomaly_type: str = "Normal Operation"


class ReservoirTwin:
    """
    Models reservoir behavior during CSS cycle
    Key relationships:
    - Steam injection → Temperature increase
    - Higher temperature → Lower viscosity
    - Viscosity decrease → Better mobility
    - Cooling over time → Viscosity increase
    """
    
    def __init__(self, initial_temp=46.0, initial_viscosity=950.0):
        self.initial_temp = initial_temp
        self.initial_viscosity = initial_viscosity
        self.current_temp = initial_temp
        self.current_viscosity = initial_viscosity
        self.cycle_history = []
    
    def simulate_steam_injection(self, steam_volume: float, steam_pressure: float,
                                injection_duration: float) -> Dict:
        """
        Simulate temperature response to steam injection
        
        Physics Model:
        Temperature_increase = (steam_volume - 60) * 0.15
        This is a simplified model based on heat transfer
        """
        # Prototype heat balance: all CSS controls contribute to thermal response.
        heat_factor = (
            (steam_volume - 60) * 0.15
            + (steam_pressure - 22) * 0.25
            + (injection_duration - 18) * 0.05
        )
        temp_increase = max(0, heat_factor)
        
        # Peak temperature reached
        peak_temp = self.current_temp + temp_increase
        self.current_temp = peak_temp
        
        # Calculate viscosity at new temperature
        # Empirical: viscosity = 1100 * exp(-0.015 * T)
        self.current_viscosity = 1100 * np.exp(-0.015 * self.current_temp)
        
        return {
            'peak_temperature': peak_temp,
            'temperature_increase': temp_increase,
            'viscosity_at_peak': self.current_viscosity,
            'oil_mobility': 1.0 / self.current_viscosity
        }
    
    def simulate_cooling(self, days_elapsed: float) -> Dict:
        """
        Simulate temperature cooling after steam injection
        Exponential cooling model
        """
        # Cooling rate: exponential return toward the initial reservoir state.
        cooling_rate = (self.current_temp - self.initial_temp) * (1 - np.exp(-days_elapsed / 10))
        self.current_temp = max(self.initial_temp, self.current_temp - cooling_rate)
        
        # Update viscosity as temperature drops
        self.current_viscosity = 1100 * np.exp(-0.015 * self.current_temp)
        
        return {
            'current_temperature': self.current_temp,
            'current_viscosity': self.current_viscosity,
            'oil_mobility': 1.0 / self.current_viscosity,
            'viscosity_increase_percent': ((self.current_viscosity - self.initial_viscosity) / 
                                          self.initial_viscosity) * 100
        }
    
    def get_production_potential(self) -> float:
        """Production potential decreases as viscosity increases"""
        viscosity_ratio = self.initial_viscosity / self.current_viscosity
        # Peak production at lower viscosity
        return min(1.0, viscosity_ratio * 0.8)
    
    def get_state(self) -> Dict:
        """Get current reservoir state"""
        return {
            'temperature': self.current_temp,
            'viscosity': self.current_viscosity,
            'mobility': 1.0 / self.current_viscosity,
            'production_potential': self.get_production_potential()
        }


class WellboreTwin:
    """
    Models wellbore behavior
    Key relationships:
    - Higher viscosity → Higher pressure drop
    - Higher flow rate → Higher pressure drop
    - Pressure drop affects pump inlet conditions
    """
    
    def __init__(self, depth: float = 1500, diameter: float = 0.05):
        self.depth = depth  # meters
        self.diameter = diameter  # meters
        self.base_pressure_drop = 8.0  # bar
    
    def calculate_pressure_drop(self, oil_viscosity: float, flow_rate: float) -> float:
        """
        Darcy-Weisbach equation simplified
        Pressure drop ∝ viscosity and flow rate
        """
        # Friction factor increases with viscosity (heavy oil effect)
        friction_component = (oil_viscosity / 500) * 2
        
        # Flow resistance component
        flow_component = (flow_rate / 200) * 1.5 if flow_rate > 0 else 0
        
        total_drop = self.base_pressure_drop + friction_component + flow_component
        return max(0, total_drop)
    
    def simulate_flow(self, oil_viscosity: float, reservoir_pressure: float,
                     oil_mobility: float) -> Dict:
        """
        Simulate flow through wellbore
        Flow rate depends on mobility and pressure gradient
        """
        # Flow rate proportional to mobility (inverse of viscosity)
        base_flow = oil_mobility * 150
        flow_rate = max(0, base_flow)
        
        # Calculate pressure drop
        pressure_drop = self.calculate_pressure_drop(oil_viscosity, flow_rate)
        
        # Pump inlet pressure (reservoir - drop)
        pump_inlet_pressure = reservoir_pressure - pressure_drop
        pump_inlet_pressure = max(0, pump_inlet_pressure)
        
        # Wellbore temperature decreases slightly from reservoir
        # (simplified - no detailed thermal model)
        wellbore_temp = 45.0 + (oil_viscosity / 1000)  # Heuristic
        
        return {
            'flow_rate': flow_rate,
            'pressure_drop': pressure_drop,
            'pump_inlet_pressure': pump_inlet_pressure,
            'wellbore_temperature': wellbore_temp,
            'flow_resistance_index': pressure_drop / max(flow_rate, 0.1)
        }
    
    def get_state(self) -> Dict:
        return {
            'base_pressure_drop': self.base_pressure_drop,
            'depth': self.depth,
            'diameter': self.diameter
        }


class SRPTwin:
    """
    Models Sucker Rod Pump (Surface) performance
    Key relationships:
    - Higher SPM & stroke → Higher theoretical production
    - Higher viscosity → Lower efficiency
    - Higher rod load → Lower efficiency, higher failure risk
    - Energy consumption increases with load and viscosity
    """
    
    def __init__(self, design_rod_load: float = 100, design_power: float = 50):
        self.design_rod_load = design_rod_load
        self.design_power = design_power
    
    def calculate_pump_displacement(self, spm: float, stroke_length: float) -> float:
        """
        Pump displacement = SPM * Stroke * Area
        Simplified: displacement ∝ spm * stroke
        """
        # Area factor (simplification)
        area_factor = 0.4  # bbl per stroke per unit stroke length
        displacement = (spm / 5) * (stroke_length / 100) * 100 * area_factor
        return displacement
    
    def calculate_pump_efficiency(self, oil_viscosity: float, rod_load: float,
                                 vfd_frequency: float) -> float:
        """
        Pump efficiency depends on:
        1. Oil viscosity (higher viscosity → lower efficiency)
        2. Rod load (higher load → more friction, lower efficiency)
        3. VFD frequency (affects synchronization)
        
        Empirical model based on field experience
        """
        # Base efficiency
        base_efficiency = 0.75
        
        # Viscosity effect (heavy oil reduces efficiency)
        viscosity_factor = 1.0 - (oil_viscosity / 2000)
        viscosity_factor = max(0.3, viscosity_factor)
        
        # Rod load effect (higher load reduces efficiency)
        load_factor = 1.0 - (rod_load / 200)
        load_factor = max(0.4, load_factor)
        
        # VFD optimization factor (optimal around 45 Hz)
        vfd_deviation = abs(vfd_frequency - 45) / 50
        vfd_factor = 1.0 - (vfd_deviation * 0.1)
        vfd_factor = max(0.9, vfd_factor)
        
        efficiency = base_efficiency * viscosity_factor * load_factor * vfd_factor
        return np.clip(efficiency, 0.2, 0.9)
    
    def calculate_energy_consumption(self, spm: float, stroke_length: float,
                                    rod_load: float, oil_viscosity: float) -> float:
        """
        Energy consumption model
        Base load + Load component + Viscosity component + SPM component
        """
        base_energy = 8.0  # kW baseline
        
        # Load-based energy
        load_energy = (rod_load / 100) * 3
        
        # Viscosity-based energy (heavy oil requires more pumping power)
        viscosity_energy = (oil_viscosity / 500) * 2
        
        # SPM-based energy
        spm_energy = (spm / 5) * 1
        
        total_energy = base_energy + load_energy + viscosity_energy + spm_energy
        return max(0, total_energy)
    
    def calculate_failure_risk(self, rod_load: float, spm: float,
                              oil_viscosity: float) -> float:
        """
        Failure risk prediction
        Based on:
        1. Rod load (overstress)
        2. SPM (vibration, fatigue)
        3. Oil viscosity (impact loads)
        """
        # Rod load component (primary failure cause)
        load_risk = (rod_load / 150) * 0.5
        
        # Viscosity component (impact loading in heavy oil)
        viscosity_risk = (oil_viscosity / 1000) * 0.3
        
        # SPM component (fatigue)
        spm_risk = (spm / 6) * 0.2
        
        total_risk = load_risk + viscosity_risk + spm_risk
        return np.clip(total_risk, 0.0, 1.0)
    
    def detect_anomaly(self, rod_load: float, oil_viscosity: float,
                       spm: float, efficiency: float) -> Tuple[int, str]:
        """
        Detect anomalous conditions
        Returns: (is_anomalous, anomaly_type)
        """
        
        # Rod floating detection: high viscosity + low load
        if oil_viscosity > 800 and rod_load < 50:
            return 1, "Rod Floating"
        
        # Impact loading: high viscosity + high SPM
        if oil_viscosity > 750 and spm > 5.5:
            return 1, "Impact Loading"
        
        # Pump inefficiency: low efficiency + normal conditions
        if efficiency < 0.35 and rod_load > 50:
            return 1, "Pump Inefficiency"
        
        # Overload: rod load exceeds design
        if rod_load > 150:
            return 1, "Rod Overload"
        
        return 0, "Normal Operation"
    
    def simulate_operation(self, spm: float, stroke_length: float,
                          vfd_frequency: float, oil_viscosity: float,
                          pump_inlet_pressure: float) -> Dict:
        """
        Complete SRP operation simulation
        """
        # Calculate pump displacement
        pump_disp = self.calculate_pump_displacement(spm, stroke_length)
        
        # Theoretical production (no losses)
        theoretical_production = pump_disp * spm
        
        # Estimate rod load (simplified)
        # Higher stroke, higher SPM, higher viscosity → higher rod load
        base_rod_load = (stroke_length / 100) * (spm / 5) * 50
        viscosity_load_factor = (oil_viscosity / 500)
        rod_load = max(0, base_rod_load * viscosity_load_factor)
        
        # Calculate pump efficiency
        efficiency = self.calculate_pump_efficiency(oil_viscosity, rod_load, vfd_frequency)
        
        # Actual production = theoretical * efficiency
        actual_production = theoretical_production * efficiency
        actual_production = max(0, actual_production)
        
        # Energy consumption
        energy = self.calculate_energy_consumption(spm, stroke_length, rod_load, oil_viscosity)
        
        # IntegratedDigitalTwin computes SOR after CSS steam usage is known.
        sor = 0.0
        
        # Failure risk
        failure_risk = self.calculate_failure_risk(rod_load, spm, oil_viscosity)
        
        # Motor current (approximation)
        motor_current = (energy / 11) * 150  # 11kV supply
        
        # Anomaly detection
        is_anomalous, anomaly_type = self.detect_anomaly(rod_load, oil_viscosity, spm, efficiency)
        
        return {
            'actual_production': actual_production,
            'pump_efficiency': efficiency,
            'energy_consumption': energy,
            'sor': sor,
            'rod_load': rod_load,
            'failure_risk': failure_risk,
            'motor_current': motor_current,
            'is_anomalous': is_anomalous,
            'anomaly_type': anomaly_type,
            'spm': spm,
            'stroke_length': stroke_length,
            'vfd_frequency': vfd_frequency
        }
    
    def get_state(self) -> Dict:
        return {
            'design_rod_load': self.design_rod_load,
            'design_power': self.design_power
        }


class IntegratedDigitalTwin:
    """
    Well-to-Surface Digital Twin integrating all three domains
    This is the core simulation engine
    """
    
    def __init__(self, well_id: str = "BW-001"):
        self.well_id = well_id
        self.reservoir_twin = ReservoirTwin()
        self.wellbore_twin = WellboreTwin()
        self.srp_twin = SRPTwin()
        self.simulation_history = []
    
    def simulate_css_cycle(self, 
                          steam_volume: float,
                          steam_pressure: float,
                          injection_duration: float,
                          soak_time: float,
                          spm: float,
                          stroke_length: float,
                          vfd_frequency: float,
                          production_cutoff: float) -> WellState:
        """
        Simulate complete CSS cycle with integrated well-to-surface model
        
        Flow:
        1. Steam injection heats reservoir
        2. Temperature changes affect viscosity
        3. Viscosity affects wellbore flow
        4. Wellbore conditions affect SRP performance
        """
        
        # Step 1: Reservoir response to steam
        reservoir_response = self.reservoir_twin.simulate_steam_injection(
            steam_volume, steam_pressure, injection_duration
        )

        # Soak time controls heat retention; production cutoff is represented as
        # a conservative production operating factor for this prototype.
        soak_retention = 1.0 - 0.08 * np.exp(-soak_time / 24.0)
        self.reservoir_twin.current_temp = (
            self.reservoir_twin.initial_temp
            + (self.reservoir_twin.current_temp - self.reservoir_twin.initial_temp) * soak_retention
        )
        self.reservoir_twin.current_viscosity = 1100 * np.exp(-0.015 * self.reservoir_twin.current_temp)
        
        # Step 2: Get current reservoir state
        reservoir_state = self.reservoir_twin.get_state()
        
        # Step 3: Simulate wellbore flow with current conditions
        wellbore_response = self.wellbore_twin.simulate_flow(
            oil_viscosity=reservoir_state['viscosity'],
            reservoir_pressure=20.0,  # Typical for Baghewala
            oil_mobility=reservoir_state['mobility']
        )
        
        # Step 4: Simulate SRP with current wellbore conditions
        srp_response = self.srp_twin.simulate_operation(
            spm=spm,
            stroke_length=stroke_length,
            vfd_frequency=vfd_frequency,
            oil_viscosity=reservoir_state['viscosity'],
            pump_inlet_pressure=wellbore_response['pump_inlet_pressure']
        )
        srp_response['actual_production'] *= np.clip(0.97 + 0.03 * production_cutoff, 0.95, 1.03)
        srp_response['sor'] = steam_volume / max(srp_response['actual_production'] * 0.5, 1.0)
        
        # Create unified well state
        well_state = WellState(
            reservoir_temperature=reservoir_state['temperature'],
            reservoir_pressure=20.0,
            oil_viscosity=reservoir_state['viscosity'],
            oil_mobility=reservoir_state['mobility'],
            wellbore_temp=wellbore_response['wellbore_temperature'],
            wellbore_pressure_drop=wellbore_response['pressure_drop'],
            pump_inlet_pressure=wellbore_response['pump_inlet_pressure'],
            current_production=srp_response['actual_production'],
            current_efficiency=srp_response['pump_efficiency'],
            current_energy=srp_response['energy_consumption'],
            current_sor=srp_response['sor'],
            rod_load=srp_response['rod_load'],
            failure_risk=srp_response['failure_risk'],
            is_anomalous=srp_response['is_anomalous'],
            anomaly_type=srp_response['anomaly_type']
        )

        # Advance the reservoir to the next cycle after the current response.
        self.reservoir_twin.simulate_cooling(max(1.0, soak_time / 24.0))
        
        # Store in history
        self.simulation_history.append(well_state)
        
        return well_state
    
    def get_simulation_summary(self) -> Dict:
        """Return summary of current simulation"""
        if not self.simulation_history:
            return {}
        
        latest = self.simulation_history[-1]
        return {
            'well_id': self.well_id,
            'reservoir_temperature': round(latest.reservoir_temperature, 2),
            'oil_viscosity': round(latest.oil_viscosity, 2),
            'actual_production': round(latest.current_production, 2),
            'pump_efficiency': round(latest.current_efficiency, 3),
            'energy_consumption': round(latest.current_energy, 2),
            'sor': round(latest.current_sor, 2),
            'rod_load': round(latest.rod_load, 2),
            'failure_risk': round(latest.failure_risk, 3),
            'is_anomalous': latest.is_anomalous,
            'anomaly_type': latest.anomaly_type
        }


if __name__ == "__main__":
    # Test the digital twin
    twin = IntegratedDigitalTwin("BW-001")
    
    # Simulate a CSS cycle
    state = twin.simulate_css_cycle(
        steam_volume=80,
        steam_pressure=25,
        injection_duration=24,
        soak_time=36,
        spm=4.0,
        stroke_length=86,
        vfd_frequency=45,
        production_cutoff=1.0
    )
    
    print("="*80)
    print("DIGITAL TWIN SIMULATION RESULTS")
    print("="*80)
    summary = twin.get_simulation_summary()
    for key, value in summary.items():
        print(f"{key:.<40} {value}")
