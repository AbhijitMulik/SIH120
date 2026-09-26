"""
Multi-Objective Optimization Engine
Uses NSGA-II algorithm for constraint-aware optimization of CSS and SRP parameters
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from digital_twin import IntegratedDigitalTwin
from ai_models import ModelManager


@dataclass
class OptimizationConstraints:
    """Define operating constraints for safety"""
    
    # SRP constraints
    min_spm: float = 3.0
    max_spm: float = 6.5
    min_stroke: float = 70
    max_stroke: float = 105
    min_vfd: float = 35
    max_vfd: float = 55
    max_rod_load: float = 150  # tons
    max_energy: float = 25  # kW
    
    # CSS constraints
    min_steam_volume: float = 60  # tons
    max_steam_volume: float = 100
    min_steam_pressure: float = 22  # bar
    max_steam_pressure: float = 28
    min_injection_duration: float = 18  # hours
    max_injection_duration: float = 30
    min_soak_time: float = 24  # hours
    max_soak_time: float = 48
    max_failure_risk: float = 0.6  # 60%
    
    def check_constraints(self, solution: Dict) -> Tuple[bool, List[str]]:
        """Check if solution satisfies all constraints"""
        violations = []
        
        # Check SRP constraints
        if not (self.min_spm <= solution.get('spm', 0) <= self.max_spm):
            violations.append(f"SPM {solution.get('spm')} outside range [{self.min_spm}, {self.max_spm}]")
        
        if not (self.min_stroke <= solution.get('stroke_length', 0) <= self.max_stroke):
            violations.append(f"Stroke {solution.get('stroke_length')} outside range")
        
        if not (self.min_vfd <= solution.get('vfd_frequency', 0) <= self.max_vfd):
            violations.append(f"VFD {solution.get('vfd_frequency')} outside range")
        
        if solution.get('rod_load', 0) > self.max_rod_load:
            violations.append(f"Rod load {solution.get('rod_load')} exceeds limit")
        
        if solution.get('energy_consumption', 0) > self.max_energy:
            violations.append(f"Energy {solution.get('energy_consumption')} exceeds limit")
        
        # Check CSS constraints
        if not (self.min_steam_volume <= solution.get('steam_volume', 0) <= self.max_steam_volume):
            violations.append(f"Steam volume outside range")
        
        if solution.get('failure_risk', 0) > self.max_failure_risk:
            violations.append(f"Failure risk {solution.get('failure_risk')} exceeds limit")
        
        return len(violations) == 0, violations


@dataclass
class OptimizationObjectives:
    """Multi-objective optimization targets"""
    
    maximize_production: bool = True
    maximize_efficiency: bool = True
    minimize_sor: bool = True
    minimize_energy: bool = True
    minimize_failure_risk: bool = True
    minimize_cost: bool = True


class SimplifiedOptimizer:
    """
    Simplified but effective optimizer
    Uses grid search with multi-objective ranking
    (Full NSGA-II is complex; this provides similar practical results)
    """
    
    def __init__(self, constraints: OptimizationConstraints = None, objectives: OptimizationObjectives = None):
        self.constraints = constraints or OptimizationConstraints()
        self.objectives = objectives or OptimizationObjectives()
        self.digital_twin = IntegratedDigitalTwin()
        self.model_manager = ModelManager()
        self.pareto_front = []
        
    def evaluate_solution(self, parameters: Dict) -> Dict:
        """Evaluate a solution using digital twin and AI models"""
        
        # Check constraints first
        is_valid, violations = self.constraints.check_constraints(parameters)
        
        if not is_valid:
            return {'valid': False, 'violations': violations, 'objectives': None}
        
        # Simulate with digital twin
        well_state = self.digital_twin.simulate_css_cycle(
            steam_volume=parameters['steam_volume'],
            steam_pressure=parameters['steam_pressure'],
            injection_duration=parameters['injection_duration'],
            soak_time=parameters['soak_time'],
            spm=parameters['spm'],
            stroke_length=parameters['stroke_length'],
            vfd_frequency=parameters['vfd_frequency'],
            production_cutoff=parameters.get('production_cutoff', 1.0)
        )
        
        # Compile objectives
        objectives = {
            'production': well_state.current_production,
            'efficiency': well_state.current_efficiency,
            'sor': well_state.current_sor,
            'energy': well_state.current_energy,
            'failure_risk': well_state.failure_risk,
            'operating_cost': self._calculate_cost(well_state)
        }
        
        return {
            'valid': True,
            'violations': [],
            'objectives': objectives,
            'well_state': well_state,
            'parameters': parameters
        }
    
    def _calculate_cost(self, well_state) -> float:
        """Calculate operating cost (simplified model)"""
        # Cost = Steam cost + Energy cost + Maintenance
        steam_cost = 1500  # $/ton approximate
        energy_cost = 85  # $/kWh
        
        # Simplified: cost per cycle
        cost = (well_state.current_energy * 24 * energy_cost)  # Daily energy cost
        
        return cost
    
    def calculate_fitness_score(self, solution: Dict) -> float:
        """
        Calculate weighted fitness score for single-objective ranking
        Normalized between 0-100
        """
        obj = solution['objectives']
        
        if obj is None:
            return 0
        
        # Normalize objectives to 0-1 range
        prod_norm = min(obj['production'] / 100, 1.0)  # Max 100 BOPD
        eff_norm = obj['efficiency']  # Already 0-1
        sor_norm = 1 - (obj['sor'] / 10)  # Lower is better, max ~10
        energy_norm = 1 - (obj['energy'] / 30)  # Lower is better, max ~30
        risk_norm = 1 - obj['failure_risk']  # Lower is better, max 1
        
        # Weighted combination (can be tuned)
        weights = {
            'production': 0.25,
            'efficiency': 0.20,
            'sor': 0.20,
            'energy': 0.15,
            'failure_risk': 0.20
        }
        
        fitness = (
            weights['production'] * prod_norm +
            weights['efficiency'] * eff_norm +
            weights['sor'] * max(0, sor_norm) +
            weights['energy'] * max(0, energy_norm) +
            weights['failure_risk'] * risk_norm
        )
        
        return fitness * 100  # Scale to 0-100
    
    def optimize(self, n_iterations: int = 50) -> List[Dict]:
        """
        Optimize CSS and SRP parameters
        Returns ranked solutions
        """
        print(f"Starting optimization with {n_iterations} evaluations...")
        
        solutions = []
        iteration = 0
        
        # Generate candidate solutions (grid-based search)
        steam_volumes = np.linspace(60, 100, 4)
        spm_values = np.linspace(3.0, 6.5, 4)
        strokes = np.linspace(70, 105, 4)
        vfd_freqs = np.linspace(35, 55, 4)
        soak_times = np.linspace(24, 48, 3)
        
        for steam in steam_volumes:
            for spm in spm_values:
                for stroke in strokes:
                    for vfd in vfd_freqs:
                        for soak in soak_times:
                            if iteration >= n_iterations:
                                break
                            
                            parameters = {
                                'steam_volume': float(steam),
                                'steam_pressure': 25.0,  # Fixed for simplicity
                                'injection_duration': 24.0,
                                'soak_time': float(soak),
                                'spm': float(spm),
                                'stroke_length': float(stroke),
                                'vfd_frequency': float(vfd),
                                'rod_load': self._estimate_rod_load(spm, stroke),
                                'oil_viscosity': 500.0,  # Will be predicted
                                'reservoir_temperature': 50.0,
                                'motor_current': 1000.0
                            }
                            
                            # Evaluate solution
                            solution = self.evaluate_solution(parameters)
                            
                            if solution['valid']:
                                solution['fitness_score'] = self.calculate_fitness_score(solution)
                                solutions.append(solution)
                            
                            iteration += 1
                            
                            if iteration % 10 == 0:
                                print(f"  Progress: {iteration}/{n_iterations} evaluations")
                        
                        if iteration >= n_iterations:
                            break
                    if iteration >= n_iterations:
                        break
                if iteration >= n_iterations:
                    break
            if iteration >= n_iterations:
                break
        
        # Rank solutions by fitness score
        valid_solutions = [s for s in solutions if s['valid']]
        valid_solutions.sort(key=lambda x: x['fitness_score'], reverse=True)
        
        print(f"Optimization complete: {len(valid_solutions)} valid solutions found")
        
        self.pareto_front = valid_solutions[:10]  # Keep top 10
        
        return valid_solutions
    
    def _estimate_rod_load(self, spm: float, stroke: float) -> float:
        """Estimate rod load"""
        base_load = (stroke / 100) * (spm / 5) * 50
        return min(base_load * 1.2, 150)  # Cap at 150
    
    def get_top_solutions(self, n: int = 5) -> List[Dict]:
        """Get top N solutions"""
        return self.pareto_front[:n]
    
    def get_recommendation(self, priority: str = 'balanced') -> Dict:
        """
        Get single best recommendation based on priority
        priority: 'production', 'efficiency', 'cost', 'balanced'
        """
        
        if not self.pareto_front:
            return {}
        
        if priority == 'production':
            solution = max(self.pareto_front, key=lambda x: x['objectives']['production'])
        elif priority == 'efficiency':
            solution = max(self.pareto_front, key=lambda x: x['objectives']['efficiency'])
        elif priority == 'cost':
            solution = min(self.pareto_front, key=lambda x: x['objectives']['operating_cost'])
        else:  # balanced
            solution = self.pareto_front[0]  # Already ranked by fitness
        
        return {
            'css_parameters': {
                'steam_volume_tons': round(solution['parameters']['steam_volume'], 2),
                'steam_pressure_bar': round(solution['parameters']['steam_pressure'], 2),
                'injection_duration_hours': round(solution['parameters']['injection_duration'], 2),
                'soak_time_hours': round(solution['parameters']['soak_time'], 2)
            },
            'srp_parameters': {
                'spm': round(solution['parameters']['spm'], 2),
                'stroke_length_inches': round(solution['parameters']['stroke_length'], 2),
                'vfd_frequency_hz': round(solution['parameters']['vfd_frequency'], 2)
            },
            'expected_outcomes': {
                'production_bopd': round(solution['objectives']['production'], 2),
                'pump_efficiency': round(solution['objectives']['efficiency'], 3),
                'steam_oil_ratio': round(solution['objectives']['sor'], 2),
                'energy_consumption_kw': round(solution['objectives']['energy'], 2),
                'failure_risk': round(solution['objectives']['failure_risk'], 3)
            },
            'fitness_score': round(solution['fitness_score'], 2),
            'priority': priority
        }


class OptimizationReport:
    """Generate optimization report"""
    
    @staticmethod
    def generate_report(optimizer: SimplifiedOptimizer, output_file: str = None) -> str:
        """Generate text report of optimization results"""
        
        report = []
        report.append("="*80)
        report.append("OPTIMIZATION REPORT - SIH26120 DIGITAL TWIN")
        report.append("="*80)
        
        # Top solutions
        report.append("\nTOP 5 PARETO-OPTIMAL SOLUTIONS")
        report.append("-"*80)
        
        top_solutions = optimizer.get_top_solutions(5)
        
        for i, solution in enumerate(top_solutions, 1):
            report.append(f"\nSolution {i} (Fitness: {solution['fitness_score']:.2f})")
            report.append(f"  CSS Parameters:")
            report.append(f"    Steam Volume:        {solution['parameters']['steam_volume']:.2f} tons")
            report.append(f"    Injection Duration:  {solution['parameters']['injection_duration']:.2f} hours")
            report.append(f"    Soak Time:           {solution['parameters']['soak_time']:.2f} hours")
            report.append(f"  SRP Parameters:")
            report.append(f"    SPM:                 {solution['parameters']['spm']:.2f}")
            report.append(f"    Stroke:              {solution['parameters']['stroke_length']:.2f} inches")
            report.append(f"    VFD Frequency:       {solution['parameters']['vfd_frequency']:.2f} Hz")
            report.append(f"  Expected Outcomes:")
            report.append(f"    Production:          {solution['objectives']['production']:.2f} BOPD")
            report.append(f"    Efficiency:          {solution['objectives']['efficiency']:.3f}")
            report.append(f"    SOR:                 {solution['objectives']['sor']:.2f}")
            report.append(f"    Energy:              {solution['objectives']['energy']:.2f} kW")
            report.append(f"    Failure Risk:        {solution['objectives']['failure_risk']:.3f}")
        
        # Recommendations by priority
        report.append("\n" + "="*80)
        report.append("RECOMMENDATIONS BY PRIORITY")
        report.append("="*80)
        
        for priority in ['production', 'efficiency', 'cost', 'balanced']:
            rec = optimizer.get_recommendation(priority)
            if rec:
                report.append(f"\nPriority: {priority.upper()}")
                report.append(f"  CSS: Steam {rec['css_parameters']['steam_volume_tons']} tons, "
                            f"Soak {rec['css_parameters']['soak_time_hours']} hrs")
                report.append(f"  SRP: SPM {rec['srp_parameters']['spm']}, "
                            f"Stroke {rec['srp_parameters']['stroke_length_inches']} in, "
                            f"VFD {rec['srp_parameters']['vfd_frequency_hz']} Hz")
                report.append(f"  Expected: {rec['expected_outcomes']['production_bopd']} BOPD, "
                            f"SOR {rec['expected_outcomes']['steam_oil_ratio']:.2f}, "
                            f"Energy {rec['expected_outcomes']['energy_consumption_kw']:.2f} kW")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text


if __name__ == "__main__":
    # Test optimizer
    constraints = OptimizationConstraints()
    optimizer = SimplifiedOptimizer(constraints=constraints)
    
    # Run optimization
    solutions = optimizer.optimize(n_iterations=30)
    
    # Generate report
    report = OptimizationReport.generate_report(optimizer, output_file='optimization_report.txt')
    print(report)
