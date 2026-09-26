"""
FastAPI Backend for SIH26120 Digital Twin
Provides REST API for well simulation, optimization, and dashboarding
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional
import pandas as pd
import os

from digital_twin import IntegratedDigitalTwin
from ai_models import ModelManager
from optimizer import SimplifiedOptimizer, OptimizationConstraints, OptimizationReport
from data_generator import BaghewalaWellDataGenerator

# Initialize FastAPI app
app = FastAPI(
    title="SIH26120 Digital Twin API",
    description="Well-to-Surface Digital Twin for CSS and SRP Optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
digital_twin = IntegratedDigitalTwin()
model_manager = ModelManager()
optimizer = SimplifiedOptimizer()
wells_data = {}


# Pydantic models for request/response
class SimulationRequest(BaseModel):
    steam_volume: float = Field(ge=60, le=100)
    steam_pressure: float = Field(ge=22, le=28)
    injection_duration: float = Field(ge=18, le=30)
    soak_time: float = Field(ge=24, le=48)
    spm: float = Field(ge=3, le=6.5)
    stroke_length: float = Field(ge=70, le=105)
    vfd_frequency: float = Field(ge=35, le=55)
    production_cutoff: float = Field(default=1.0, ge=0.5, le=2.0)


class OptimizationRequest(BaseModel):
    n_iterations: int = Field(default=50, ge=10, le=768)
    priority: Literal["balanced", "production", "efficiency", "cost"] = "balanced"


class FeedbackRequest(SimulationRequest):
    observed_production: float = Field(ge=0)
    observed_energy: float = Field(ge=0)
    observed_failure_risk: float = Field(ge=0, le=1)
    observed_temperature: Optional[float] = Field(default=None, ge=-20, le=200)


class WellStateResponse(BaseModel):
    well_id: str
    reservoir_temperature: float
    oil_viscosity: float
    actual_production: float
    pump_efficiency: float
    energy_consumption: float
    sor: float
    rod_load: float
    failure_risk: float
    is_anomalous: int
    anomaly_type: Optional[str] = None


class PredictionResponse(BaseModel):
    production: float
    temperature: float
    energy: float
    failure_risk: tuple


class RecommendationResponse(BaseModel):
    css_parameters: Dict
    srp_parameters: Dict
    expected_outcomes: Dict
    fitness_score: float
    priority: str


def simulate_snapshot(parameters: Dict) -> object:
    """Evaluate one independent cycle from the configured initial well state."""
    scenario = IntegratedDigitalTwin(digital_twin.well_id)
    return scenario.simulate_css_cycle(**parameters)


def ensure_models_loaded() -> None:
    """Load persisted models on demand after a reload or direct API import."""
    if model_manager.production_predictor.model is not None:
        return
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(os.path.join(model_dir, 'production_model.pkl')):
        raise HTTPException(status_code=503, detail="Models are not trained yet")
    try:
        model_manager.load_all(model_dir)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Models could not be loaded: {exc}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the interactive dashboard."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "dashboard.html"))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "digital_twin": "ready"}


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@app.post("/api/simulate")
async def simulate_well(request: SimulationRequest):
    """
    Simulate a CSS cycle with given parameters
    Returns predicted well state
    """
    try:
        parameters = request.model_dump()
        well_state = simulate_snapshot(parameters)
        digital_twin.simulation_history.append(well_state)
        
        return {
            "success": True,
            "simulation": {
                "reservoir_temperature": round(well_state.reservoir_temperature, 2),
                "oil_viscosity": round(well_state.oil_viscosity, 2),
                "actual_production": round(well_state.current_production, 2),
                "pump_efficiency": round(well_state.current_efficiency, 3),
                "energy_consumption": round(well_state.current_energy, 2),
                "sor": round(well_state.current_sor, 2),
                "rod_load": round(well_state.rod_load, 2),
                "failure_risk": round(well_state.failure_risk, 3),
                "is_anomalous": well_state.is_anomalous,
                "anomaly_type": well_state.anomaly_type
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/simulate/what-if")
async def what_if_simulator(
    steam_volume: float = Query(default=80, ge=60, le=100),
    steam_pressure: float = Query(default=25, ge=22, le=28),
    injection_duration: float = Query(default=24, ge=18, le=30),
    soak_time: float = Query(default=36, ge=24, le=48),
    spm: float = Query(default=4.0, ge=3, le=6.5),
    stroke_length: float = Query(default=86, ge=70, le=105),
    vfd_frequency: float = Query(default=45, ge=35, le=55),
    production_cutoff: float = Query(default=1.0, ge=0.5, le=2.0)
):
    """
    What-If Simulator: Evaluate multiple scenarios
    Returns comparison of current vs proposed parameters
    """
    try:
        # Current scenario (baseline)
        current = simulate_snapshot({
            'steam_volume': 80, 'steam_pressure': 25,
            'injection_duration': 24, 'soak_time': 36,
            'spm': 4.0, 'stroke_length': 86, 'vfd_frequency': 45,
            'production_cutoff': 1.0
        })
        
        # Proposed scenario
        proposed = simulate_snapshot({
            'steam_volume': steam_volume,
            'steam_pressure': steam_pressure,
            'injection_duration': injection_duration,
            'soak_time': soak_time,
            'spm': spm,
            'stroke_length': stroke_length,
            'vfd_frequency': vfd_frequency,
            'production_cutoff': production_cutoff
        })
        
        return {
            "success": True,
            "comparison": {
                "current": {
                    "production": round(current.current_production, 2),
                    "efficiency": round(current.current_efficiency, 3),
                    "energy": round(current.current_energy, 2),
                    "sor": round(current.current_sor, 2)
                },
                "proposed": {
                    "production": round(proposed.current_production, 2),
                    "efficiency": round(proposed.current_efficiency, 3),
                    "energy": round(proposed.current_energy, 2),
                    "sor": round(proposed.current_sor, 2)
                },
                "changes": {
                    "production_change_percent": round(
                        ((proposed.current_production - current.current_production) / current.current_production) * 100
                        if current.current_production > 0 else 0, 2),
                    "energy_change_percent": round(
                        ((proposed.current_energy - current.current_energy) / current.current_energy) * 100, 2)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/api/predict")
async def predict(request: SimulationRequest):
    """
    Generate AI predictions for given parameters
    """
    try:
        ensure_models_loaded()
        twin_state = simulate_snapshot(request.model_dump())
        input_dict = {
            'steam_volume': request.steam_volume,
            'steam_pressure': request.steam_pressure,
            'injection_duration': request.injection_duration,
            'soak_time': request.soak_time,
            'spm': request.spm,
            'stroke_length': request.stroke_length,
            'vfd_frequency': request.vfd_frequency,
            'rod_load': twin_state.rod_load,
            'oil_viscosity': twin_state.oil_viscosity,
            'reservoir_temperature': twin_state.reservoir_temperature,
            'motor_current': (twin_state.current_energy / 11) * 150,
            'previous_temperature': 46.0
        }
        
        predictions = model_manager.predict_all(input_dict)
        
        return {
            "success": True,
            "predictions": {
                "production_bopd": round(predictions['production'], 2),
                "temperature_celsius": round(predictions['temperature'], 2),
                "energy_consumption_kw": round(predictions['energy'], 2),
                "failure_risk_level": predictions['failure_risk'][0],
                "failure_risk_probability": round(predictions['failure_risk'][1], 3),
                "twin_reference": {
                    "production_bopd": round(twin_state.current_production, 2),
                    "temperature_celsius": round(twin_state.reservoir_temperature, 2),
                    "energy_consumption_kw": round(twin_state.current_energy, 2),
                    "failure_risk": round(twin_state.failure_risk, 3)
                },
                "agreement": {
                    "production_error_percent": round((predictions['production'] - twin_state.current_production) / max(twin_state.current_production, 1) * 100, 2),
                    "energy_error_percent": round((predictions['energy'] - twin_state.current_energy) / max(twin_state.current_energy, 1) * 100, 2),
                    "warning": bool(abs(predictions['production'] - twin_state.current_production) / max(twin_state.current_production, 1) > 0.2)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/feedback")
async def record_feedback(request: FeedbackRequest):
    """Record an observed cycle outcome for audit and future recalibration."""
    try:
        ensure_models_loaded()
        parameters = request.model_dump(exclude={
            'observed_production', 'observed_energy',
            'observed_failure_risk', 'observed_temperature'
        })
        twin_state = simulate_snapshot(parameters)
        input_dict = {
            **parameters,
            'rod_load': twin_state.rod_load,
            'oil_viscosity': twin_state.oil_viscosity,
            'reservoir_temperature': twin_state.reservoir_temperature,
            'motor_current': (twin_state.current_energy / 11) * 150,
            'previous_temperature': 46.0
        }
        prediction = model_manager.predict_all(input_dict)
        observed_temperature = request.observed_temperature
        feedback_row = {
            **parameters,
            'predicted_production': prediction['production'],
            'observed_production': request.observed_production,
            'predicted_energy': prediction['energy'],
            'observed_energy': request.observed_energy,
            'predicted_failure_risk': prediction['failure_risk'][1],
            'observed_failure_risk': request.observed_failure_risk,
            'predicted_temperature': prediction['temperature'],
            'observed_temperature': observed_temperature,
            'production_error': request.observed_production - prediction['production'],
            'energy_error': request.observed_energy - prediction['energy'],
            'risk_error': request.observed_failure_risk - prediction['failure_risk'][1]
        }
        feedback_path = os.path.join(os.path.dirname(__file__), 'data', 'feedback_log.csv')
        feedback_df = pd.DataFrame([feedback_row])
        if os.path.exists(feedback_path):
            feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_path, index=False)
        return {
            'success': True,
            'feedback_file': feedback_path,
            'records_added': 1,
            'errors': {
                'production': round(feedback_row['production_error'], 3),
                'energy': round(feedback_row['energy_error'], 3),
                'failure_risk': round(feedback_row['risk_error'], 3)
            },
            'next_step': 'Review feedback_log.csv before merging validated field observations into training data.'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/models/retrain-with-feedback")
async def retrain_with_feedback():
    """Retrain using reviewed feedback rows plus the synthetic baseline dataset."""
    try:
        ensure_models_loaded()
        base_path = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_data.csv')
        feedback_path = os.path.join(os.path.dirname(__file__), 'data', 'feedback_log.csv')
        if not os.path.exists(feedback_path):
            raise HTTPException(status_code=404, detail="No feedback records available")

        training_df = pd.read_csv(base_path)
        feedback_df = pd.read_csv(feedback_path)
        template = training_df.iloc[0].copy()
        reviewed_rows = []
        for _, feedback in feedback_df.iterrows():
            parameters = {field: float(feedback[field]) for field in SimulationRequest.model_fields if field in feedback}
            state = simulate_snapshot(parameters)
            row = template.copy()
            row.update(parameters)
            row.update({
                'reservoir_temperature': feedback.get('observed_temperature', state.reservoir_temperature),
                'reservoir_pressure': state.reservoir_pressure,
                'oil_viscosity': state.oil_viscosity,
                'oil_mobility': state.oil_mobility,
                'wellbore_temperature': state.wellbore_temp,
                'wellbore_pressure_drop': state.wellbore_pressure_drop,
                'pump_inlet_pressure': state.pump_inlet_pressure,
                'actual_production': feedback.observed_production,
                'pump_efficiency': state.current_efficiency,
                'energy_consumption': feedback.observed_energy,
                'sor': parameters['steam_volume'] / max(feedback.observed_production * 0.5, 1.0),
                'rod_load_actual': state.rod_load,
                'rod_load': state.rod_load,
                'failure_risk': feedback.observed_failure_risk,
                'motor_current': (feedback.observed_energy / 11) * 150,
                'well_id': 'FEEDBACK',
                'cycle_id': int(len(training_df) + len(reviewed_rows) + 1)
            })
            reviewed_rows.append(row)
        combined = pd.concat([training_df, pd.DataFrame(reviewed_rows)], ignore_index=True)
        results = model_manager.train_all(combined)
        model_manager.save_all()
        return {
            'success': True,
            'training_rows': len(combined),
            'feedback_rows_used': len(reviewed_rows),
            'results': results,
            'warning': 'Only reviewed observations should be retained in feedback_log.csv.'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# OPTIMIZATION ENDPOINTS
# ============================================================================

@app.post("/api/optimize")
async def optimize(request: OptimizationRequest):
    """
    Run multi-objective optimization
    Returns top solutions and recommendations
    """
    try:
        # Run optimization
        solutions = optimizer.optimize(n_iterations=request.n_iterations)
        
        # Get recommendation
        recommendation = optimizer.get_recommendation(priority=request.priority)
        
        # Get top 5 solutions
        top_solutions = optimizer.get_top_solutions(5)
        
        return {
            "success": True,
            "optimization": {
                "total_solutions_evaluated": len(solutions),
                "valid_solutions": len([s for s in solutions if s['valid']]),
                "recommendation": recommendation,
                "top_5_solutions": [
                    {
                        "rank": i + 1,
                        "fitness_score": round(s['fitness_score'], 2),
                        "css": {
                            "steam_volume_tons": round(s['parameters']['steam_volume'], 2),
                            "soak_time_hours": round(s['parameters']['soak_time'], 2)
                        },
                        "srp": {
                            "spm": round(s['parameters']['spm'], 2),
                            "stroke_inches": round(s['parameters']['stroke_length'], 2),
                            "vfd_hz": round(s['parameters']['vfd_frequency'], 2)
                        },
                        "expected_outcomes": {
                            "production": round(s['objectives']['production'], 2),
                            "sor": round(s['objectives']['sor'], 2),
                            "energy": round(s['objectives']['energy'], 2)
                        }
                    }
                    for i, s in enumerate(top_solutions)
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/optimize/report")
async def optimization_report():
    """Get optimization report"""
    try:
        report = OptimizationReport.generate_report(optimizer)
        return {
            "success": True,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# DATA ENDPOINTS
# ============================================================================

@app.get("/api/data/generate")
async def generate_synthetic_data(n_cycles: int = 100, n_wells: int = 5):
    """Generate fresh synthetic dataset"""
    try:
        generator = BaghewalaWellDataGenerator()
        df = generator.generate_complete_dataset(n_cycles=n_cycles, n_wells=n_wells)
        filepath = generator.save_dataset(df, f'synthetic_{n_wells}wells_{n_cycles}cycles.csv')
        
        return {
            "success": True,
            "data_generation": {
                "filepath": filepath,
                "rows_generated": len(df),
                "wells": n_wells,
                "cycles_per_well": n_cycles,
                "features": list(df.columns)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/data/statistics")
async def data_statistics(filename: str = 'synthetic_data.csv'):
    """Get statistics of current dataset"""
    try:
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = pd.read_csv(filepath)
        
        return {
            "success": True,
            "statistics": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "wells": df['well_id'].nunique() if 'well_id' in df.columns else 0,
                "numeric_features": df.select_dtypes(include=['number']).shape[1],
                "summary": df.describe().to_dict()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# MODEL ENDPOINTS
# ============================================================================

@app.post("/api/models/train")
async def train_models(filename: str = 'synthetic_data.csv'):
    """Train all AI models on dataset"""
    try:
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(filepath)
        results = model_manager.train_all(df)
        model_manager.save_all()
        
        return {
            "success": True,
            "training": {
                "models_trained": list(results.keys()),
                "results": results,
                "models_saved": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models/status")
async def model_status():
    """Check if models are loaded"""
    return {
        "success": True,
        "models": {
            "production_predictor": model_manager.production_predictor.model is not None,
            "temperature_predictor": model_manager.temperature_predictor.model is not None,
            "energy_predictor": model_manager.energy_predictor.model is not None,
            "failure_risk_classifier": model_manager.failure_risk_classifier.model is not None
        }
    }


# ============================================================================
# DASHBOARD DATA ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/well-summary")
async def well_summary():
    """Get current well state summary for dashboard"""
    try:
        summary = digital_twin.get_simulation_summary()
        
        return {
            "success": True,
            "well_state": summary
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dashboard/metrics")
async def dashboard_metrics():
    """Get key metrics for dashboard display"""
    try:
        summary = digital_twin.get_simulation_summary()
        
        return {
            "success": True,
            "metrics": {
                "production": {
                    "value": summary.get('actual_production', 0),
                    "unit": "BOPD",
                    "status": "normal"
                },
                "efficiency": {
                    "value": summary.get('pump_efficiency', 0),
                    "unit": "fraction",
                    "status": "normal"
                },
                "energy": {
                    "value": summary.get('energy_consumption', 0),
                    "unit": "kW",
                    "status": "normal"
                },
                "sor": {
                    "value": summary.get('sor', 0),
                    "unit": "SOR",
                    "status": "normal"
                },
                "failure_risk": {
                    "value": summary.get('failure_risk', 0),
                    "unit": "Risk %",
                    "status": "high" if summary.get('failure_risk', 0) > 0.6 else "normal"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("Starting Digital Twin API...")
    
    # Check if synthetic data exists
    if not os.path.exists('data/synthetic_data.csv'):
        print("Generating synthetic data...")
        generator = BaghewalaWellDataGenerator()
        df = generator.generate_complete_dataset(n_cycles=50, n_wells=3)
        generator.save_dataset(df)
    
    # Try to load pre-trained models
    try:
        if os.path.exists('models/production_model.pkl'):
            model_manager.load_all('models')
            print("Pre-trained models loaded successfully")
        else:
            print("No pre-trained models found. Train models using /api/models/train endpoint")
    except Exception as e:
        print(f"Could not load models: {e}")
    
    print("Digital Twin API is ready!")


if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("SIH26120 DIGITAL TWIN - FastAPI Server")
    print("="*80)
    print("\nStarting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
