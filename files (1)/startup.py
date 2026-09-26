#!/usr/bin/env python3
"""
SIH26120 Digital Twin - Startup Script
Initializes and starts the entire system with one command
"""

import os
import sys
import subprocess
import time
from pathlib import Path


class StartupManager:
    """Manages startup sequence"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'models'
        self.logs_dir = self.base_dir / 'logs'
    
    def print_banner(self):
        """Print ASCII banner"""
        banner = """
        ╔════════════════════════════════════════════════════════════════╗
        ║                                                                ║
        ║        🔧  SIH26120 DIGITAL TWIN - STARTUP SEQUENCE  🔧       ║
        ║                                                                ║
        ║  Well-to-Surface Optimization for CSS & SRP Operations        ║
        ║  Oil India Limited - Smart Automation                         ║
        ║                                                                ║
        ╚════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def create_directories(self):
        """Create necessary directories"""
        print("[1/6] Creating directories...")
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
            print(f"  ✓ {directory}")
    
    def check_dependencies(self):
        """Check if all dependencies are installed"""
        print("\n[2/6] Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'xgboost', 'fastapi', 'uvicorn'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                missing.append(package)
                print(f"  ✗ {package}")
        
        if missing:
            print(f"\n❌ Missing packages: {', '.join(missing)}")
            print("Run: pip install -r requirements.txt")
            return False
        
        print("  ✓ All dependencies installed")
        return True
    
    def generate_synthetic_data(self):
        """Generate synthetic data if it doesn't exist"""
        print("\n[3/6] Checking synthetic data...")
        
        synthetic_file = self.data_dir / 'synthetic_data.csv'
        
        if synthetic_file.exists():
            print(f"  ✓ Using existing: {synthetic_file}")
            return True
        
        print("  ! Synthetic data not found. Generating...")
        try:
            from data_generator import BaghewalaWellDataGenerator
            
            generator = BaghewalaWellDataGenerator(seed=42)
            print("    Generating 5 wells × 100 cycles...")
            df = generator.generate_complete_dataset(n_cycles=100, n_wells=5)
            
            generator.save_dataset(df)
            print(f"  ✓ Generated: {len(df)} rows, {len(df.columns)} columns")
            return True
        
        except Exception as e:
            print(f"  ✗ Error generating data: {e}")
            return False
    
    def train_models(self):
        """Train AI models if they don't exist"""
        print("\n[4/6] Checking AI models...")
        
        model_file = self.models_dir / 'production_model.pkl'
        
        if model_file.exists():
            print(f"  ✓ Using existing models")
            return True
        
        print("  ! Pre-trained models not found. Training...")
        try:
            import pandas as pd
            from ai_models import ModelManager
            
            # Load synthetic data
            synthetic_file = self.data_dir / 'synthetic_data.csv'
            if not synthetic_file.exists():
                print("    ✗ Synthetic data not available")
                return False
            
            print("    Loading synthetic data...")
            df = pd.read_csv(synthetic_file)
            
            print("    Training models (this may take 1-2 minutes)...")
            manager = ModelManager()
            results = manager.train_all(df)
            
            manager.save_all(str(self.models_dir))
            
            print("  ✓ Models trained and saved")
            for model_name, metrics in results.items():
                if 'r2_score' in metrics:
                    print(f"    - {model_name}: R²={metrics['r2_score']}")
                elif 'accuracy' in metrics:
                    print(f"    - {model_name}: Accuracy={metrics['accuracy']}")
            
            return True
        
        except Exception as e:
            print(f"  ✗ Error training models: {e}")
            return False
    
    def test_digital_twin(self):
        """Test Digital Twin simulation"""
        print("\n[5/6] Testing Digital Twin...")
        try:
            from digital_twin import IntegratedDigitalTwin
            
            print("  Running test simulation...")
            twin = IntegratedDigitalTwin("BW-001")
            
            state = twin.simulate_css_cycle(
                steam_volume=80, steam_pressure=25,
                injection_duration=24, soak_time=36,
                spm=4.0, stroke_length=86,
                vfd_frequency=45, production_cutoff=1.0
            )
            
            summary = twin.get_simulation_summary()
            print(f"  ✓ Digital Twin operational")
            print(f"    - Production: {summary['actual_production']:.2f} BOPD")
            print(f"    - Temperature: {summary['reservoir_temperature']:.1f}°C")
            print(f"    - Failure Risk: {summary['failure_risk']:.2%}")
            
            return True
        
        except Exception as e:
            print(f"  ✗ Error testing Digital Twin: {e}")
            return False
    
    def start_api_server(self):
        """Start FastAPI server"""
        print("\n[6/6] Starting API Server...")
        print("\n" + "="*70)
        print("API Server Starting on http://localhost:8000")
        print("="*70)
        print("\n📊 Dashboard:     http://localhost:8000")
        print("📚 API Docs:      http://localhost:8000/docs")
        print("🔧 ReDoc:         http://localhost:8000/redoc")
        print("\n" + "="*70)
        print("Press CTRL+C to stop the server")
        print("="*70 + "\n")
        
        try:
            import uvicorn
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8000,
                reload=False,
                log_level="info"
            )
        except Exception as e:
            print(f"\n✗ Error starting server: {e}")
            return False
    
    def run(self):
        """Run full startup sequence"""
        self.print_banner()
        
        # Step 1: Create directories
        self.create_directories()
        
        # Step 2: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 3: Generate synthetic data
        if not self.generate_synthetic_data():
            print("\n⚠️  Warning: Synthetic data generation failed")
            print("    You can still continue but some features may not work")
        
        # Step 4: Train models
        if not self.train_models():
            print("\n⚠️  Warning: Model training failed")
            print("    You can still continue but predictions won't work")
            print("    Go to Data & Models tab and retrain manually")
        
        # Step 5: Test Digital Twin
        if not self.test_digital_twin():
            print("\n⚠️  Warning: Digital Twin test failed")
        
        # Step 6: Start API Server
        print("\n✅ Startup sequence complete!\n")
        
        return self.start_api_server()


def main():
    """Main entry point"""
    try:
        manager = StartupManager()
        success = manager.run()
        
        if not success and isinstance(success, bool):
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⛔ Startup interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
