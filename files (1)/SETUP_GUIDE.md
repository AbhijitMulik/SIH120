# 🚀 SIH26120 Digital Twin - Complete Setup & Run Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation (5 minutes)](#installation)
3. [Quick Start (2 minutes)](#quick-start)
4. [Detailed Setup Steps](#detailed-setup-steps)
5. [Running the System](#running-the-system)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB for code and data

### Check Python Version

Open Command Prompt/Terminal and run:
```bash
python --version
```

Should show: `Python 3.8.x` or higher

If not installed, download from [python.org](https://www.python.org/downloads/)

---

## Installation

### Step 1: Download/Clone the Project

**Option A: If you have the folder ready**
- Just open the `SIH26120-Digital-Twin` folder in VS Code

**Option B: If you have a ZIP file**
- Extract the ZIP file
- Remember the location (e.g., `C:\Users\YourName\Desktop\SIH26120-Digital-Twin`)

### Step 2: Open in VS Code

1. Open VS Code
2. Click `File → Open Folder`
3. Navigate to `SIH26120-Digital-Twin` folder
4. Click `Select Folder`

### Step 3: Open Terminal in VS Code

Press: **Ctrl + `** (backtick key) or `Ctrl + Shift + `

You should see a terminal window at the bottom.

### Step 4: Install Dependencies

Copy and paste this command in the terminal:

```bash
pip install -r requirements.txt
```

This will install all required packages:
- numpy, pandas (data handling)
- scikit-learn, xgboost (machine learning)
- fastapi, uvicorn (web server)
- plotly, matplotlib (visualization)

**Wait for it to finish** (usually 2-5 minutes)

You'll see: `Successfully installed ...`

---

## Quick Start

**One command to run everything:**

```bash
python startup.py
```

This will automatically:
1. ✅ Create necessary directories
2. ✅ Check dependencies
3. ✅ Generate synthetic data (if needed)
4. ✅ Train AI models (if needed)
5. ✅ Test the Digital Twin
6. ✅ Start the API server

Wait for the server to start, then:
1. Open browser: http://localhost:8000
2. You'll see the interactive dashboard!

---

## Detailed Setup Steps

### Manual Step-by-Step Setup (If Quick Start doesn't work)

#### Step 1: Create Directories

```bash
mkdir data models logs
```

#### Step 2: Generate Synthetic Data

```bash
python data_generator.py
```

**Output you should see:**
```
Generating data for well BW-001...
Generating data for well BW-002...
...
Dataset saved to data/synthetic_data.csv
Shape: (500, 28)
```

This creates a CSV file with 500 rows of synthetic well data.

#### Step 3: Train AI Models

```bash
python ai_models.py
```

**Output you should see:**
```
Training AI models...
MODEL TRAINING RESULTS
════════════════════════════════════════════════════════════════════════════════

PRODUCTION_MODEL
────────────────────────────────────────────
  model_type..................... XGBoost
  rmse........................... 12.345
  r2_score...................... 0.856
  samples_trained............... 400
...
```

This trains and saves 4 models to the `models/` folder.

#### Step 4: Test Digital Twin

```bash
python digital_twin.py
```

**Output you should see:**
```
================================================================================
DIGITAL TWIN SIMULATION RESULTS
================================================================================
well_id...................... BW-001
reservoir_temperature......... 51.23
oil_viscosity................ 487.65
actual_production............ 45.67
...
```

This tests that the Digital Twin core is working.

#### Step 5: Start the API Server

```bash
python main.py
```

**Output you should see:**
```
================================================================================
SIH26120 DIGITAL TWIN - FastAPI Server
================================================================================

Starting server at http://localhost:8000
API Documentation: http://localhost:8000/docs
================================================================================
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
```

#### Step 6: Open Dashboard in Browser

Open your web browser and go to:
```
http://localhost:8000
```

You should see the interactive dashboard with:
- 📊 Current Well State
- 🛢️ Production Metrics
- ⚠️ Equipment Status
- And tabs for What-If Simulator, Optimization, etc.

---

## Running the System

### Method 1: Quick Start (Recommended)

One command, everything works:
```bash
python startup.py
```

### Method 2: Start Just the Server

If models and data are already generated:
```bash
python main.py
```

### Method 3: Run Individual Components

Generate data only:
```bash
python data_generator.py
```

Train models only:
```bash
python ai_models.py
```

Test Digital Twin:
```bash
python digital_twin.py
```

Test optimizer:
```bash
python optimizer.py
```

---

## Using the Dashboard

### Dashboard Tab (Default View)
1. **Current Well State**: Shows temperature, viscosity, efficiency
2. **Production Metrics**: Shows BOPD, SOR, energy consumption
3. **Equipment Status**: Shows rod load and failure risk
4. Click **Run Simulation** to update metrics

### What-If Simulator Tab
1. Modify CSS parameters (steam volume, soak time, etc.)
2. Modify SRP parameters (SPM, stroke, VFD)
3. Click **Run What-If** to see predicted results
4. Compare "Current" vs "Proposed" scenarios

### Optimization Tab
1. Set number of iterations (30 recommended)
2. Choose priority (Balanced, Production, Efficiency, Cost)
3. Click **Start Optimization** (takes 1-2 minutes)
4. See recommended settings and top 5 solutions

### Data & Models Tab
1. Generate new synthetic data with custom well count/cycles
2. Train AI models
3. View training results and statistics

---

## Troubleshooting

### ❌ Error: "Module not found: fastapi"

**Solution:**
```bash
pip install -r requirements.txt
```

Ensure you're in the correct folder (containing `requirements.txt`)

### ❌ Error: "Port 8000 is already in use"

**Solution:**
```bash
python main.py --port 8001
```

Or kill the process using port 8000:

**Windows:**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

**macOS/Linux:**
```bash
lsof -i :8000
kill -9 <PID_NUMBER>
```

### ❌ Error: "No module named 'main'"

**Solution:**
Make sure you're running from the correct directory:
```bash
cd SIH26120-Digital-Twin
python main.py
```

### ❌ Dashboard doesn't load

**Solution:**
1. Check if server is running (you should see "Uvicorn running...")
2. Try: http://localhost:8000/health
3. If that works, try: http://localhost:8000
4. Check browser console (F12) for errors

### ❌ Models not training

**Solution:**
```bash
python ai_models.py --verbose
```

Check if `data/synthetic_data.csv` exists. If not:
```bash
python data_generator.py
```

### ❌ "Permission denied" on Linux/macOS

**Solution:**
```bash
chmod +x startup.py
python startup.py
```

### ❌ Out of memory error

**Solution:**
Generate smaller dataset:
```python
# In data_generator.py, change:
synthetic_df = generator.generate_complete_dataset(n_cycles=50, n_wells=2)
```

---

## File Structure After Setup

After running the system, your folder should look like:

```
SIH26120-Digital-Twin/
├── data_generator.py              ✓ Python file
├── digital_twin.py                ✓ Python file
├── ai_models.py                   ✓ Python file
├── optimizer.py                   ✓ Python file
├── main.py                        ✓ Python file
├── startup.py                     ✓ Python file
├── dashboard.html                 ✓ HTML file
├── requirements.txt               ✓ Config file
├── config.yaml                    ✓ Config file
├── README.md                      ✓ Documentation
├── SETUP_GUIDE.md                 ✓ This file
│
├── data/                          ✓ GENERATED
│   └── synthetic_data.csv         ✓ GENERATED (500 rows)
│
├── models/                        ✓ GENERATED
│   ├── production_model.pkl       ✓ GENERATED
│   ├── production_model_scaler.pkl ✓ GENERATED
│   ├── temperature_model.pkl      ✓ GENERATED
│   ├── temperature_model_scaler.pkl ✓ GENERATED
│   ├── energy_model.pkl           ✓ GENERATED
│   ├── energy_model_scaler.pkl    ✓ GENERATED
│   ├── failure_risk_model.pkl     ✓ GENERATED
│   └── failure_risk_model_scaler.pkl ✓ GENERATED
│
└── logs/                          ✓ GENERATED (if needed)
```

---

## Testing Each Component

### Test 1: Check Python

```bash
python --version
```

Should show: `Python 3.8.x` or higher

### Test 2: Check Installed Packages

```bash
pip list | grep -E "pandas|numpy|fastapi|xgboost"
```

Should show all packages listed

### Test 3: Run Data Generator

```bash
python data_generator.py
```

Should create `data/synthetic_data.csv`

### Test 4: Run AI Models

```bash
python ai_models.py
```

Should create files in `models/` folder

### Test 5: Run Digital Twin

```bash
python digital_twin.py
```

Should show simulation results

### Test 6: Start Server

```bash
python main.py
```

Should show "Uvicorn running on http://0.0.0.0:8000"

### Test 7: Check Dashboard

Open http://localhost:8000 in browser

Should see colored cards with well metrics

---

## Performance Tips

### For Faster Startup
- Skip model training if models already exist
- Use startup.py (it's smart about skipping done steps)

### For Faster Optimization
- Reduce `n_iterations` to 20-30 (instead of 100)
- Use simpler priority (e.g., "production" vs "balanced")

### For Faster Data Generation
- Reduce number of wells and cycles:
```bash
# In data_generator.py change to:
synthetic_df = generator.generate_complete_dataset(n_cycles=50, n_wells=3)
```

---

## API Testing with CURL

Test API without dashboard:

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

### Test 2: Run Simulation
```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "steam_volume": 80,
    "steam_pressure": 25,
    "injection_duration": 24,
    "soak_time": 36,
    "spm": 4.0,
    "stroke_length": 86,
    "vfd_frequency": 45,
    "production_cutoff": 1.0
  }'
```

### Test 3: Get Predictions
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "steam_volume": 80,
    "steam_pressure": 25,
    "injection_duration": 24,
    "soak_time": 36,
    "spm": 4.0,
    "stroke_length": 86,
    "vfd_frequency": 45
  }'
```

---

## Next Steps

1. **Explore the Dashboard**: Click through all tabs
2. **Run What-If Scenarios**: Change parameters and see predictions
3. **Optimize**: Find best operating conditions
4. **Train Custom Models**: Use your own data (when available)
5. **Integrate Real Data**: Connect to Oil India SCADA

---

## Getting Help

### Common Issues Checklist
- [ ] Python 3.8+ installed?
- [ ] All packages installed? (`pip install -r requirements.txt`)
- [ ] In correct folder? (`cd SIH26120-Digital-Twin`)
- [ ] Port 8000 free? (Change with `--port 8001`)
- [ ] Data generated? (Run `python data_generator.py`)
- [ ] Models trained? (Run `python ai_models.py`)

### Still Need Help?
1. Check the terminal output for error messages
2. Look at `logs/` folder for detailed errors
3. Review README.md for architecture details
4. Check API docs at http://localhost:8000/docs

---

## Summary

**Quick Path to Success:**

1. Open Terminal
2. Go to folder: `cd SIH26120-Digital-Twin`
3. Install: `pip install -r requirements.txt`
4. Run: `python startup.py`
5. Open: http://localhost:8000

**Time to working system: ~10 minutes** ⏱️

---

**Made with ❤️ for better oil field operations**

Version: 1.0.0 | Last Updated: September 2026

