# Robust Efficient Global Optimization with Dual Uncertainty Quantification

This repository contains the implementation for our paper: **"Robust Efficient Global Optimization Using Random Forest with Dual Uncertainty Quantification for Microalgae Cultivation Experiments"**. 

Our approach enhances optimization efficiency in microalgae cultivation experiments by incorporating both model uncertainty and input uncertainty into a Random Forest-based EGO (Efficient Global Optimization) algorithm.
## 📁 Repository Structure
├── microalgae_model/ # Microalgae growth kinetics model for algorithm evaluation
├── robust_RF_algae.py # Main experimental code
├── integrated.py # Core implementation of uncertainty-robust surrogate model
├── setup.py # Compilation script for Windows systems
├── extension.pyx # Modified component from golom1.0
└── golem.py # Modified component from golom1.0
## 📝 Key Features

- Dual uncertainty quantification (model + input) in Random Forest-based EGO
- Robust optimization framework for microalgae cultivation experiments
- Modified implementation of golom1.0 components for our specific needs

## ⚙️ Installation & Setup

### Windows 11
1. Compile the extension module in the repository root:
   ```bash
   python setup.py build_ext --inplace
This will generate extension.pyd which is imported in robust_RF_algae.py.
Ubuntu/Linux
Compile extension.pyx to extension.c for use in the system.

📄 Dependencies
Python 3.9

Required Python packages (list your dependencies here)

numpy

scipy

scikit-learn

skopt

geatpy

⁉️ Support
For questions or issues, please open an issue in this repository.

text

You should:
1. Fill in the proper citation information where indicated
2. Add all the specific Python dependencies your project requires
3. Include any additional usage instructions specific to your implementation
4. Add any relevant publication links or project websites

Would you like me to modify or expand any particular section?
