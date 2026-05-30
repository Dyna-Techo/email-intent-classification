# Email Intent Classification System

A machine learning application that automatically classifies emails into different intent categories using scikit-learn and provides a user-friendly GUI interface.

## 📋 Features

- **Automated Email Classification**: Classifies emails into predefined intent categories
- **GUI Interface**: User-friendly desktop application (customtkinter)
- **ML Model**: Pre-trained scikit-learn models for accurate predictions
- **Data Management**: Storage and management of classified results
- **Model Training**: Capability to train and update models with new data
- **Analytics Dashboard**: View classification results and statistics

## 📁 Project Structure

```
├── src/
│   ├── gui_app.py              # GUI application
│   ├── train_model.py          # Model training
│   ├── predict_engine.py       # Prediction engine
│   └── storage_manager.py      # File I/O management
├── data/
│   └── combined_dataset.csv    # Training data
├── model/                      # Trained models (generated)
├── results/                    # Classification outputs
├── requirements.txt
└── README.md
```

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/email-intent-classification.git
cd email-intent-classification

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Launch GUI
```bash
python src/gui_app.py
```

### Train Model
```bash
python src/train_model.py
```

### Command Line Prediction
```bash
python src/predict_engine.py
```

### Main Menu
```bash
python main.py
```

## 📊 Email Categories

- Education
- Entertainment
- Finance
- General
- Healthcare
- Shopping
- Technology
- Travel

## 📦 Dependencies

- pandas
- scikit-learn
- joblib
- openpyxl
- customtkinter
- matplotlib

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

See CONTRIBUTING.md for guidelines

---

**Classify emails with AI! 🎉**
```
