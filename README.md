# Household Waste Classification System (EcoSort)

A lightweight Deep Learning image classification system designed for automatic segregation of household waste into 12 categories using **MobileNetV2 Transfer Learning**, **TensorFlow/Keras**, and a responsive **Flask** web application.

---

## 📊 Dataset & Categories

The system uses the 12-class dataset located in `dataset/archive/garbage_classification/`:

| # | Category | Waste Stream | Recommended Bin |
|---|---|---|---|
| 1 | **battery** | Hazardous & E-Waste | 🔴 Red / E-Drop Depot |
| 2 | **biological** | Organic / Compostable | 🟢 Green Bin |
| 3 | **brown-glass** | Recyclable Glass | 🔵 Blue Bin |
| 4 | **cardboard** | Paper & Cardboard | 🔵 Blue Bin |
| 5 | **clothes** | Textile Reuse / Recycling | 🟡 Yellow / Purple Bin |
| 6 | **green-glass** | Recyclable Glass | 🔵 Blue Bin |
| 7 | **metal** | Dry Recyclable (Metals) | 🔵 Blue Bin |
| 8 | **paper** | Dry Recyclable (Paper) | 🔵 Blue Bin |
| 9 | **plastic** | Dry Recyclable (Plastics) | 🔵 Blue Bin |
| 10 | **shoes** | Footwear & Textiles | 🟡 Yellow / Purple Bin |
| 11 | **trash** | Residual / Non-Recyclable | ⚫ Black / Grey Bin |
| 12 | **white-glass** | Recyclable Glass (Clear) | 🔵 Blue Bin |

---

## 📈 Empirical Test Results

Evaluated on an **isolated held-out test set** (540 images, 45 per category; zero data leakage):

* **Overall Test Accuracy:** `88.33%`
* **Macro F1-Score:** `88.11%`
* **Weighted F1-Score:** `88.11%`

### Per-Class Test Performance:
```text
              precision    recall  f1-score   support

     battery       0.93      0.93      0.93        45
  biological       0.93      0.96      0.95        45
 brown-glass       0.88      0.84      0.86        45
   cardboard       0.93      0.87      0.90        45
     clothes       0.94      0.98      0.96        45
 green-glass       0.87      0.91      0.89        45
       metal       0.84      0.82      0.83        45
       paper       0.88      0.96      0.91        45
     plastic       0.86      0.56      0.68        45
       shoes       0.96      0.96      0.96        45
       trash       0.93      0.96      0.95        45
 white-glass       0.68      0.87      0.76        45

    accuracy                           0.88       540
   macro avg       0.89      0.88      0.88       540
weighted avg       0.89      0.88      0.88       540
```

* **Confusion Matrix Plot:** `static/confusion_matrix.png`
* **Convergence History Plot:** `static/training_history.png`
* **Metrics Data JSON:** `evaluation_results.json`

---

## 📁 Project Structure

```text
Waste_Classification_Project/
├── app.py                      # Flask backend API & MobileNetV2 inference engine
├── train.py                    # Transfer learning training & test evaluation script
├── waste_classifier.keras       # Trained MobileNetV2 neural network weights
├── class_names.json            # 12 category labels in alphabetical index order
├── waste_metadata.json         # Bin colors, waste streams, and disposal guidance
├── evaluation_results.json     # Empirical test evaluation metrics (Accuracy, F1, CM)
├── test_split.json             # Held-out test set filepaths and labels
├── requirements.txt            # Python library dependencies
├── README.md                   # Complete documentation & commands
├── dataset/
│   └── archive/
│       └── garbage_classification/  # 12 category folders
├── static/
│   ├── style.css               # Clean green-and-white responsive CSS stylesheet
│   ├── script.js               # Drag-and-drop, real-time prediction, and top-3 charts
│   ├── confusion_matrix.png    # Test evaluation confusion matrix visualization
│   └── training_history.png    # Training & validation accuracy/loss convergence curve
└── templates/
    └── index.html              # Responsive landing page with classifier and metrics
```

---

## 🚀 Exact Commands to Train, Test, and Run

### 1. Activate the Virtual Environment
Open PowerShell in `c:\Waste_Classification_Project`:
```powershell
.\.venv\Scripts\activate
```

### 2. Install / Verify Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Train & Evaluate the Model
To re-train the model and automatically generate the test evaluation report:
```powershell
python train.py --max-samples-per-class 300 --epochs 4 --batch-size 32
```
*Note: This automatically isolates the test set (15%), computes actual test metrics (accuracy, precision, recall, F1-score), exports `static/confusion_matrix.png` and `static/training_history.png`, and saves the weights to `waste_classifier.keras`.*

### 4. Run the Website Locally
```powershell
python app.py
```

### 5. Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

Upload any waste photo (or sample from `dataset/archive/garbage_classification/`) to see real-time MobileNetV2 classification, confidence percentages, bin color badges, and recycling advice.
