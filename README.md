🏥 Patient Appointment No-Show Prediction

> An interactive **Streamlit dashboard** that predicts patient no-shows for medical appointments using machine learning and **SHAP explainability**.  
> Built with an end-to-end pipeline: preprocessing → training → evaluation → interactive app.

---

## 🚀 Features

- 🤖 **ML-powered predictions** — Predict whether a patient is likely to miss their appointment.  
- 📊 **End-to-end pipeline** — Clean data, train models, evaluate, and deploy.  
- 🧠 **Explainability with SHAP** — Understand which factors drive the model’s predictions.  
- 🧭 **Interactive UI** — User-friendly input for non-technical users.  
- 📈 **Model metrics dashboard** — Real-time accuracy, precision, recall, and confusion matrix.

---

## 🧰 Tech Stack & Libraries

| Category                | Tools & Libraries                                                |
|--------------------------|------------------------------------------------------------------|
| 💻 Frontend (App)        | [Streamlit](https://streamlit.io/)                               |
| 🧠 ML / Modeling         | [XGBoost](https://xgboost.ai/), [scikit-learn](https://scikit-learn.org/) |
| 🧼 Preprocessing         | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| 📊 Visualization         | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| 🧠 Explainability        | [SHAP](https://shap.readthedocs.io/)                              |
| 🧾 Model Management      | [joblib](https://joblib.readthedocs.io/)                          |

---
📊 Dataset

This project uses the No-show Medical Appointments
 dataset from Kaggle, which contains 110,527 appointment records collected from public hospitals in Brazil.

🏥 Key Features:

Gender — Patient gender

Age — Patient age

ScheduledDay / AppointmentDay — When the appointment was booked and when it occurs

Neighbourhood — Location of the hospital

MedicalCoverage (originally Scholarship) — Indicates if the patient receives government health coverage

Hypertension, Diabetes, Alcoholism, Handcap — Health conditions

SMS_received — Whether the patient received a reminder SMS

No-show (Target) — Indicates if the patient attended (No) or missed (Yes) the appointment.
## 📂 Project Structure

```

📦 Patient-Appointment-Prediction
├── 📁 src
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train.py               # Model training
│   ├── evaluate.py            # Evaluation & metrics
│   └── app_streamlit.py       # Streamlit UI
├── 📁 data
│   ├── raw/
│   └── processed/
├── 📁 models
│   └── best_model.joblib
├── 📁 notebooks               # Exploratory analysis
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 🧪 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ms-codess/Patient-Appointment-Prediction.git
cd Patient-Appointment-Prediction
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run src/app_streamlit.py
```

👉 Open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🧠 Model Explainability Example

| Feature                 | SHAP Impact                    |
| ----------------------- | ------------------------------ |
| Waiting Time (31+ days) | 🔺 Increases no-show risk      |
| Age Group (Senior)      | 🟩 Decreases no-show risk      |
| SMS Reminder            | 🟩 Reduces missed appointments |

🧭 The dashboard uses **SHAP summary plots** to make model decisions transparent.

