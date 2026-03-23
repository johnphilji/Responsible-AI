import os
import urllib.request
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

dataset_url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
dataset_path = "loan_dataset.csv"

if not os.path.exists(dataset_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_path)

df = pd.read_csv(dataset_path)
df = df.drop(columns=['Unnamed: 0', 'Loan_ID'], errors='ignore')

for col in df.columns:
    if df[col].isna().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False).astype(int)

categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop(columns=['Loan_Status'], errors='ignore')
X = X.fillna(0)
y = pd.to_numeric(df['Loan_Status'], errors='coerce').fillna(0).astype(int)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)
feature_names = X.columns.tolist()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/evaluate")
def evaluate(
    applicant_income: float = Form(...),
    coapplicant_income: float = Form(0),
    loan_amount: float = Form(...),
    loan_amount_term: float = Form(...),
    credit_score: int = Form(...),
    credit_history: int = Form(...),
    age: int = Form(30)
):
    # ==========================
    # 0. INPUT EDGE CHECKS
    # ==========================
    if loan_amount > 10_000_000:
        return {"error": "Validation Issue: Loan amount extremely high (> 10 million)."}
    if applicant_income < 0 or loan_amount < 0 or credit_score < 0 or loan_amount_term <= 0:
        return {"error": "Validation Issue: Negative values entered. Please provide valid financial data."}
    if age < 18 or age > 100:
        return {"error": "Validation Issue: Age must be between 18 and 100."}
    if credit_score < 300 or credit_score > 850:
        return {"error": "Validation Issue: Credit Score must be between 300 and 850."}

    # ==========================
    # 1. Normal AI (Gemini APIs)
    # ==========================
    normal_decision = ""
    clean_key = GEMINI_API_KEY.strip()
    
    try:
        if not clean_key:
            raise ValueError("Empty API Key. Please provide it in the UI or hardcode it in main.py")
        genai.configure(api_key=clean_key)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"temperature": 0.1})
        prompt = f"""You are a strict loan approval system. Based only on financial risk, answer strictly YES or NO. No explanation.
Income: ${applicant_income}
Loan: ${loan_amount}
Credit Score: {credit_score}
Term: {loan_amount_term}
History: {credit_history}"""
        response = model.generate_content(prompt)
        raw_ans = response.text.strip().upper()
        if "YES" in raw_ans: normal_decision = "Yes"
        elif "NO" in raw_ans: normal_decision = "No"
        else: normal_decision = "Indeterminate"
    except Exception as e:
        normal_decision = f"API Error: {str(e)}"

    # ==========================
    # 2. Responsible AI (Financial Engine)
    # ==========================
    riskScore = 0
    explanations = []

    # Credit Score Logic
    if credit_score < 600:
        riskScore += 2
        explanations.append({"feature": f"Credit Score ({credit_score})", "text": "High risk range (<600)", "impact": "High risk", "color": "red"})
    elif credit_score <= 750:
        riskScore += 1
        explanations.append({"feature": f"Credit Score ({credit_score})", "text": "Moderate range (600-750)", "impact": "Moderate risk", "color": "yellow"})
    else:
        explanations.append({"feature": f"Credit Score ({credit_score})", "text": "Strong standing (>750)", "impact": "Low risk", "color": "green"})

    # Income Logic
    if applicant_income < 30000:
        riskScore += 1
        explanations.append({"feature": f"Income (${applicant_income:,.0f})", "text": "Below stability threshold (<30k)", "impact": "Risk", "color": "yellow"})
    elif applicant_income > 80000:
        explanations.append({"feature": f"Income (${applicant_income:,.0f})", "text": "Stable high income", "impact": "Positive contribution", "color": "green"})

    # Loan Amount vs Income Logic
    loan_to_income = loan_amount / max(1.0, applicant_income)
    if loan_to_income > 50:
        riskScore += 2
        explanations.append({"feature": f"Loan Amount (${loan_amount:,.0f})", "text": "Extremely high compared to income (>50x)", "impact": "High risk", "color": "red"})
    elif loan_to_income > 20:
        riskScore += 1
        explanations.append({"feature": f"Loan Amount (${loan_amount:,.0f})", "text": "Disproportionately high vs income (>20x)", "impact": "Risk", "color": "yellow"})
    else:
        explanations.append({"feature": f"Loan Amount (${loan_amount:,.0f})", "text": "Within acceptable range", "impact": "Low risk", "color": "green"})

    # Credit History Logic
    if credit_history == 0:
        riskScore += 3
        explanations.append({"feature": "Credit History", "text": "Record indicates bad or missing credit", "impact": "Strong reject signal", "color": "red"})

    # Loan Term Logic
    if loan_amount_term < 180:
        riskScore += 1
        explanations.append({"feature": f"Loan Term ({loan_amount_term} days)", "text": "Aggressive short-term schedule (<180 days)", "impact": "Higher risk", "color": "yellow"})
    elif loan_amount_term >= 720:  # 2 years or more
        riskScore -= 1
        explanations.append({"feature": f"Loan Term ({loan_amount_term} days)", "text": "Extended term reduces monthly burden", "impact": "Positive contribution", "color": "green"})
    else:
        explanations.append({"feature": f"Loan Term ({loan_amount_term} days)", "text": "Standard repayment schedule", "impact": "Neutral", "color": "yellow"})

    # Determine Decision
    if riskScore >= 4:
        resp_decision = "Rejected"
    elif riskScore >= 2:
        resp_decision = "Review"
    else:
        resp_decision = "Approved"

    confidence = min(100, max(0, 100 - (riskScore * 15)))

    # ==========================
    # 3. Validation Fairness Simulation (Using ML)
    # ==========================
    def encode_safe(val, col):
        le = encoders[col]
        return le.transform([val])[0] if val in le.classes_ else 0

    # Base profile for simulation
    def sim_ml(test_gender, test_area):
        input_data = pd.DataFrame([
            [encode_safe(test_gender, 'Gender'), encode_safe('Yes', 'Married'), 0, encode_safe('Graduate', 'Education'), encode_safe('No', 'Self_Employed'),
             applicant_income, coapplicant_income, loan_amount / 1000.0, loan_amount_term,
             float(credit_history), encode_safe(test_area, 'Property_Area')]
        ], columns=feature_names)
        return 'Approved' if clf.predict(input_data)[0] == 1 else 'Rejected'

    ml_base = sim_ml("Male", "Urban")
    ml_female = sim_ml("Female", "Urban")
    ml_rural = sim_ml("Male", "Rural")

    bias_detected = False
    bias_messages = []

    if ml_base != ml_female:
        bias_detected = True
        bias_messages.append({"test": "Gender Simulation (Male vs Female)", "base": ml_base, "flipped": ml_female})
    
    if ml_base != ml_rural:
        bias_detected = True
        bias_messages.append({"test": "Property Area Simulation (Urban vs Rural)", "base": ml_base, "flipped": ml_rural})

    return {
        "normal_ai_decision": normal_decision,
        "responsible_ai_decision": resp_decision,
        "explanations": explanations,
        "confidence": confidence,
        "bias": {
            "detected": bias_detected,
            "messages": bias_messages,
            "base_case": ml_base
        }
    }
