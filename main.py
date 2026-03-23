import os
import urllib.request
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

dataset_url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
dataset_path = "loan_dataset.csv"

if not os.path.exists(dataset_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_path)

print("Loading and preparing ML dataset...")
df = pd.read_csv(dataset_path)

# Preprocessing: Handle missing values robustly
df = df.drop(columns=['Unnamed: 0', 'Loan_ID'], errors='ignore')

# Fillna dynamically
for col in df.columns:
    if df[col].isna().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False).astype(int)

# Encoding Categorical Features
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Prepare Train Features X and Label y
X = df.drop(columns=['Loan_Status'], errors='ignore')
X = X.fillna(0)
    
# The Kaggle dataset uses 1 (Yes) and 0 (No)
y = pd.to_numeric(df['Loan_Status'], errors='coerce').fillna(0).astype(int)

# Train a simple interpretable Decision Tree Model
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)
feature_names = X.columns.tolist()

print("Model trained successfully.")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/evaluate")
def evaluate(
    api_key: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    married: str = Form(...),
    dependents: str = Form(...),
    education: str = Form(...),
    self_employed: str = Form(...),
    applicant_income: float = Form(...),
    coapplicant_income: float = Form(...),
    loan_amount: float = Form(...),
    loan_amount_term: float = Form(...),
    credit_score: int = Form(...),
    property_area: str = Form(...)
):
    # ==========================
    # 1. Normal AI (Black-Box Gemini via API)
    # ==========================
    normal_decision = ""
    clean_key = api_key.strip()
    
    try:
        if not clean_key:
            raise ValueError("Empty API Key")
        genai.configure(api_key=clean_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        A person has applied for a loan with these details:
        Age: {age}
        Gender: {gender}
        Married: {married}
        Dependents: {dependents}
        Education: {education}
        Self Employed: {self_employed}
        Applicant Income: {applicant_income}
        Coapplicant Income: {coapplicant_income}
        Loan Amount: {loan_amount}
        Loan Amount Term (days): {loan_amount_term}
        Credit Score: {credit_score}
        Property Area: {property_area}

        Should their loan be approved? Answer only 'Yes' or 'No'. No explanation.
        """
        response = model.generate_content(prompt)
        normal_decision = response.text.strip()
    except Exception as e:
        print(f"GEMINI EXCEPTION: {repr(e)}")
        normal_decision = f"Error: {str(e)[:40]}..."

    # ==========================
    # 2. Responsible AI (Explainable Local ML Model)
    # ==========================
    def encode_safe(val, col):
        le = encoders[col]
        if val in le.classes_:
            return le.transform([val])[0]
        else:
            return 0 # Fallback
            
    encoded_gender = encode_safe(gender, 'Gender')
    encoded_married = encode_safe(married, 'Married')
    dep = int(dependents.replace('3+', '3'))
    encoded_education = encode_safe(education, 'Education')
    encoded_self_emp = encode_safe(self_employed, 'Self_Employed')
    encoded_prop = encode_safe(property_area, 'Property_Area')

    # Convert numeric Credit Score back to Kaggle's binary Credit_History for the ML model
    credit_history = 1.0 if credit_score >= 600 else 0.0

    input_data = pd.DataFrame([[
        encoded_gender, encoded_married, dep, encoded_education, encoded_self_emp,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
        credit_history, encoded_prop
    ]], columns=feature_names)

    prediction = clf.predict(input_data)[0]
    resp_decision = "Approved" if prediction == 1 else "Rejected"

    # ==========================
    # 3. Rule-based Custom Explanations & Confidence
    # ==========================
    explanations = []
    riskScore = 0

    if credit_score < 600:
        explanations.append({"feature": "Credit Score", "value": credit_score, "text": "Below threshold (600)", "impact": "High negative", "color": "red"})
        riskScore += 2
    else:
        explanations.append({"feature": "Credit Score", "value": credit_score, "text": "Strong standing (>=600)", "impact": "Positive", "color": "green"})

    if applicant_income < 30000:
        explanations.append({"feature": "Income", "value": f"${applicant_income}", "text": "Below stability threshold ($30k)", "impact": "Moderate negative", "color": "red"})
        riskScore += 1
    elif applicant_income > 80000:
        explanations.append({"feature": "Income", "value": f"${applicant_income}", "text": "High income improves eligibility", "impact": "Positive", "color": "green"})
    else:
        explanations.append({"feature": "Income", "value": f"${applicant_income}", "text": "Standard income range", "impact": "Moderate", "color": "yellow"})

    if age < 21:
        explanations.append({"feature": "Age", "value": age, "text": "Below standard credit age (21)", "impact": "Moderate negative", "color": "red"})
        riskScore += 1
        
    if coapplicant_income < 2000:
        explanations.append({"feature": "Coapplicant Income", "value": f"${coapplicant_income}", "text": "Low additional contribution", "impact": "Low", "color": "yellow"})

    confidence = max(0, 100 - (riskScore * 20))

    # ==========================
    # 4. Bias & Fairness Check (Counterfactual Testing)
    # ==========================
    bias_detected = False
    
    flipped_gender = "Female" if gender == "Male" else "Male"
    encoded_flipped_gender = encode_safe(flipped_gender, 'Gender')
    
    case_b_df = input_data.copy()
    case_b_df['Gender'] = encoded_flipped_gender
    pred_b_val = clf.predict(case_b_df)[0]
    pred_b = "Approved" if pred_b_val == 1 else "Rejected"
    
    if resp_decision != pred_b:
        bias_detected = True

    return {
        "normal_ai_decision": normal_decision,
        "responsible_ai_decision": resp_decision,
        "explanations": explanations,
        "confidence": confidence,
        "bias": {
            "detected": bias_detected,
            "original_case": f"{gender}: {resp_decision}",
            "flipped_case": f"{flipped_gender}: {pred_b}"
        }
    }
