import os
import urllib.request
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

dataset_url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
dataset_path = "loan_dataset.csv"

# Download the Kaggle dataset replica if not present
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

# Clean the 'Dependents' column (e.g., '3+')
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
    gender: str = Form(...),
    married: str = Form(...),
    dependents: str = Form(...),
    education: str = Form(...),
    self_employed: str = Form(...),
    applicant_income: float = Form(...),
    coapplicant_income: float = Form(...),
    loan_amount: float = Form(...),
    loan_amount_term: float = Form(...),
    credit_history: float = Form(...),
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
        # We can use gemini-2.5-flash or gemini-pro. gemini-1.5-flash is safer if 2.5 isn't available everywhere
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        A person has applied for a loan with these details:
        Gender: {gender}
        Married: {married}
        Dependents: {dependents}
        Education: {education}
        Self Employed: {self_employed}
        Applicant Income: {applicant_income}
        Coapplicant Income: {coapplicant_income}
        Loan Amount: {loan_amount}
        Loan Amount Term (days): {loan_amount_term}
        Credit History (1=Good, 0=Bad): {credit_history}
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

    input_data = pd.DataFrame([[
        encoded_gender, encoded_married, dep, encoded_education, encoded_self_emp,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
        credit_history, encoded_prop
    ]], columns=feature_names)

    prediction = clf.predict(input_data)[0]
    resp_decision = "Approved" if prediction == 1 else "Rejected"

    # Explanation Logic using Decision Tree Feature Importances
    importances = clf.feature_importances_
    top_indices = importances.argsort()[-3:][::-1] # Top 3 features
    
    explanation_parts = []
    for idx in top_indices:
        feat = feature_names[idx]
        imp = importances[idx]
        if imp > 0.00: # Always show the top features for full transparency
            val = input_data.iloc[0][feat]
            # Convert back to readable text
            readable_val = val
            if feat in encoders:
                readable_val = encoders[feat].inverse_transform([int(val)])[0]
            
            impact = "High" if imp > 0.3 else ("Moderate" if imp > 0.1 else "Low")
            explanation_parts.append({
                "feature": feat.replace('_', ' '),
                "value": readable_val,
                "impact": impact
            })

    return {
        "normal_ai_decision": normal_decision,
        "responsible_ai_decision": resp_decision,
        "explanations": explanation_parts
    }
