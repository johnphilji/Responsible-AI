# 🛡️ Responsible-AI Loan Approval System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-purple.svg)

An enterprise-grade, web-based open-source application designed to showcase the critical difference between traditional "black box" Artificial Intelligence and "Responsible, Explainable AI."

<p align="center">
  <img src="screenshots/Screenshot%202026-03-25%20180438.png" alt="Project Screenshot 1" width="32%">
  <img src="screenshots/Screenshot%202026-03-25%20180452.png" alt="Project Screenshot 2" width="32%">
  <img src="screenshots/Screenshot%202026-03-25%20180500.png" alt="Project Screenshot 3" width="32%">
</p>

## 📖 Project Overview

This system evaluates loan applications by comparing two AI paradigms side-by-side:
1. **Normal AI (Black Box):** Uses a generative AI model (Google Gemini 1.5 Flash) to make a binary Approved/Rejected decision without providing structured mathematical reasoning.
2. **Responsible AI (Explainable AI):** Uses a transparent, rule-based scorecard engine that calculates risk based on strict financial logic. It provides deep, feature-level explanations and a confidence score for its decision.
3. **Fairness & Bias Audit Module:** Runs a simulated Machine Learning model (Decision Tree) to check if the applicant would be treated differently based on protected attributes (e.g., Gender, Location).

## 🛠️ Technology Stack
- **Backend:** Python, FastAPI
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Generative AI:** `google-generativeai` (Gemini 1.5 Flash)
- **Machine Learning & Data Science:** `scikit-learn` (DecisionTreeClassifier, LabelEncoder), `pandas`
- **Dataset:** Loan Prediction Dataset from `dphi-official`

## 🚀 How It Works

### The Responsible AI Engine
The core of the application is a **Rule-Based Financial Engine** that calculates a precise risk score using:
- **Credit Score:** Evaluates standard risk brackets (<600, 600-750, >750).
- **Loan-to-Income Ratio:** Checks if the loan requested is disproportionately huge compared to income.
- **Debt-to-Income (DTI) Ratio:** Evaluated across the specified loan term.
- **Credit History:** Severe penalty for missing/bad history.
- **Loan Term:** Very short terms add repayment pressure.

### The ML Validation Fairness Simulation
A Scikit-Learn `DecisionTreeClassifier` is automatically trained on `loan_train.csv` at startup. During evaluation, the application passes the applicant's input through this trained ML model as a "base case" (e.g., Male, Urban). It then flips protected features (e.g., changing Gender from Male to Female) behind the scenes.
If the prediction changes *only* because of these protected attributes, the UI visually flags a **Bias Warning**, proving why black-box ML models need strict oversight.

## 💻 Local Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Responsible-AI.git
   cd Responsible-AI
   ```
2. **Setup virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
5. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```
6. **Open your browser**
   Navigate to `http://localhost:8000`

## 🌐 Deployment Note
This application requires a Python backend (FastAPI) to run the Machine Learning models, manipulate pandas data, and fetch Generative AI results securely. It cannot be hosted purely on static-site servers like GitHub Pages. Recommended free hosting platforms include **Render**, **Railway**, or **Vercel** (using Serverless Python functions).