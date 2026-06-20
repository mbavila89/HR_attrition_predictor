from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model artifacts
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("attrition_scaler.pkl")
feature_columns = joblib.load("attrition_features.pkl")

# Fields the user fills in, with the type of input to render
NUMERIC_FIELDS = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Categorical dropdown options (label-encoded fields use 2 options,
# one-hot fields list every category seen during training)
BINARY_FIELDS = {
    "Gender": ["Female", "Male"],
    "OverTime": ["No", "Yes"],
}

ONEHOT_FIELDS = {
    "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                        "Human Resources", "Other"],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
}

BINARY_MAPS = {"Gender": {"Female": 0, "Male": 1}, "OverTime": {"No": 0, "Yes": 1}}


def build_feature_row(form):
    # Turn raw form input into a single-row, encoded, ordered DataFrame
    row = {}

    for field in NUMERIC_FIELDS:
        row[field] = float(form[field])

    for field, mapping in BINARY_MAPS.items():
        row[field] = mapping[form[field]]

    raw_df = pd.DataFrame([row])

    # One-hot encode the multi-category fields the same way training did
    for field in ONEHOT_FIELDS:
        raw_df[field] = form[field]
    encoded = pd.get_dummies(raw_df, columns=list(ONEHOT_FIELDS.keys()), drop_first=True)

    # Align to the exact columns/order the scaler and model expect
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            input_row = build_feature_row(request.form)
            input_scaled = scaler.transform(input_row)
            proba_leave = model.predict_proba(input_scaled)[0][1]
            prediction = "Employee likely to leave" if proba_leave > 0.5 else "Employee likely to stay"
            probability = round(proba_leave * 100, 2)
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        numeric_fields=NUMERIC_FIELDS,
        binary_fields=BINARY_FIELDS,
        onehot_fields=ONEHOT_FIELDS,
        prediction=prediction,
        probability=probability,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
