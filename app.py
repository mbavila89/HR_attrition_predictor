from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model artifacts
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("attrition_scaler.pkl")
feature_columns = joblib.load("attrition_features.pkl")

# --- True continuous numeric fields: rendered as number inputs with sane ranges ---
# (field, label, min, max, default/median, unit/help text)
NUMERIC_FIELDS = [
    ("Age", "Age", 18, 60, 36, "years"),
    ("DailyRate", "Daily Rate", 100, 1500, 800, "$ per day"),
    ("DistanceFromHome", "Distance From Home", 1, 30, 7, "miles"),
    ("HourlyRate", "Hourly Rate", 30, 100, 66, "$ per hour"),
    ("MonthlyIncome", "Monthly Income", 1000, 20000, 4900, "$ per month"),
    ("MonthlyRate", "Monthly Rate", 2000, 27000, 14000, "$"),
    ("NumCompaniesWorked", "Number of Companies Worked At", 0, 9, 2, "companies"),
    ("PercentSalaryHike", "Percent Salary Hike (last review)", 11, 25, 14, "%"),
    ("TotalWorkingYears", "Total Working Years", 0, 40, 10, "years"),
    ("TrainingTimesLastYear", "Training Sessions Last Year", 0, 6, 3, "sessions"),
    ("YearsAtCompany", "Years At This Company", 0, 40, 5, "years"),
    ("YearsInCurrentRole", "Years In Current Role", 0, 18, 3, "years"),
    ("YearsSinceLastPromotion", "Years Since Last Promotion", 0, 15, 1, "years"),
    ("YearsWithCurrManager", "Years With Current Manager", 0, 17, 3, "years"),
]

# --- Small discrete-range fields: no official text labels, shown as a number select ---
DISCRETE_FIELDS = [
    ("JobLevel", "Job Level", 1, 5, "1 = entry level, 5 = most senior"),
    ("StockOptionLevel", "Stock Option Level", 0, 3, "0 = none, 3 = highest"),
]

# --- Ordinal fields with official IBM HR dataset text labels: shown as labeled dropdowns ---
ORDINAL_LABELS = {
    "Education": "Education Level",
    "EnvironmentSatisfaction": "Environment Satisfaction",
    "JobInvolvement": "Job Involvement",
    "JobSatisfaction": "Job Satisfaction",
    "PerformanceRating": "Performance Rating",
    "RelationshipSatisfaction": "Relationship Satisfaction",
    "WorkLifeBalance": "Work-Life Balance",
}

ORDINAL_FIELDS = {
    "Education": [
        (1, "Below College"), (2, "College"), (3, "Bachelor's"),
        (4, "Master's"), (5, "Doctorate"),
    ],
    "EnvironmentSatisfaction": [
        (1, "Low"), (2, "Medium"), (3, "High"), (4, "Very High"),
    ],
    "JobInvolvement": [
        (1, "Low"), (2, "Medium"), (3, "High"), (4, "Very High"),
    ],
    "JobSatisfaction": [
        (1, "Low"), (2, "Medium"), (3, "High"), (4, "Very High"),
    ],
    "PerformanceRating": [
        (1, "Low"), (2, "Good"), (3, "Excellent"), (4, "Outstanding"),
    ],
    "RelationshipSatisfaction": [
        (1, "Low"), (2, "Medium"), (3, "High"), (4, "Very High"),
    ],
    "WorkLifeBalance": [
        (1, "Bad"), (2, "Good"), (3, "Better"), (4, "Best"),
    ],
}

# --- Binary categorical fields: label-encoded under the hood ---
BINARY_FIELDS = {
    "Gender": ["Female", "Male"],
    "OverTime": ["No", "Yes"],
}
BINARY_MAPS = {"Gender": {"Female": 0, "Male": 1}, "OverTime": {"No": 0, "Yes": 1}}

# --- Multi-category fields: one-hot encoded under the hood ---
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


def build_feature_row(form):
    # Turn raw form input into a single-row, encoded, ordered DataFrame
    row = {}

    for field, *_ in NUMERIC_FIELDS:
        row[field] = float(form[field])

    for field, *_ in DISCRETE_FIELDS:
        row[field] = float(form[field])

    for field in ORDINAL_FIELDS:
        row[field] = float(form[field])

    for field, mapping in BINARY_MAPS.items():
        row[field] = mapping[form[field]]

    raw_df = pd.DataFrame([row])

    for field in ONEHOT_FIELDS:
        raw_df[field] = form[field]
    encoded = pd.get_dummies(raw_df, columns=list(ONEHOT_FIELDS.keys()), drop_first=True)

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
        discrete_fields=DISCRETE_FIELDS,
        ordinal_fields=ORDINAL_FIELDS,
        ordinal_labels=ORDINAL_LABELS,
        binary_fields=BINARY_FIELDS,
        onehot_fields=ONEHOT_FIELDS,
        prediction=prediction,
        probability=probability,
        error=error,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
