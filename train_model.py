import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
data = pd.read_csv("student._performance.full.csv")

# Encode categorical
le_gender = LabelEncoder()
le_school = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])
data["School_Type"] = le_school.fit_transform(data["School_Type"])

data["Extracurricular_Activities"] = data["Extracurricular_Activities"].map({"Yes":1, "No":0})

# Features
X = data[[
    "Hours_Studied", "Previous_Scores", "Extracurricular_Activities",
    "Sleep_Hours", "Sample_Question_Papers_Practiced",
    "Attendence", "Gender", "School_Type"
]]

y = data["Score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Model (NO SCALER)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(le_gender, open("le_gender.pkl", "wb"))
pickle.dump(le_school, open("le_school.pkl", "wb"))

print("Model trained successfully!")