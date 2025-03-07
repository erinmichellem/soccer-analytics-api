from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load Data (Modify Path)
DB_PATH = "database.sqlite"

def load_model():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT m.match_api_id, m.season, m.date, 
           t1.team_long_name AS home_team, 
           t2.team_long_name AS away_team,
           m.home_team_goal, m.away_team_goal
    FROM Match m
    JOIN Team t1 ON m.home_team_api_id = t1.team_api_id
    JOIN Team t2 ON m.away_team_api_id = t2.team_api_id;
    """
    matches = pd.read_sql(query, conn)
    conn.close()

    matches["match_result"] = np.where(matches["home_team_goal"] > matches["away_team_goal"], "Home Win",
                            np.where(matches["home_team_goal"] < matches["away_team_goal"], "Away Win", "Draw"))

    label_encoder = LabelEncoder()
    matches["match_result"] = label_encoder.fit_transform(matches["match_result"])
    
    team_encoder = LabelEncoder()
    matches["home_team"] = team_encoder.fit_transform(matches["home_team"])
    matches["away_team"] = team_encoder.transform(matches["away_team"])

    X = matches[["home_team", "away_team"]]
    y = matches["match_result"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, team_encoder, label_encoder

model, team_encoder, label_encoder = load_model()

@app.route("/predict", methods=["POST"])
def predict_match():
    data = request.json
    home_team_name = data.get("home_team")
    away_team_name = data.get("away_team")

    if home_team_name not in team_encoder.classes_ or away_team_name not in team_encoder.classes_:
        return jsonify({"error": "Invalid team name!"})

    home_team_id = team_encoder.transform([home_team_name])[0]
    away_team_id = team_encoder.transform([away_team_name])[0]

    match_input = pd.DataFrame([[home_team_id, away_team_id]], columns=["home_team", "away_team"])
    pred = model.predict(match_input)[0]
    result = label_encoder.inverse_transform([pred])[0]

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
