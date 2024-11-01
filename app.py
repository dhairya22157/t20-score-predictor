from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# List of teams and cities for dropdowns
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]
cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Auckland', 'Cape Town', 
    'London', 'Barbados', 'Durban', 'St Lucia', 'Wellington', 
    'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 
    'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui', 
    'Chittagong', 'Kolkata', 'Sydney', 'Lahore', 'Delhi', 
    'Nagpur', 'Chandigarh', 'Bangalore', 'St Kitts', 'Cardiff', 
    'Christchurch', 'Trinidad'
]

@app.route('/')
def index():
    return render_template('index.html', teams=sorted(teams), cities=sorted(cities))

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']
    current_score = int(request.form['current_score'])
    overs_done = float(request.form['overs_done'])
    wickets_down = int(request.form['wickets_down'])
    last_five = int(request.form['last_five'])
    
    # Prepare input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'balls_left': [(120 - overs_done * 6)],  # remaining balls
        'wickets_left': [10 - wickets_down],  # remaining wickets
        'crr': [current_score / overs_done],  # current run rate
        'last_five': [last_five]
    })

    # Make prediction
    try:
        prediction = pipe.predict(input_df)[0]
        predicted_score = int(prediction)
        return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=predicted_score)
    except Exception as e:
        return render_template('index.html', teams=sorted(teams), cities=sorted(cities), error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
