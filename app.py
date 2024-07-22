from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/')
def index():
    return "Welcome to the prediction service!"
POSITIVE_WORDS = {'happy', 'joy', 'good', 'excellent', 'great', 'amazing', 'positive'}
NEGATIVE_WORDS = {'sad', 'bad', 'terrible', 'horrible', 'negative', 'poor'}

def analyze_sentiment(text):
    text = text.lower()
    words = set(text.split())
    positive_score = len(words & POSITIVE_WORDS)
    negative_score = len(words & NEGATIVE_WORDS)
    
    if positive_score > negative_score:
        sentiment = 'POSITIVE'
        score = positive_score / max(1, positive_score + negative_score)
    elif negative_score > positive_score:
        sentiment = 'NEGATIVE'
        score = negative_score / max(1, positive_score + negative_score)
    else:
        sentiment = 'NEUTRAL'
        score = 0.5  # Neutral score

    return sentiment, score

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis_route():
    try:
        request_data = request.json
        text = request_data.get('text', '')

        sentiment, score = analyze_sentiment(text)
        return jsonify({'sentiment': sentiment, 'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/performance-prediction', methods=['POST'])
def performance_prediction():
    try:
        # Retrieve and convert input data
        data = request.json
        no_of_days = int(data.get('days', 0))  # Convert to integer
        max_bid_cpm = float(data.get('max_bid_cpm', 0.0))  # Convert to float
        impressions = int(data.get('impressions', 0))  # Convert to integer
        cost = float(data.get('cost', 0.0))  # Convert to float

        # For debugging
        print(f"Input values for prediction: no_of_days={no_of_days}, max_bid_cpm={max_bid_cpm}, impressions={impressions}, cost={cost}")

        # Dummy prediction logic (replace with your actual prediction code)
        predicted_clicks = (impressions / 1000) * (max_bid_cpm / 100)  # Example calculation
        engagement_rate = (predicted_clicks / impressions) * 100
        cost_per_click = cost / predicted_clicks if predicted_clicks != 0 else 0
        estimated_reach = impressions * (max_bid_cpm / 100)

        # Return prediction results
        return jsonify({
            'predicted_clicks': predicted_clicks,
            'engagement_rate': engagement_rate,
            'cost_per_click': cost_per_click,
            'estimated_reach': estimated_reach
        })

    except Exception as e:
        print(f"Error in /performance-prediction route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/optimize-schedule', methods=['OPTIONS', 'GET'])
def optimize_schedule_route():
    try:
        with open('sentiment_data.pkl', 'rb') as f:
            df, platform_hourly_active_users = pickle.load(f)

        platforms = df['Platform'].unique()
        optimized_schedule = {}
        for platform in platforms:
            platform_data = platform_hourly_active_users[platform].sort_values(ascending=False).head(10)
            optimized_schedule[platform] = platform_data.index.tolist()

        return jsonify(optimized_schedule)
    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/user-activity', methods=['GET'])
def user_activity_route():
    hour = request.args.get('hour', type=int)
    if hour is None:
        return jsonify({'error': 'Hour parameter is required'}), 400

    try:
        with open('sentiment_data.pkl', 'rb') as f:
            df, platform_hourly_active_users = pickle.load(f)
        
        result = {}
        for platform in df['Platform'].unique():
            activity = df[(df['Platform'] == platform) & (df['Hour'] == hour)]
            total_activity = activity['Likes'].sum() + activity['Retweets'].sum()
            result[platform] = total_activity
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/competitor-analysis', methods=['OPTIONS', 'GET'])
def competitor_analysis_route():
    try:
        insights = competitor_analysis()
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/audience-analysis', methods=['OPTIONS', 'POST'])
def audience_analysis_route():
    try:
        with open('social_media_model.pkl', 'rb') as f:
            clf, label_encoders = pickle.load(f)

        request_data = request.json
        user_info = {
            'Gender': request_data['Gender'],
            'DOB': request_data['DOB'],
            'City': request_data['City'],
            'Country': request_data['Country']
        }

        user_info_encoded = {}
        for column in ['Gender', 'City', 'Country']:
            user_info_encoded[column] = label_encoders[column].transform(np.array([user_info[column]]))[0] if user_info[column] in label_encoders[column].classes_ else -1

        user_info_encoded['Age'] = pd.Timestamp('now').year - pd.to_datetime(user_info['DOB']).year

        X_user = np.array([[user_info_encoded['Gender'], user_info_encoded['Age'], user_info_encoded['City'], user_info_encoded['Country']]])

        user_prediction = clf.predict(X_user)[0]

        return jsonify({'predicted_interests': user_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/content-suggestions', methods=['OPTIONS', 'POST'])
def content_suggestions_route():
    try:
        request_data = request.json
        description = request_data.get("product_description", "")

        prompt = (
            f"Create a compelling and engaging piece of content based on the following description: {description}. "
            "Do not include headers or bullet points. Provide the content as a single block of text."
        )

        response = client.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=300
        )

        content = response.choices[0].text.strip()
        return jsonify({'content': content})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/generateImage", methods=["OPTIONS", "POST"])
def generate_image():
    try:
        request_data = request.json
        text = request_data["String"]

        response = client.images.generate(
            model="dall-e-3",
            prompt=text,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        return response.data[0].url
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/sales', methods=['POST'])
def get_sales():
    try:
        # Get JSON data from the request
        with open('advertising_sales_model.pkl', 'rb') as f:
            model, X_train, y_train = pickle.load(f)

        data = request.get_json()
        
        # Log received data
        app.logger.info(f"Received data: {data}")
        
        # Extract the features from the received data
        instagram = data.get('Instagram', [])
        facebook = data.get('Facebook', [])
        youtube = data.get('YouTube', [])
        twitter = data.get('Twitter', [])
        tiktok = data.get('TikTok', [])
        
        # Create a DataFrame from the received data
        input_data = pd.DataFrame({
            'Instagram': instagram,
            'Facebook': facebook,
            'YouTube': youtube,
            'Twitter': twitter,
            'TikTok': tiktok
        })
        
        # Log the input data
        app.logger.info(f"Input DataFrame: {input_data}")
        
        # Predict sales using the model
        predictions = model.predict(input_data)
        
        # Log the predictions
        app.logger.info(f"Predictions: {predictions}")
        
        # Return the predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        # Log the exception
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
