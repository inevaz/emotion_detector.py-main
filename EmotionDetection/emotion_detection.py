import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure the vader_lexicon is downloaded
nltk.download("vader_lexicon", quiet=False)

nltk.download("vader_lexicon", quiet=True)  # Download the lexicon if not already available

def emotion_detector(text_to_analyze):
    # Handle blank or None input (like API status code 400)
    if not text_to_analyze or text_to_analyze.strip() == "":
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": "No text provided",
        }

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text_to_analyze)

    # Adjust emotion scores dynamically
    emotion_scores = {
        "anger": scores["neg"] * 0.6,
        "disgust": scores["neg"] * 0.4 + (0.2 if "disgust" in text_to_analyze.lower() else 0),
        "fear": scores["neg"] * 0.5 + (0.2 if "afraid" in text_to_analyze.lower() or "fear" in text_to_analyze.lower() else 0),
        "joy": scores["pos"],
        "sadness": scores["neg"] * 0.5 + (0.2 if "sad" in text_to_analyze.lower() else 0),
    }

    # Apply a threshold for dominant emotion
    threshold = 0.5
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    if emotion_scores[dominant_emotion] < threshold:
        dominant_emotion = "No clear emotion detected"

    return {
        "anger": emotion_scores["anger"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "joy": emotion_scores["joy"],
        "sadness": emotion_scores["sadness"],
        "dominant_emotion": dominant_emotion,
    }
