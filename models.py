
from sklearn.ensemble import RandomForestClassifier

def get_model(params):
    return RandomForestClassifier(**params, random_state=42)
