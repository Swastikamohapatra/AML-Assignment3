#score.py

def score(text, model, threshold):

    # Perform prediction
    propensity = model.predict_proba([text])[0][1]  # Assuming the positive class is at index 1

    # Apply threshold for binary classification
    prediction = bool(propensity >= threshold)
    

    return bool(prediction), float(propensity)