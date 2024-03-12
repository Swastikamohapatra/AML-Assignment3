# test.py

import unittest
from score import score
import joblib
import subprocess
import os
import pytest
import time
import requests

class TestSpam(unittest.TestCase):
    
    def setUp(self):
        self.model = joblib.load('model_logisticregression.pkl')

    def test_smoke_test(self):
        text = 'hi'
        threshold = 0.6
        prediction, propensity = score(text, self.model, threshold)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)

    def test_format_test(self):
        threshold = 0.5
        text = 'some text to test'
        prediction, propensity = score(text, self.model, threshold)
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)

    def test_prediction_values(self):
        threshold = 0.5
        text = 'some text to test'
        prediction, propensity = score(text, self.model, threshold)
        self.assertIn(prediction, [True, False]) 

    def test_propensity_range(self):
        threshold = 0.5
        text = 'some text to test'
        prediction, propensity = score(text, self.model, threshold)
        self.assertGreaterEqual(propensity, 0.0)
        self.assertLessEqual(propensity, 1.0)

    def test_threshold_effect(self):
        threshold = 0.0
        text = 'some text to test'
        prediction, propensity = score(text, self.model, threshold)
        self.assertTrue(prediction)

    def text_threshold_effect_1(self):
        threshold = 1.0
        text = 'some text to test'
        prediction, propensity = score(text, self.model, threshold)
        self.assertFalse(prediction)

    def test_spam_input(self):
        threshold = 0.5
        spam_text = 'Buy now and get rich!'
        prediction, propensity = score(spam_text, self.model, threshold)
        self.assertTrue(prediction)

    def test_non_spam_input(self):
        threshold = 0.5
        non_spam_text = 'hello how are you'
        prediction, propensity = score(non_spam_text, self.model, threshold)
        self.assertFalse(prediction)

@pytest.fixture
def example_text():
    return "URGENT: Exclusive Insurance Offer! Act Now to Secure Your Future! Limited Time Only: Claim Your Policy Before It's Too Late! Don't Miss Out on This Life-Saving Opportunity!"


@pytest.fixture
def example_threshold():
    return 0.5

# Integration test function
def test_flask(example_text, example_threshold):

    flask_process = subprocess.Popen(["python3", 'app.py'], stdout=subprocess.PIPE)
    
    time.sleep(30)

    assert type(example_text) == str
    assert example_threshold >=0 and example_threshold <=1
    
    payload = {
        'text': example_text,
        'threshold': example_threshold
    }
    response = requests.post('http://127.0.0.1:5000/score', json=payload)
    data = response.json()
    prediction = data['prediction']
    propensity = data['propensity']
    assert 'prediction' in data
    assert 'propensity' in data

    prediction_str = "spam" if prediction else "non-spam"


    print(f'The text to be tested was "{example_text}"')
    print(f'It was classified as {prediction_str} with score {propensity}')
    

    print('Closing the app')
    os.system("pkill -f app.py")
    