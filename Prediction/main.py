import pandas as pd
from classifierV3 import PromptClassifierV3
from classifier import PromptClassifier
from classifierV2 import PromptClassifierV2
import emoji
test = "Text you want to test"
def text_has_emoji(text):
    for character in text:
        if character in emoji.EMOJI_DATA:
            return True
    return False
def run_classification(texts, labels):
    classifier = PromptClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data(texts, labels)
    results = classifier.train_evaluate(X_train, X_test, y_train, y_test)
    return classifier, results
def predict(text):
    classifier = PromptClassifier()
    return classifier.predict(text)
def run_inference(text):
"""
Method for predicting how dangerous a prompt is, artificially add .05 to texts with emojies
"""
    if text_has_emoji(text):
        print("has emoji")
        results = predict(text)
        results['score'][1] = results['score'][1]+ 0.05
        return results
    else:
        return predict(test)
if __name__ =="__main__":

    results = run_inference(test)
    print(results)
