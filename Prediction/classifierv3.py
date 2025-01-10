import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
class PromptClassifier:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", model_path="rf.joblib"):
        self.sentence_transformer = SentenceTransformer(embedding_model)
        self.model = RandomForestClassifier()
        self.model_path = model_path

    def get_embeddings(self, texts):
        return self.sentence_transformer.encode(texts, show_progress_bar=True)

    def prepare_data(self, texts, labels):
        if os.path.exists("embeddings.joblib"):
            embeddings = load("embeddings.joblib")
        else:
            embeddings = self.get_embeddings(texts)
            dump(embeddings, "embeddings.joblib")

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        return X_train, X_test, y_train, y_test

    def train_evaluate(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        self.save_model()
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return report

    def predict(self, text):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found. Train the model first.")
        self.model = self.load_model()
        if isinstance(text, str):
            text = [text]
        embeddings = self.get_embeddings(text)
        prediction = self.model.predict(embeddings)
        scores = self.model.predict_proba(embeddings)
        return {
            "prediction": prediction[0],
            "score": scores[0].tolist()
        }

    def save_model(self):
        dump(self.model, self.model_path)

    def load_model(self):
        return load(self.model_path)
