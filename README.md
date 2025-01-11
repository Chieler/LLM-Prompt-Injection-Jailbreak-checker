![Transformer](https://img.shields.io/badge/all_MiniLM_L6_v2-white) ![Model](https://img.shields.io/badge/sklearn-blue) ![Languages](https://img.shields.io/badge/Languages-Python-white)

# Watcher

Watcher is a checker for Prompt Injection/Jailbreak designed with LLms in mind.


# How to use:

1. Clone the repository and set up environment:
```bash
git clone https://github.com/Chieler/LLM-Prompt-Injection-Jailbreak-checker.git
cd LLM-Prompt-Injection-Jailbreak-checker
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```
2. Open up main.py (/Prediction):
```python
import ...
#Locate this part at the top of file and replace it with the text you want to check
test = "Enter text"
```
3. Run code:
```python
python Prediction/main.py
```


# Configuring it for Your Own Use:
The script currently uses ![Database](https://img.shields.io/badge/all_MiniLM_L6_v2-blue) as the transformer, and all .joblib related files in /Resources are modeled off it. If you want to change it you'd have to
re-embed the ds_combined.csv file (data for training) and then re-train the model (RandomForestClassifier() from scikit-learn). This can be done by changing the classifier.py file
```python
def __init__(self, embedding_model={Model to replace}, model_path="new file name"):
```

# Details:
Using ![Database](https://img.shields.io/badge/train_test_split-blue) from scikit-learn, the current implementation has a precision of ~85%, a recall of ~85%, and a f-1 score of ~81%. 
The dataset (ds_combined.csv) has ~460k entries, categorized by 0 (non malicious) and 1 (malicious). It is a combination of the following datasets:
# 
[Deepset](https://huggingface.co/datasets/deepset/prompt-injections) ,
[TrustAIRLab](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts/viewer/jailbreak_2023_05_07/train) ,
[TrustAIRLab](https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set) ,
![verazuo](https://github.com/verazuo/jailbreak_llms) 



