# Spam Email Classifier

A simple Machine Learning project that detects whether a message is Spam or Not Spam (Ham).

This project uses Natural Language Processing (NLP) and the Naive Bayes algorithm to classify SMS or email messages.

---

## About the Project

Spam messages are unwanted messages that often contain advertisements, fake offers, or phishing links.

The goal of this project is to build a machine learning model that can automatically identify spam messages.

---

## Features

- Detects Spam and Ham messages
- Cleans and preprocesses text data
- Converts text into numerical features using TF-IDF
- Trains a Naive Bayes classifier
- Predicts new messages
- Simple Streamlit web application

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit

---

## Project Structure

```text
Spam-Email-Classifier/
│
├── dataset/
│   ├── spam.csv
│   └── clean_spam.csv
│
├── models/
├── notebooks/
│
├── app.py
├── preprocess.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Spam-Email-Classifier.git
```

Go to the project folder

```bash
cd Spam-Email-Classifier
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Run the Project

Preprocess the dataset

```bash
python preprocess.py
```

Train the model

```bash
python train_model.py
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Machine Learning Workflow

1. Load the dataset
2. Clean the data
3. Preprocess the text
4. Convert text into TF-IDF features
5. Split the dataset into training and testing data
6. Train the Naive Bayes model
7. Evaluate the model
8. Predict whether a message is Spam or Ham

---

## Dataset

This project uses the SMS Spam Collection Dataset.

Labels:

- Ham (0) - Normal message
- Spam (1) - Unwanted message

---

## Future Improvements

- Email attachment detection
- Phishing URL detection
- Better user interface
- Deep Learning models
- Multi-language support

---

## Author

Santhosh P

B.E. Computer Science and Engineering

---

## License

This project is created for learning and educational purposes.