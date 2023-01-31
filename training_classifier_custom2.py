import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

# Your training data
training_data = [("This movie is great!", "positive"),
                  ("This movie is terrible!", "negative"),
                  ("This movie is average.", "neutral"),
                  ("This movie is good!", "positive"),
                  ("This movie is bad!", "negative")]

# Your test data (can be the same as training data)
test_data = [("This movie is great!", "positive"),
             ("This movie is terrible!", "negative"),
             ("This movie is average.", "neutral"),
             ("This movie is good!", "positive"),
             ("This movie is bad!", "negative")]

def extract_features(sentence):
    words = word_tokenize(sentence)
    return dict([(word, True) for word in words])

def train_classifier(training_data):
    training_features = [extract_features(sentence) for sentence, label in training_data]
    training_labels = [label for sentence, label in training_data]
    classifier = NaiveBayesClassifier.train(zip(training_features, training_labels))
    return classifier

def evaluate_classifier(classifier, test_data):
    test_features = [extract_features(sentence) for sentence, label in test_data]
    test_labels = [label for sentence, label in test_data]
    results = classifier.classify_many(test_features)
    for sentence, label, result in zip(test_data, test_labels, results):
        print(f"Sentence: {sentence[0]}")
        print(f"Expected: {label}")
        print(f"Result: {result}\n")

classifier = train_classifier(training_data)
evaluate_classifier(classifier, test_data)
