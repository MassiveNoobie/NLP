import nltk
from nltk.classify import NaiveBayesClassifier

def extract_features(data):
    return [({word: (word in nltk.word_tokenize(data)) for word in nltk.word_tokenize(data)}, label) for (data, label) in data]

def train_classifier(training_data):
    training_features = extract_features(training_data)
    classifier = NaiveBayesClassifier.train(training_features)
    return classifier

def evaluate_classifier(classifier, test_data):
    test_features = extract_features(test_data)
    accuracy = nltk.classify.accuracy(classifier, test_features)
    return accuracy

training_data = [("This movie is great!", "positive"),
                 ("This movie is terrible!", "negative"),
                 ("This movie is okay.", "neutral")]

test_data = [("This movie is not great!", "positive"),
             ("This movie is terrible!", "negative"),
             ("This movie is okay.", "neutral")]

classifier = train_classifier(training_data)
accuracy = evaluate_classifier(classifier, test_data)
print("Accuracy:", accuracy)
