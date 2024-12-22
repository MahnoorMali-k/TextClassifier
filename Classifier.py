from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Updated Dataset
print("Step 1: Creating the dataset")
texts = [
    # Tech examples
    "I love programming in Python", 
    "Python is a versatile language", 
    "I enjoy solving problems with code", 
    "Software development is fascinating", 
    "Data science and machine learning are exciting fields", 
    "I am learning algorithms to improve coding skills", 
    "Python is great for data analysis", 
    "Coding in Python is fun and rewarding", 
    "Artificial intelligence is a growing field", 
    "Web development using Python is very popular", 

    # Sports examples
    "Football is a popular sport", 
    "I love watching football games", 
    "Sports events are exciting to watch", 
    "Basketball is an exciting team sport", 
    "The Olympics bring the best athletes together", 
    "I enjoy watching soccer games", 
    "Tennis is a fun sport to play and watch", 
    "Running is a great way to stay fit", 
    "Swimming is a great workout", 
    "Baseball games are fun to attend", 
]
labels = [
    "tech", "tech", "tech", "tech", "tech", "tech", "tech", "tech", "tech", "tech", 
    "sports", "sports", "sports", "sports", "sports", "sports", "sports", "sports", "sports", "sports", 
]

# Step 2: Feature Extraction with N-grams
print("Step 2: Extracting features using N-grams")
vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigrams and Bigrams
X = vectorizer.fit_transform(texts)
print(f"Sample N-grams: {vectorizer.get_feature_names_out()[:20]}")  # Show first 20 features
print(f"Feature Matrix Shape: {X.shape}\n")

# Step 3: Splitting Dataset
print("Step 3: Splitting dataset into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
print(f"Training samples: {len(X_train.toarray())}")
print(f"Testing samples: {len(X_test.toarray())}\n")

# Step 4: Training Naive Bayes Classifier
print("Step 4: Training the Naive Bayes Classifier")
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
print("Training complete!\n")

# Step 5: Testing the Classifier
print("Step 5: Testing the Classifier")
y_pred = classifier.predict(X_test)
print("y_pred",y_pred)
print("Predictions on the test set:")
for i, pred in enumerate(y_pred):
    print(f"Test sample {i + 1}: True label = {y_test[i]}, Predicted label = {pred}")
print("\n")

# Step 6: Evaluating the Model
print("Step 6: Evaluating the model")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 7: Classify User Input
print("Step 7: Classify new sentences provided by the user")
while True:
    user_input = input("\nEnter a sentence to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting program. Goodbye!")
        break
    user_features = vectorizer.transform([user_input])  # Transform user input to feature space
    prediction = classifier.predict(user_features)[0]  # Predict class
    print(f"The predicted category is: {prediction}")
