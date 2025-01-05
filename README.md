# Naive Bayes Text Classifier with N-grams

This repository contains a **Naive Bayes Text Classifier** implemented in Python, designed to classify textual data into predefined categories. The project leverages the **Naive Bayes algorithm**, a probabilistic machine learning approach based on Bayes' Theorem, which assumes independence between features given the class label.

## Features
- **Text Preprocessing**:
  - Tokenization of text into words.
  - Removal of stop words to reduce noise.
  - Conversion of text to lowercase for uniformity.
  - Optional stemming or lemmatization for word normalization.

- **Feature Extraction**:
  - Utilizes **n-grams** (e.g., unigrams, bigrams, trigrams) to capture contextual patterns and relationships between words in the text.

- **Training the Model**:
  - Trains on labeled datasets where each text sample is associated with a category.
  - Computes prior probabilities of each class and conditional probabilities of n-grams given the class.

- **Classification**:
  - Assigns a category to new, unseen text by calculating the posterior probabilities for each class and selecting the one with the highest probability.

- **Evaluation**:
  - Measures performance using metrics such as accuracy.

## Use Cases
- **Sentiment Analysis**: Classify text as positive, negative, or neutral.
- **Spam Detection**: Categorize emails or messages as spam or not.
- **Topic Categorization**: Automatically tag articles or documents with relevant topics.

## How It Works

1. **Data Preparation**:
   - Load and preprocess the dataset for classification.
   - Perform tokenization, stop word removal, and optional stemming/lemmatization.

2. **Feature Engineering**:
   - Extract n-grams from text to represent contextual patterns.
   - Compute frequencies or probabilities for these n-grams.

3. **Model Training**:
   - Train the Naive Bayes model using labeled data.
   - Calculate prior and conditional probabilities for classification.

4. **Classification and Evaluation**:
   - Classify new text samples using the trained model.
   - Evaluate performance using metrics like accuracy.

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/MahnoorMali-k/TextClassifier.git
   cd TextClassifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python src/Classifier.py
   ```

## Project Structure

```plaintext
NaiveBayesTextClassifier/
├── src/
│   ├── classifier.py       # Script to train and evaluate the model
├── output/                       # Directory for saving output images
├── README.md                     # Project documentation
├── requirements.txt              # List of dependencies
```

## Dependencies

Ensure the following libraries are installed:

- Python 3.7+
- NumPy
- Scikit-learn
- NLTK

Install them using:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy text classifying! 🚀
