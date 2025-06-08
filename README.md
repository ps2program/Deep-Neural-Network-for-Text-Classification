# Deep Neural Network for Text Classification

This project implements a deep neural network for text classification using the Yelp Review Polarity dataset. The model is designed to classify text reviews into positive or negative sentiments.

## Repository

```bash
https://github.com/ps2program/Deep-Neural-Network-for-Text-Classification.git
```

## Project Structure

```
Deep-Neural-Network-for-Text-Classification/
├── data/                   # Directory for dataset (not tracked in git)
│   └── yelp_review_polarity_csv/
├── models/                 # Directory for saved model checkpoints (not tracked in git)
├── .gitignore             # Git ignore rules
├── requirements.txt        # Project dependencies
├── setup.py               # Setup script for environment and dependencies
├── DNN_Assignment01_Template.ipynb  # Main notebook for implementation
└── README.md              # This file
```

## Features

- Text preprocessing including cleaning, tokenization, and lemmatization
- TF-IDF vectorization for text feature extraction
- Deep feedforward neural network architecture
- Model training with early stopping and checkpointing
- Comprehensive evaluation metrics
- Performance analysis and visualization

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ps2program/Deep-Neural-Network-for-Text-Classification.git
cd Deep-Neural-Network-for-Text-Classification
```

2. Run the setup script to create a virtual environment and install dependencies:
```bash
python setup.py
```

3. Activate the virtual environment:
- On Windows:
```bash
.\.venv\Scripts\activate
```
- On macOS/Linux:
```bash
source ./.venv/bin/activate
```

4. Download the Yelp Review Polarity dataset:
```bash
# The dataset will be downloaded automatically when running the notebook
# or you can manually download it from:
# https://www.kaggle.com/datasets/yelp-dataset/yelp-review-polarity
```

## Dependencies

The project uses the following main dependencies:
- TensorFlow (>=2.17.0)
- Pandas (>=2.0.0)
- NumPy (>=1.26.0)
- scikit-learn (>=1.5.0)
- NLTK (>=3.8.1)
- Matplotlib (>=3.8.0)
- Seaborn (>=0.13.0)
- Jupyter (>=1.0.0)

## Model Architecture

The deep neural network consists of:
- Input layer with 2000 features (TF-IDF vectorization)
- Hidden layers with 512, 256, and 128 neurons
- Dropout layers for regularization
- Output layer with sigmoid activation for binary classification

## Training

The model is trained with:
- Adam optimizer
- Binary cross-entropy loss
- Early stopping to prevent overfitting
- Model checkpointing to save the best model

## Evaluation

The model's performance is evaluated using:
- Test accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Training and validation curves

## Performance Analysis

The project includes analysis of:
- Training and validation accuracy curves
- Training and validation loss curves
- Learning rate analysis
- Model complexity analysis

## Future Improvements

Potential areas for improvement:
1. Experiment with different architectures
2. Implement hyperparameter tuning
3. Try different text preprocessing techniques
4. Explore alternative word embeddings
5. Implement cross-validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Prahlad

## Acknowledgments

- Yelp Review Polarity dataset
- TensorFlow team for the deep learning framework
- scikit-learn for machine learning tools 