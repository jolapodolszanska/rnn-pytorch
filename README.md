**This project is part of the assignments for students in the computer science major in the subject of Artificial Neural Networks and Deep Learning.**

# Sentiment Analysis with LSTM RNN using PyTorch

This project focuses on building a sentiment analysis model using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) units. The dataset used is the IMDb Movie Reviews dataset, and the model utilizes GloVe embeddings for word representation.

## Dataset

The dataset used in this project is the IMDb Movie Reviews dataset, which contains 50,000 reviews labeled as positive or negative. The dataset is split into a training set and a validation set.

## Project Structure

- `data/`: Contains the dataset and pre-trained GloVe embeddings.
- `notebooks/`: Jupyter Notebooks used for experimentation and prototyping.
- `src/`: Source code for the project, including data processing, model definition, and training scripts.
- `models/`: Saved models and checkpoints.

## Preprocessing

1. **Data Cleaning**: The text data undergoes cleaning by removing HTML tags, non-alphabetic characters, and expanding contractions.
2. **Tokenization**: The text is tokenized using a basic English tokenizer, and a vocabulary is built. Each word is assigned an index based on its frequency in the dataset.
3. **Padding**: The sequences are padded to ensure that all input vectors are of uniform length, which is essential for batch processing in RNNs.

## GloVe Embeddings

GloVe embeddings are used to represent words as vectors in a high-dimensional space. The embeddings are loaded from a pre-trained model (`glove.840B.300d.pkl`), and a matrix is created for the words in the vocabulary. These embeddings are frozen during training to prevent modification.

## Model Architecture

The sentiment analysis model is built using PyTorch and includes the following layers:

- **Embedding Layer**: Initialized with pre-trained GloVe embeddings.
- **LSTM Layer**: A bi-directional LSTM layer captures context from both directions.
- **Dropout Layer**: A dropout layer helps prevent overfitting.
- **Fully Connected Layer**: A dense layer outputs the final sentiment classification, which consists of two classes: positive and negative.

## Training

The training process involves:

- **Loss Function**: Cross-Entropy Loss is used as the loss function, which is suitable for classification tasks.
- **Optimizer**: The Adam optimizer is employed for its efficiency in handling sparse gradients.
- **Batch Size**: A batch size of 64 is used.
- **Epochs**: The model is trained for 5 epochs.

## Evaluation

The model's performance is evaluated on the validation set using metrics such as loss and accuracy. The evaluation involves computing the loss and accuracy for each epoch.

## Visualization

The project includes functions to plot the loss and accuracy over epochs to visualize the training and validation process.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
