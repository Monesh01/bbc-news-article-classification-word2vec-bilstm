## Topic : BBC News Article Classification using Word2Vec Embeddings and BiLSTM
The NLP model which classifies the News Article into different classes based on the context.

## Motivation
With the rapid growth of digital news content, automatically categorizing news articles has become an important task in natural language processing. Manual classification is time-consuming and inefficient when dealing with large volumes of text data. This project aims to build a BBC News classifier using Word2Vec and Bi-LSTM to automatically identify the category of a news article. By using word embeddings and sequence-based learning, the model can capture semantic meaning and contextual information from text more effectively than traditional methods and techniques.

## Dataset and Preprocessing
The dataset used in this project is the BBC News dataset, which consists of news articles taken from Kaggle. The articles are categorized into multiple classes such as business, entertainment, politics, sports, and technology. Each sample contains the raw text of a news article along with its corresponding label.

Before training the model, the text data is preprocessed to improve learning efficiency. The preprocessing steps include converting text to lowercase, removing punctuation and special characters, tokenizing the text into words, and converting words into numerical sequences. Since news articles vary in length, padding is applied to ensure uniform input size to the sentences used for the LSTM model.

## Model Architecture 
The proposed model follows a sequential deep learning architecture designed for text classification. The preprocessed text sequences are first converted into dense vector representations using a Word2Vec-based embedding layer. These embeddings are then passed through an BiDirectional LSTM layer to capture sequential and contextual information from the text. Finally, dense layers are used to perform multi-class classification and produce the final output probabilities.

### Architecture Flow:

Input Text → Embedding (Word2Vec) → biLSTM → Dense → Softmax Output

## Word2Vec Embedding
Word2Vec is used to convert words into dense vector representations that capture semantic relationships between words. Instead of representing words as sparse one-hot vectors, Word2Vec maps each word to a continuous vector space where similar words have similar representations in a space dimension of vectors.

In this project, the embedding layer transforms the tokenized text sequences into fixed-dimensional vectors, allowing the BiLSTM model to learn meaningful patterns from the text. This embedding-based representation helps the model understand contextual similarity and improves classification performance compared to traditional text representations.

## LSTM Layer

A Bidirectional LSTM (BiLSTM) layer is used to model the sequential nature of text data and capture long-term dependencies between words in a news article. Unlike a standard LSTM, a BiLSTM processes the text in both forward and backward directions, allowing the model to utilize past and future context for better understanding of the sequence.

In this project, a Bidirectional LSTM with 128 units is used to learn rich sequence representations from the embedded text. A dropout rate of 0.3 is applied to reduce overfitting by randomly disabling a fraction of neurons during training. Recurrent dropout is not used in order to maintain stable learning of temporal dependencies.

## Dense Layers and Output

The output from the LSTM layer is passed to a dense layer to perform final classification. The dense layer helps in transforming the learned sequential features into a form suitable for decision making.

The Dense layer is a 32 channel MLP which then passes through layer normalization and ReLU activation for the internal decision making.

The final output layer uses a Softmax activation function to produce probability scores for each news category. The class with the highest probability is selected as the predicted category of the news article. This setup enables effective multi-class classification of BBC news articles.

## Results

The proposed Word2Vec and LSTM-based model achieves strong performance on the BBC News classification task. The model attains a validation accuracy of 0.975, indicating effective learning of semantic and contextual information from news articles. The high accuracy demonstrates the suitability of LSTM-based sequence models for text classification problems.

## Limitations

Despite its good performance, the model has certain limitations. The Word2Vec embeddings used in this project are static and do not capture context-dependent word meanings. Additionally, the model may not generalize well to unseen domains or significantly different news sources. The absence of attention mechanisms may also limit the model’s ability to focus on the most important words in longer articles.

## Future Work

The model can be further improved by incorporating pretrained embeddings such as GloVe or FastText. Using advanced architectures like  attention-based models can enhance contextual understanding. In future work, transformer-based models such as BERT can be explored to achieve better performance and robustness.

## Conclusion

This project successfully demonstrates the use of Word2Vec embeddings and Bi LSTM networks for multi-class text classification. The model effectively captures semantic and sequential information from news articles and achieves high validation accuracy. Overall, the project provides a solid foundation for understanding deep learning-based NLP techniques and can be extended using more advanced models in future work.
