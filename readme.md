# Text-Based Sentiment Detection (Mark II)

## Project Overview

This repository contains a research-focused deep learning project for multi-class emotion and sentiment detection from text. Unlike traditional sentiment analysis systems that limit predictions to positive or negative polarity, this project is designed to identify **11 fine-grained emotional and psychological states** from raw text input, such as anxiety, trauma, stress, joy, and depression.

The entire implementation is intentionally contained within a Jupyter Notebook to preserve **methodological transparency**. Every stage of the pipeline—from data cleaning and linguistic preprocessing to feature engineering, model training, and evaluation—is explicitly visible. This repository is maintained as a **research and academic project**.

Class imbalance is deliberately accepted as a constraint of real-world emotional data rather than artificially corrected which allows the model to learn from naturally occurring distributions.

This project was developed and tested using Python 3.10.6 and TensorFlow 2.14.0. All required dependencies and their tested versions are listed in the provided `requirements.txt` file.
---

## Emotion Classes and Label Encoding

The model predicts one of the following **11 emotion classes**, encoded numerically during training and decoded using a persisted `LabelEncoder`:

0 → addiction  
1 → anger  
2 → anxiety  
3 → depression  
4 → fear  
5 → joy  
6 → love  
7 → sadness  
8 → stress  
9 → surprise  
10 → trauma

These mappings are stored externally to ensure interpretability and reproducibility during future analysis or inference.

---

## Dataset Description

The dataset used in this project is a **custom-constructed corpus** created manually by **our team** 
— *Bishal Ch. Debnath (Pranoy71)*,
— *Aishik Debnath*, 
— *Sagar Jana*
— *Anjana Kumari*
The repository includes a compressed ZIP archive containing the **raw text files and intermediate CSV/Excel sources** that were curated and prepared by us during the early stages of dataset development. These raw files consist of emotion-specific text collections (such as addiction, anger, anxiety, depression, fear, stress, surprise, trauma, and others), along with preliminary train, validation, and test splits used during initial experimentation.

All raw sources from the ZIP archive were **combined, cleaned, normalized, and restructured entirely within the notebook**, following a transparent and reproducible preprocessing pipeline. After applying text normalization, stopword removal, stemming/lemmatization, n-gram generation, and class consolidation, the final unified dataset was saved as `data.csv`.

The resulting `data.csv` contains **39,466 labeled text samples**, each mapped to one of the **11 emotion categories** used to train and evaluate the model. No external datasets are downloaded or referenced at runtime; the final dataset is derived exclusively from the raw files included in this repository, ensuring full traceability from the original text sources to the trained CNN-based sentiment classification model.


---

## Text Preprocessing Pipeline

Prior to feature extraction and modeling, the raw text data undergoes multiple natural language preprocessing steps to improve signal quality and reduce noise. These steps include text normalization, token cleaning, and linguistic reduction techniques.

Lemmatization is applied to reduce words to their base or dictionary form while preserving semantic meaning. Stemming is also explored as part of preprocessing experimentation to analyze its impact on model performance. To capture contextual patterns beyond single words, **n-gram representations** are used, enabling the model to learn from both unigrams and higher-order word sequences.

This preprocessing strategy allows the model to better capture emotionally meaningful expressions rather than isolated tokens.

---

## Feature Engineering and Vocabulary Optimization

Text is converted into numerical form using **CountVectorizer**, producing a bag-of-words representation based on token frequency. Instead of selecting vocabulary size arbitrarily, this project uses **Grid Search–based experimentation** to identify the most efficient `max_features` value that balances representational power and model generalization.

Multiple vocabulary sizes are evaluated, and model performance is compared across configurations. Based on empirical results, the optimal vocabulary size is determined to be:

**max_features = 2100**

This value is subsequently used both as the input dimension for the embedding layer and as the fixed vocabulary size for feature extraction. The trained vectorizer is saved as `CountVectorizer_11_class.pkl` to preserve consistency.

---

## Model Architecture (CNN-Based)

The classification model is built using a **Convolutional Neural Network (CNN) architecture tailored for text data**. CNNs are particularly effective in text classification tasks because they can capture local n-gram–like patterns and spatial dependencies within sequences.

The architecture begins with an **Embedding layer** that transforms sparse token indices into dense 32-dimensional vector representations. This embedding layer uses the optimized vocabulary size of 2100 and produces a sequence-level embedding representation.

A **1D Convolutional layer** with 64 filters and a kernel size of 5 is applied next, enabling the model to learn local feature patterns across adjacent tokens. This is followed by **MaxPooling**, which reduces dimensionality while retaining the most salient features.

The extracted feature maps are then flattened and passed through a series of fully connected (Dense) layers with progressively decreasing dimensionality. These layers allow the model to learn increasingly abstract emotional representations. A Dropout layer is included to mitigate overfitting. The final output layer uses a softmax activation function to produce probability distributions across the 11 emotion classes.

---

## Model Summary

The final model consists of approximately **1.15 million trainable parameters**, all optimized during training. The architecture summary is as follows:

Embedding Layer → Conv1D → MaxPooling1D → Flatten → Dense (32) → Dropout → Dense (16) → Dense (12) → Dense (8) → Dense (11, Softmax)

This design balances expressive power with computational efficiency while remaining interpretable for research analysis.

---

## Training and Evaluation

The model is trained using a train–test split, with performance monitored across epochs using both training and validation metrics. Learning behavior is visualized using **loss and accuracy plots**, enabling inspection of convergence trends and overfitting signals.

Evaluation is performed using multiple metrics to provide a comprehensive assessment of performance, especially under class imbalance:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)

A confusion matrix is also generated to analyze class-wise prediction behavior and misclassification patterns across emotion categories.

---

## Saved Artifacts

To ensure reproducibility and future reuse, key components of the pipeline are saved as standalone files:

- `11_class_sentiment_detection_model_v2.keras` – Trained CNN-based emotion classification model  
- `CountVectorizer_11_class.pkl` – Fitted CountVectorizer with optimized vocabulary size  
- `encoder_11_class.pkl` – LabelEncoder mapping emotion names to numeric labels  
- `data.csv` – Cleaned and finalized dataset used for training and evaluation  
- `Text based sentiment detection mark II.ipynb` – Complete research notebook

---
## Project Lineage and Context

This project is the successor and extended version of my earlier work,
[Text-based-Sentiment-Analysis](https://github.com/Pranoy71/Text-based-sentiment-analysis)

While the previous project focused on 6-class sentiment classification, this version significantly expands both the scope and technical depth of the work. The current project transitions from coarse sentiment polarity to fine-grained emotion recognition, introducing an 11-class emotion taxonomy, a larger and fully custom-built dataset, more advanced preprocessing (including n-grams and feature optimization), and a CNN-based deep learning architecture designed specifically for text emotion classification.

This repository represents a methodological and experimental progression rather than a replacement, preserving continuity while exploring a more expressive and realistic formulation of text-based emotional understanding.

---
## Project Scope and Intent

This repository is maintained as a **research and academic project**, emphasizing reproducibility, and conceptual depth over deployment readiness. It is suitable for academic review, experimentation, and future extension into applied research domains such as mental health–aware NLP, emotional analytics, or psychological text mining.

## Run and test the trained model 

To run and test the trained model, 
- First download the model file and both encoder and countVectorizer file  
  - (`11_class_sentiment_detection_model_v2.keras`)
  - (`CountVectorizer_11_class.pkl`)
  - (`encoder_11_class.pkl`)

- After that run the **Inference code: Loading & Real-time Testing** section of the jupyter notebook from top to bottom. 

