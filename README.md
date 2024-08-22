# Network Intrusion Detection System Using Machine Learning

In the evolving landscape of cyber threats, robust network security is crucial. Traditional Intrusion Detection Systems (IDS) often struggle to keep up with sophisticated attacks. This project leverages AI, combining Machine Learning (ML) and Deep Learning (DL) techniques to enhance the detection and prediction of network intrusions using advanced datasets like NSL-KDD and UNSW-NB 15.

## Dataset Overview

### NSL-KDD & UNSW-NB 15 Datasets

- **NSL-KDD**: A refined version of the KDD'99 dataset, addressing issues like redundant records and biases, making it a reliable benchmark for evaluating IDS models.

  ![NSL-KDD Test Dataset](https://github.com/Mugeshcybersecurity/Network-Intrusion-Detection-System-Using-Machine-Learning/blob/main/Dataset/nsl-kdd/KDDTest1.jpg)
  ![NSL-KDD Train Dataset](https://github.com/Mugeshcybersecurity/Network-Intrusion-Detection-System-Using-Machine-Learning/blob/main/Dataset/nsl-kdd/KDDTrain1.jpg)

- **UNSW-NB 15**: Provides a blend of real and synthetic attack data, offering a complex environment for testing modern IDS models.

### Why Not KDD'99?

The KDD'99 dataset, despite its historical significance, suffers from data redundancy and biases, which can lead to skewed model performance. NSL-KDD resolves these issues, offering a more balanced and reliable dataset.

## Attack Type Categorization Table

| Attack Type | Description | Attack Labels |
|-------------|-------------|---------------|
| DoS         | Denial of Service - attacks that shut down a network making it inaccessible to its intended users. | apache2, back, land, neptune, mailbomb, pod, processtable, smurf, teardrop, udpstorm, worm (...) |
| R2L         | Root to Local - unauthorized access from a remote machine. | ftp_write, guess_passwd, httptunnel, imap, multihop, named (...) |
| Probe       | Surveillance and other probing, such as port scanning. | ipsweep, mscan, nmap, portsweep, saint, satan (...) |
| U2R         | User to Root - unauthorized access to local superuser privileges. | buffer_overflow, loadmodule, perl, ps, rootkit, sqlattack (...) |

## Methodology

### Data Preprocessing

1. **Loading Datasets**: Using Pandas to load training and test sets.
2. **Feature Selection**: Dropping irrelevant features and normalizing numerical ones.
3. **Encoding**: One-hot encoding categorical features and label encoding the target variable.
4. **Data Splitting**: Dividing the dataset into training and testing sets, reshaped to fit DL models like CNNs.

### Feature Identification and Categorization

Features are categorized into three groups:

1. **Connection Information**: Basic details like protocol type and service accessed.
2. **Connection Content**: Deeper insights such as failed login attempts.
3. **Traffic Information**: Patterns and trends in network traffic, crucial for detecting distributed attacks.

### Model Development

#### Machine Learning Models

Evaluated models include:

- **Decision Trees**
- **Random Forest**
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Logistic Regression**

#### Deep Learning Models

Implemented models:

- **Convolutional Neural Networks (CNN)**
- *Future Work*: **Recurrent Neural Networks (RNN)**, **LSTM**, **Autoencoders**, **GANs**

### Model Evaluation

#### Binary Classification

For binary classification (Normal vs. Attack):

- **Confusion Matrix**: Analyzes true positives/negatives and false positives/negatives.
- **Performance Metrics**:
  - Training Accuracy: 99.62%
  - Testing Accuracy: 99.45%
  - Precision, Recall, and F1 Scores indicate a well-balanced model.

  ![Binary Classification Output](https://github.com/Mugeshcybersecurity/Network-Intrusion-Detection-System-Using-Machine-Learning/blob/main/Dataset/nsl-kdd/output.png)

#### Multiclass Classification

For multiclass classification (Normal, DoS, R2L, Probe, U2R):

- **Confusion Matrix**: Evaluates model performance across multiple attack categories.
- **Performance Metrics**:
  - Similar high performance in precision, recall, and F1 scores.

  ![Multiclass Classification Output](https://github.com/Mugeshcybersecurity/Network-Intrusion-Detection-System-Using-Machine-Learning/blob/main/Dataset/nsl-kdd/output2.png)

  **Classification Report**

  | Precision | Recall | F1-Score | Support |
  |-----------|--------|----------|---------|
  | normal    | 0.89   | 0.80     | 0.85    | 7460    |
  | Dos       | 0.81   | 0.55     | 0.65    | 2421    |
  | Probe     | 0.96   | 0.09     | 0.17    | 2885    |
  | R2L       | 0.88   | 0.21     | 0.34    | 67      |
  | U2R       | 0.66   | 0.94     | 0.78    | 9711    |

### Optimization and Hyperparameter Tuning

- **Randomized Search**: Used for optimizing CNN parameters, achieving a best score of 99.61% on validation sets.
- **Optimized Hyperparameters**:
  - Neurons: 64 (1st layer), 64 (2nd layer)
  - Learning Rate: 0.001
  - Batch Size: 64
  - Epochs: 20

## Results and Discussion

The project has demonstrated promising results in both binary and multiclass classification tasks for network intrusion detection. The binary classification model achieved a training accuracy of 99.62% and a testing accuracy of 99.45%, indicating excellent performance in distinguishing between normal and attack traffic. Additionally, the multiclass classification model showed strong performance, with high precision, recall, and F1-scores across the different attack categories.

### Challenges and Limitations

- **Data Bias**: Even in improved datasets, biases can still affect model generalization.
- **Complexity**: Deep learning models require significant computational resources.

### Conclusion

This project demonstrates the effectiveness of combining ML and DL techniques in developing a robust IDS. While models like CNN show promising results, ongoing work in implementing advanced DL models could further enhance performance, making the IDS more adaptive to evolving cyber threats.
