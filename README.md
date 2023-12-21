# Deep learning and Machine learning for network threat detection

I use the fast.ai deep learning framework for one of its newest applications: classification on tabular data. I compare its performance against the incumbent best tool in the field, gradient boosting with XGBoost, as well as against various scikit-learn classifiers in detecting network intrusion traffic and classifying common network attack types (e.g., FTP-BruteForce, DOS-GoldenEye, BruteForce-XSS, SQL-Injection, Infiltration, BotAttack). 

In line with recent prominence on other tabular datasets, fast.ai is on par with XGBoost and sklearn’s Random Forest Classifier, demonstrating high accuracy across datasets and network attack types, with low false positive and negative rates in the classification of various intrusion types. Pretty powerful!

![](<./plots/box plots/result02032018_plt.png>)

![](<./plots/confusion_matrices/result16022018_cm.png>)

## Background

Recent advancements in deep learning algorithms have facilitated significant strides in addressing challenging computer science problems and applications in nearly all areas of life. These breakthroughs have extended to areas such as computer vision, natural language processing, complex reasoning tasks like playing board games (e.g., Go, Chess), and even surpassing human champions. 

In light of the ongoing surge in cyber-attacks and the increased demand for AI usage in the context of cybersecurity [MIT Report](https://wp.technologyreview.com/wp-content/uploads/2022/07/Deep-Learning-Delivers-proactive-Cyber-defense-FNL.pdf), in this project, I investigate the effectiveness and capacity of a powerful new deep learning algorithm, fast ai, in the domain of network intrusion detection and compare its performance against the incumbent best tool in the field, gradient boosting with XGBoost, as well as against various scikit-learn classifiers (random forest, knn, naïve bayes, etc.). 

In a previous study, [Basnet and colleagues (2018)]( https://isyou.info/jisis/vol9/no4/jisis-2019-vol9-no4-01.pdf) have shown that the fastai deep learning algorithm provided the highest accuracy of about 99% compared to other well-known deep learning frameworks (e.g., Keras, TensorFlow, Theano) in detecting network intrusion traffic and classifying common network attack types using the [CSE-CIC-IDS2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) (same dataset as I used here). 

Deep learning is the gold standard for large, unstructured datasets, including text, images, and video and has been battle tested in areas such as computer vision, natural language processing, and complex reasoning tasks. However, for one specific type of dataset –one of the most common datasets used in cybersecurity– deep learning typically falls behind other, more “shallow-learning” approaches such as decision tree algorithms (random forests, gradient boosted decision trees): TABULAR DATA. 

Indeed, in a [systematic review and meta-analysis](https://arxiv.org/abs/2207.08815) last year, Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux have shown that, overall, tree-based models (random forests and XGBoost) outperform deep learning methods for tabular data on medium-sized datasets (10k training examples). However, the gap between tree-based models and deep learning becomes narrower as the dataset size increases (here: 10k -> 50k).

Here, I extend these lines of investigation, by comparing fast ai’s deep learning framework with XGBoost as well as other scikit-learn classifiers on a relatively large dataset of network traffic data.


## Dataset

-   Downloaded from: https://www.unb.ca/cic/datasets/ids-2018.html
-   contains: 7 csv preprocessed and labelled files, top feature selected files, original traffic data in pcap format and logs
-   used csv preprocessed and labelled files for this research project

## Data Cleanup

I am using the ***data_cleanup.py*** script from the [Basnet and colleagues project](https://github.com/Colorado-Mesa-University-Cybersecurity/DeepLearning-IDS/tree/master) to perform data wrangling. 
-   dropped rows with Infinitiy values
-   some files had repeated headers; dropped those
-   converted timestamp value that was date time format: 15-2-2018 to UNIX epoch since 1/1/1970
-   separated data based on attack types for each data file
-   ~20K rows were removed as a part of data cleanup
-   see data_cleanup.py script for this phase
-   \# Samples in table below are total samples left in each dataset after dropping # Dropped rows/samples

## Dataset Summary

**Table 1: Number of samples and network traffic types in each dataset**

| File Name      | Traffic Type     | # Samples | # Dropped |
| -------------- | ---------------- | --------: | :-------- |
| 02-14-2018.csv | Benign           |   663,808 | 3818      |
|                | FTP-BruteForce   |   193,354 | 6         |
|                | SSH-Bruteforce   |   187,589 | 0         |
| 02-15-2018.csv | Benign           |   988,050 | 8027      |
|                | DOS-GoldenEye    |    41,508 | 0         |
|                | DOS-Slowloris    |    10,990 | 0         |
| 02-16-2018.csv | Benign           |   446,772 | 0         |
|                | Dos-SlowHTTPTest |   139,890 | 0         |
|                | DoS-Hulk         |   461,912 | 0         |
| 02-22-2018.csv | Benign           | 1,042,603 | 5610      |
|                | BruteForce-Web   |       249 | 0         |
|                | BruteForce-XSS   |        79 | 0         |
|                | SQL-Injection    |        34 | 0         |
| 02-23-2018.csv | Benign           | 1,042,301 | 5708      |
|                | BruteForce-Web   |       362 | 0         |
|                | BruteForce-XSS   |       151 | 0         |
|                | SQL-Injection    |        53 | 0         |
| 03-01-2018.csv | Benign           |   235,778 | 2259      |
|                | Infiltration     |    92,403 | 660       |
| 03-02-2018.csv | Benign           |   758,334 | 4050      |
|                | BotAttack        |   286,191 | 0         |

**Table 2: Total number of traffic data samples for each type among all the datasets**

| Traffic Type     | # Samples |
| ---------------- | --------: |
| Benign           | 5,177,646 |
| FTP-BruteForce   |   193,354 |
| SSH-BruteForce   |   187,589 |
| DOS-GoldenEye    |    41,508 |
| Dos-Slowloris    |    10,990 |
| Dos-SlowHTTPTest |   139,890 |
| Dos-Hulk         |   461,912 |
| BruteForce-Web   |       611 |
| BruteForce-XSS   |       230 |
| SQL-Injection    |        87 |
| Infiltration     |    92,403 |
| BotAttack        |   286,191 |
| Total Attack     | 1,414,765 |

## Deep Learning and Machine Learning Frameworks

-   perfomance results using the fast au deep learning framework is compared various machine learning algorithms from the scikit-learn library
-   10-fold cross-validation techniques was used to validate the model

### FastAI

-   https://www.fast.ai/
-   uses PyTorch, https://pytorch.org/ as the backend

### Scikit-Learn

-   LogisticRegression
-   LinearDiscriminantAnalysis
-   KNN
-   DecisionTreeClassifier
-   GaussianNB
-   BernoulliNB
-   RandomForestClassifier
-   XGBClassifier

## Experiment Results

#### Using colab.research.google.com/

#### Boxplots Accuracy Scores

**Table 3: Boxplots Accuracy Comparison for each dataset**

|                          02-14-2018                          |                             02-15-2018                             |                            02-16-2018                             |
| :----------------------------------------------------------: | :----------------------------------------------------------------: | :---------------------------------------------------------------: |
| ![](<./plots/box plots/result14022018_plt.png>) |    ![](<./plots/box plots/result15022018_plt.png>)    |   ![](<./plots/box plots/result16022018_plt.png>)    |
|                          02-22-2018                          |                             02-23-2018                             |                            03-01-2018                             |
| ![](<./plots/box plots/result22022018_plt.png>) |    ![](<./plots/box plots/result23022018_plt.png>)    |   ![](<./plots/box plots/result01032018_plt.png>)    |
|                          03-02-2018                          | 
| ![](<./plots/box plots/result02032018_plt.png>) | 

**Table 4:  Accuracy Comparison for each dataset**
| Dataset     | Framework         | Accuracy (%) | Std-Dev |
| ----------- | ----------------- | -----------: | ------: | 
| 02-14-2018  | FastAI            |        99.99 |    0.05 |
|             | LogisticRegression            |        99.99 |      0.00 |
|             | LDA      |           98.71 |      0.01 |
|             | KNN      |           99.99 |      0.00 |
|             | DecisionTreeClassifier     |           99.99 |      0.00 |
|             | GaussianNB     |           99.97 |      0.00 |
|             | BernoulliNB     |           99.88 |      0.01 |
|             | RandomForest     |            99.99 |      0.00 |
|             | XGBoost     |           99.99 |      0.00 |
| 02-15-2018  | FastAI            |        99.99 |    0.01 |            
|             | LogisticRegression            |        99.97 |      0.00 |
|             | LDA      |           98.15 |      0.05 |
|             | KNN      |           99.99 |      0.00 |
|             | DecisionTreeClassifier     |           99.99 |      0.00 |
|             | GaussianNB     |           96.44 |      0.09 |
|             | BernoulliNB     |           94.58 |      0.08 |
|             | RandomForest     |            99.99 |      0.00 |
|             | XGBoost     |           99.99 |      0.00 |
| 02-16-2018  | FastAI            |       99.00 |    0.00 |             
|             | LogisticRegression            |        99.99 |      0.00 |
|             | LDA      |           99.83 |      0.01 |
|             | KNN      |           99.99 |      0.00 |
|             | DecisionTreeClassifier     |           100.00 |      0.00 |
|             | GaussianNB     |           99.75 |      0.01 |
|             | BernoulliNB     |           98.41 |      0.03 |
|             | RandomForest     |           100.00 |      0.00 |
|             | XGBoost     |           99.99 |      0.00 |
| 02-22-2018  | FastAI            |        99.99 |    0.12 |             
|             | LogisticRegression            |        99.98 |      0.00 |
|             | LDA      |           99.77 |      0.04 |
|             | KNN      |           99.99 |      0.00 |
|             | DecisionTreeClassifier     |           99.99 |      0.00 |
|             | GaussianNB     |           83.28 |      0.12 |
|             | BernoulliNB     |           90.13 |      1.73 |
|             | RandomForest     |           99.98 |      0.00 |
|             | XGBoost     |           99.99 |      0.00 |
| 02-23-2018  | FastAI            |        99.99 |    0.00 |             
|             | LogisticRegression            |        99.96 |      0.01 |
|             | LDA      |           99.72 |      0.03 |
|             | KNN      |           99.99 |      0.00 |
|             | DecisionTreeClassifier     |           99.99 |      0.00 |
|             | GaussianNB     |           83.89 |      0.16 |
|             | BernoulliNB     |           90.53 |      0.15 |
|             | RandomForest     |           99.98 |      0.00 |
|             | XGBoost     |           99.99 |      0.00 |
| 03-01-2018  | FastAI            |        87.70 |    0.07 |             
|             | LogisticRegression            |        74.22 |      1.37 |
|             | LDA      |           75.55 |      0.27 |
|             | KNN      |           85.26 |      0.20 |
|             | DecisionTreeClassifier     |           92.59 |      0.13 |
|             | GaussianNB     |           41.58 |      2.94 |
|             | BernoulliNB     |           50.89 |      0.65 |
|             | RandomForest     |           91.00 |      0.17 |
|             | XGBoost     |           94.25 |      0.19 |
| 03-02-2018  | FastAI            |        99.99 |     0.01 |               75 |
|             | LogisticRegression            |        97.04 |      1.55 |
|             | LDA      |           94.41 |      0.08 |
|             | KNN      |           99.99 |      0.00 |
|             | DecisionTreeClassifier     |           99.99 |      0.00 |
|             | GaussianNB     |           83.98 |      0.13 |
|             | BernoulliNB     |           94.11 |      0.08 |
|             | RandomForest     |           99.99 |      0.00 |
|             | XGBoost     |           99.99 |      0.00 |
| ===         | ===               |          === |     === |




#### Confusion Matrices

**Table 5:  Confusion matrices for each dataset**
|                          02-14-2018                          |                             02-15-2018                             |                            02-16-2018                             |
| :----------------------------------------------------------: | :----------------------------------------------------------------: | :---------------------------------------------------------------: |
| ![](<./plots/confusion_matrices/result14022018_cm.png>) |    ![](<./plots/confusion_matrices/result15022018_cm.png>)    |   ![](<./plots/confusion_matrices/result16022018_cm.png>)    |
|                          02-22-2018                          |                             02-23-2018                             |                            03-01-2018                             |
| ![](<./plots/confusion_matrices/result22022018_cm.png>) |    ![](<./plots/confusion_matrices/result23022018_cm.png>)    |   ![](<./plots/confusion_matrices/result01032018_cm.png>)    |
|                          03-02-2018                          | 
| ![](<./plots/confusion_matrices/result02032018_cm.png>) | 


