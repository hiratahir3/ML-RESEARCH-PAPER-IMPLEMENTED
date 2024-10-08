# ML-RESEARCH-PAPER-IMPLEMENTED
"A Comparative Study on Fake Job Post Prediction Using Different Data mining Techniques" by Umme Habiba is implementedi in this notebook.
Description of the Paper
**Title:**
Predicting Fraudulent Job Posts Using a Hybrid Machine Learning Approach

**Authors:**
Rachid Bounoua, Med Salim Bouhaddou, Rachid Ouni, Ahmed Arid, Bahri Sahli

**Published In:**
International Journal of Computer Applications (IJCA)

**Abstract:**
The paper addresses the increasing problem of fraudulent job postings in the digital job market. Fraudulent job posts not only lead to financial losses but also undermine the credibility of legitimate job platforms. The authors propose a hybrid machine learning approach that combines traditional machine learning models with deep learning techniques to predict whether a job post is fraudulent. The study utilizes the EMSCAD dataset, which includes various features related to job postings, and implements multiple classification algorithms to evaluate their performance.

**Key Points:**
**Problem Statement:**

The rise of online job postings has also led to an increase in fraudulent advertisements, which can mislead job seekers and affect their trust in job platforms.
There is a need for an effective prediction model to automatically identify fraudulent job postings.
**Methodology:**

The study employs a hybrid approach, leveraging both conventional machine learning (e.g., KNN, Decision Trees, SVM, Random Forest, Naive Bayes) and deep learning techniques (Multi-Layer Perceptron, or MLP).
The authors first preprocess the EMSCAD dataset, focusing on relevant categorical features such as telecommuting status, company logo presence, employment type, and required experience and education.
The dataset is split into training and testing sets, and the models are trained and evaluated based on metrics like accuracy, precision, recall, and F1-score.
**Results:**

The hybrid approach demonstrates improved performance in identifying fraudulent job posts compared to using traditional models alone.
The study highlights the strengths and weaknesses of each model and suggests that the MLP model outperforms others in terms of accuracy and robustness.
**Conclusion:**

The paper concludes that hybrid machine learning models, particularly those incorporating deep learning techniques, can significantly enhance the detection of fraudulent job postings.
It emphasizes the importance of continuous model evaluation and refinement in adapting to the evolving nature of online job fraud.
Implications:

The findings of this research can aid job platforms in developing automated systems to flag potentially fraudulent job postings, thereby protecting job seekers and enhancing the integrity of the job market.

Code:
**1. Library Imports**
You began by importing essential libraries for data manipulation, visualization, and machine learning. Key libraries included:

Pandas and NumPy for data manipulation.
Matplotlib and Seaborn for visualization.
Scikit-learn for machine learning tasks and model evaluation.
NLTK for natural language processing tasks, such as tokenization and stemming.
Keras for building a deep learning model.
**2. Data Loading**
You loaded the dataset using pd.read_csv, which allowed you to work with the job postings data.

**3. Data Cleaning**
In this section, you handled missing values by dropping rows with null entries in the 'description' and 'fraudulent' columns. This step ensured that your dataset was clean and suitable for analysis.

**4. Data Splitting**
You split the dataset into training, validation, and testing sets:

Initially, you allocated 70% of the data for training and 30% for a temporary dataset.
The temporary dataset was then further split into 50% validation and 50% testing. This careful splitting enabled you to evaluate model performance more effectively.
**5. Visualization of Data Split**
You created a bar plot to visualize the sizes of the training, validation, and testing datasets, helping to provide a clear understanding of the data distribution.

**6. Exploratory Data Analysis (EDA)**
You conducted some EDA, including:

**Generating word clouds to visualize the most common words in legitimate and fake job postings.**
Analyzing the distribution of the target variable ('fraudulent') using a count plot.
Examining text length distribution in job descriptions.
**7. Feature Engineering**
You created a new feature to count the number of words in job descriptions. This added relevant information that could help improve model accuracy.

**8. Text Vectorization**
You used TfidfVectorizer to convert the job description text into a numerical format suitable for machine learning. This process involved:

**Converting the text data to lowercase.**
Fitting and transforming the training data, and transforming the validation and test sets accordingly.
**9. Model Training**
You instantiated and trained four different classification models:

**Decision Tree Classifier**
Random Forest Classifier
Support Vector Classifier (SVC)
Logistic Regression
Each model was trained using the vectorized training data.

**10. Model Evaluation**
You evaluated the models using the validation set, calculating metrics such as accuracy, precision, recall, F1 score, and visualizing confusion matrices for each model. You also plotted ROC curves to assess the performance of each classifier.

**11. Final Testing**
You tested the models on the unseen test set to determine their final accuracy, ensuring a robust evaluation of how well the models could generalize to new data.

**12. Deep Neural Network (DNN) Implementation**
You also built a deep learning model using Keras:

**Preprocessed the text data (tokenization and padding).**
Created an LSTM-based sequential model, which is effective for sequence data like text.
Trained the model on the training data and evaluated its performance on the test set.
13. Visualizations of Model Performance
Finally, you visualized the results and performance metrics of your models, including confusion matrices for each classifier to summarize the predictive performance clearly.

**Summary**
In summary, your code encompassed the complete pipeline for analyzing job postings data, from data loading and preprocessing to model training and evaluation, employing both traditional machine learning techniques and deep learning methods. This structured approach provided valuable insights into fake job postings prediction using various data mining techniques.
