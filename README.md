PySpark Arabic News Classification 

This project is an end-to-end text classification pipeline built using PySpark in a Jupyter Notebook environment. The goal is to process and categorize a large corpus of Arabic news articles stored in a MinIO (S3-compatible) object storage server.

This project demonstrates the ability to build and manage a complete Machine Learning pipeline on a distributed computing framework (Apache Spark), handling all steps from data ingestion to model training at scale.

🚀 Key Features

End-to-End Pipeline: A single, runnable pipeline that handles data ingestion, preprocessing, feature extraction, model training, and evaluation.

Scalable NLP Processing: Built entirely on PySpark to handle datasets that are too large to fit into the memory of a single machine.

Arabic Text Preprocessing: Includes custom logic for processing Arabic text (e.g., normalization, stop-word removal, tokenization).

ML with PySpark MLlib: Uses PySpark MLlib's Pipeline API to implement and train classification models (e.g., Logistic Regression, Naive Bayes) using TF-IDF features.

S3-Compatible Storage: Seamlessly integrates with MinIO (or any S3-compatible service) for reading raw data and saving the final trained model.

🔄 Data & ML Pipeline Workflow

The PySpark.ml.Pipeline is structured as follows:

Data Ingestion: Load raw news articles (text) and their categories (labels) from MinIO into a Spark DataFrame.

Text Preprocessing (Custom Transformer): A custom transformer cleans and normalizes the Arabic text.

Tokenization: Splits the text into individual words.

Stop-Word Removal: Removes common Arabic stop-words.

Feature Extraction (TF-IDF): Converts the processed text into numerical TF-IDF vectors.

Model Training: Feeds the feature vectors into a classification algorithm (e.g., LogisticRegression).

Prediction & Evaluation: The trained model is used to make predictions, and its performance is measured (Accuracy, F1-Score).

Model Saving: The entire trained pipeline (including preprocessing steps) is saved back to MinIO for future use.

🛠️ Technologies Used

Apache Spark:

PySpark: The core Python API for Spark.

Spark SQL: For DataFrame manipulation.

PySpark MLlib: For the machine learning pipeline.

MinIO: As an S3-compatible object storage for data and models.

Python 3.x

NLTK (or similar): For Arabic NLP utilities (e.g., stop-words list).

⚙️ How to Run

1. Clone the repository:

git clone [https://github.com/Djilit-Abdellahi/pyspark-arabic-news-classification.git](https://github.com/Djilit-Abdellahi/pyspark-arabic-news-classification.git)
cd pyspark-arabic-news-classification


2. Set up your environment:
This project is a Jupyter Notebook (.ipynb) and requires an environment with PySpark installed.

Local: You can use a local Jupyter Notebook server with pyspark installed.

Cloud: You can upload this notebook to platforms like Google Colab, Databricks, or Google Cloud Vertex AI Notebooks.

3. Install dependencies:

pip install pyspark pandas numpy
# or
pip install -r requirements.txt 


(Make sure to create a requirements.txt file with these packages)

4. Configure MinIO/S3 Connection:
Before running the notebook, you must update the Spark Session configuration cell with your own MinIO/S3 credentials:

# (Inside the notebook cell)
spark = SparkSession.builder \
    .appName("NewsClassification") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://your-minio-server:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "YOUR-ACCESS-KEY") \
    .config("spark.hadoop.fs.s3a.secret.key", "YOUR-SECRET-KEY") \
    .config("spark.hadoop.fs.s3a.path.style.access", True) \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()


5. Run the Notebook:
Open the pyspark_arabic_news_classification.ipynb file in your chosen Jupyter environment and run the cells from top to bottom.
