# Customer Churn Prediction with PySpark

## Overview
This project aims to predict customer churn using PySpark, leveraging big data processing capabilities to handle and analyze large datasets. The goal is to develop and evaluate machine learning models to identify customers who are likely to leave a service, enabling proactive retention strategies.

## Motivation
Any organization that deals with big data and data warehousing needs some form of distributed system. Among the most widely used distributed systems, Apache Spark stands out due to its capability to handle several petabytes of data across thousands of cooperating servers, whether physical or virtual. Its simplicity and speed make it particularly appealing.

Data professionals can greatly benefit from learning the logistics and practical applications of Spark. To facilitate this, I have created a repository showcasing various examples of PySpark functions and utilities that can be used to build complete ETL processes for data modeling. Spark also offers a Python API, making it easy to manage data with Python, especially using Jupyter notebooks.

This repository is geared towards individuals who are already familiar with Python and have some background in data analytics, although I often skip the environment setup. However, if you follow the installation section, you should be able to work through the notebooks without any major issues. PySpark's integration with Jupyter Notebooks, coupled with its pre-built functions, makes data processing efficient and accessible. The goal of this repository is to help you get up and running with PySpark quickly.

## Project Structure
- `data/`: Directory containing the dataset.
- `notebooks/`: Jupyter notebooks for analysis and modeling.
- `models/`: Saved models and related artifacts.
  
## Dataset
The dataset utilized in this project provides comprehensive information about the customers of a telecommunications company, covering aspects such as duration of service, call patterns, and churn behavior. Here is a detailed overview of the dataset:

* Column Count: 20 columns in total.
 
* Data Types: A mixture of continuous, discrete, and categorical data.

* Target Variable: The target variable for prediction is 'Churn'.

* Key Features: Certain attributes like the International plan, Total day charge, and Customer service calls are anticipated to significantly influence the likelihood of customer churn.

The dataset, stored in the data/ directory, encompasses features such as customer demographics, usage patterns, and service subscription details. The target variable, churn, indicates whether a customer has discontinued the service.

## Prerequisites
* PySpark
* Jupyter Notebook

## Usage 
- Preprocess data
- Train the model
- Evaluate the model 
## Future Work
+ Fine-tune hyperparameters
+ Experiment with additional algorithms
+ Implement real-time monitoring for early

## Resources
[Report](report/report.pdf)

[View PowerPoint Presentation](https://www.canva.com/design/DAGYYjXLDq8/eWAaYeecaHkId6K_gUjoeQ/edit)

[Source Code]('notebooks/ProjectBigData_CustomerChurnPrediction.ipynb')
