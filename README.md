# Flood-Prediction-System
Flood Prediction System
Project Overview

This project applies deep learning techniques using TensorFlow and Keras to analyze multiple environmental factors and estimate flood risk levels. By processing historical and geographical data, the model learns patterns associated with flood occurrences and generates probability-based predictions for future flood events.

The workflow includes data preprocessing, feature scaling, exploratory data analysis, neural network development, model training, and performance evaluation. Various environmental indicators such as rainfall intensity, drainage efficiency, urbanization, climate conditions, and infrastructure quality are used as input features to improve prediction accuracy.

The objective of this project is to demonstrate how artificial intelligence can support disaster management by providing data-driven flood risk assessments. The resulting model can assist researchers, planners, and decision-makers in identifying vulnerable regions and improving preparedness strategies.

Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib

Key Features

- Data cleaning and preprocessing
- Feature scaling and normalization
- Deep neural network implementation
- Model training and validation
- Flood probability prediction
- Performance evaluation and analysis

Future Scope

- Integration of real-time weather data
- Hyperparameter optimization for improved accuracy
- Deployment as a web-based prediction system
- Expansion with additional environmental datasets

How to run this code
This project has 3 csv files which need to be downloaded and one main.py file which has the python code to run the project. 

The project was changed to adapt Jupyter Notebook on Pycharm editor using Tensorflow Keras environment. So the code is directly compatible to Jupyter Notebook too. 

Only thing left is to set up tensorflow environments. 
If its Jupyter Notebook run this-

python -m venv tf_env
tf_env\Scripts\activate (windows)
source tf_env/bin/activate (Linux or mac)

pip install tensorflow jupyter notebook ipykernel pandas numpy matplotlib scikit-learn

python -c "import tensorflow as tf; print(tf.__version__)"

python -m ipykernel install --user --name=tf_env --display-name="TensorFlow Environment"

jupyter notebook

For Pycharm follow same steps except the last two instead use this-
tf_env\Scripts\python.exe

Now you have your environment and now you can execute this code in your environment.