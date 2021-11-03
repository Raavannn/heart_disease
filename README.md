# How to Deploy?
> Create a virtual environment with python 3.8.
> Install all the libraries using pip install -f requirements.txt
> Activate Virtual Environment.
> Run streamlit run app.py

# Architecture
> app.py is the application file, the user always needs to run this file.
> Other python files will be called by app.py
> The application uses heart.csv. If at any time the user wants to change the file, move your new dataset file here and replace it with heart.csv
Note: The plots & accuracy for the predictor can vary with new dataset.