
# Heart Disease Prediction Project

This project involves building a machine learning model to predict the presence or absence of heart disease using a dataset sourced from Kaggle. The goal is to create a robust and accurate prediction model using various machine learning techniques.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Steps to Run the Project with Docker](#steps-to-run-the-project-with-docker)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Preprocessed and cleaned heart disease dataset.
- Exploratory Data Analysis (EDA) to understand the dataset.
- Implementation of multiple machine learning models (For this release I used Logistic Regression / Binary Classificattion).
- Model evaluation and hyperparameter tuning.
- Dockerized setup for consistent development and deployment.

---

## Dataset

The dataset used for this project is sourced from Kaggle. It contains information about various medical attributes such as age, sex, cholesterol levels, and more, which are used to predict the presence of heart disease.

**[Dataset Link](https://www.kaggle.com/datasets/kapoorprakhar/cardio-health-risk-assessment-dataset)**

---

## Technologies Used

- Python (with libraries such as pandas, numpy, matplotlib, scikit-learn, etc.)
- Jupyter Notebook
- Docker
- Flask (for deployment)

---

## Getting Started

To get a local copy up and running, follow these steps:

### Prerequisites
- Install [Docker](https://www.docker.com/get-started)
- Download the dataset from Kaggle and place it in the `data/` folder(Optional).

---

## Steps to Run the Project with Docker

1. **Clone the Repository**
   ```bash
   git clone https://github.com/zerokodes/Heart_Disease_Prediction.git
   cd heart-disease-prediction
   ```

2. **Prepare the Environment**
   Ensure the dataset is placed in the `data/` folder of the project(Optional).

3. **Build the Docker Image**
   Use the provided `Dockerfile` to build the Docker image:
   ```bash
   docker build -t heart-disease-prediction .
   ```

4. **Run the Docker Container**
   Start the container to run the project:
   ```bash
   docker run -p 9696:9696 heart-disease-prediction
   ```

5. **Access the Application**
   Open your browser and navigate to:
   ```
   http://localhost:9696
   ```

6. **Perform Predictions**
   - input patient data in the prediction-test.py file and predict the presence of heart disease.
   - In subsequent release, you can interact with the API endpoint (e.g., using Postman) and also the use of web interface .

---

## Project Structure

```
heart-disease-prediction/
│
├── data/                   # Folder for dataset
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for analysis
├── app/                    
│   ├── static/             # Static files (CSS, JS, etc.)
│   ├── templates/          # HTML templates for Flask
│   └── app.py              # Flask app for deployment
│
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

---

## Contributing

Contributions are welcome! If you want to contribute to this project, please fork the repository and create a pull request with detailed information about the changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
