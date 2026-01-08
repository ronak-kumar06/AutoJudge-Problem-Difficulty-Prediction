
# AutoJudge: Programming Problem Difficulty Prediction

AutoJudge is a machine learning–based system that automatically predicts the difficulty of programming problems using only their textual descriptions.
The system performs two tasks:

- **Classification**: Predicts the difficulty class (Easy / Medium / Hard)
- **Regression**: Predicts a relative numerical difficulty score


The predictions are based on the problem’s title, description, input description, and output description.
A simple Flask web interface allows users to paste a new problem statement and get instant predictions along with confidence and explanation.


## Dataset Used

The dataset consists of competitive programming problems collected from online platforms.

Each data sample contains:

- title

- description

- input_description

- output_description

- sample_io

- problem_class (Easy / Medium / Hard)

- problem_score (numerical difficulty score)

- https://github.com/AREEG94FAHAD/TaskComplexityEval-24

The dataset is provided in JSONL format and already contains labeled difficulty classes and scores.
No manual labeling was performed.

## Approach

### Data Preprocessing

Combined all textual fields into a single representation

Handled missing values by replacing them with empty strings

Ensured consistent text formatting

### Feature Engineering

#### Two types of features were used:

    1. Textual Features

-TF-IDF vectors extracted from combined problem text

    2. Handcrafted Features

-Length of the text

-Count of numbers and mathematical symbols

-Frequency of algorithmic keywords (e.g., graph, dfs, bfs, dp, recursion)

#### Models Used

    1. Classification Model

-Random Forest Classifier

-Predicts: Easy / Medium / Hard

    2. Regression Model

-Random Forest Regressor

-Predicts: Relative difficulty score

- Both models are trained using the same feature pipeline.

## Evaluation Metrics

    1. Classification

-Accuracy

-Confusion Matrix

    2.Regression

-Mean Absolute Error (MAE)

-Root Mean Squared Error (RMSE)

- Difficulty prediction is inherently subjective; therefore, results are interpreted as relative rather than absolute difficulty.

## Run Locally

Clone the project

```bash
  git clone https://github.com/ronak-kumar06/AutoJudge-Problem-Difficulty-Prediction.git
```

Go to the project directory

```bash
  cd AutoJudge-Problem-Difficulty-Prediction
```

Create and activate virtual environment

- Windows
```bash
  python -m venv .venv
  .venv\Scripts\activate
```

- Linux / macOS
```bash
  python3 -m venv .venv
  source .venv/bin/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Train the models

```bash
  python train_models.py
```

Run the web application

```bash
  python app_web.py
```

Open in browser

```bash
  http://127.0.0.1:5000
```
## Web Interface

#### A simple Flask-based web interface is provided where users can:

    1. Enter problem title and descriptions

    2. Click Predict

    3. View:

        - Predicted difficulty class

        - Relative difficulty score

        - Prediction confidence

        - Explanation of prediction (XAI-lite)

- The UI is designed for local execution and demonstration.
## Explainable Difficulty Prediction (Unique Feature)

To improve transparency, the system provides lightweight explainability for each prediction.

#### The explanation highlights:

- Detected algorithmic keywords (e.g., graph, bfs, dp)

- Text length and complexity

- Presence of numeric constraints

This helps users understand why a particular difficulty was predicted, without using heavy deep-learning explainability tools.
## Project Structure

```
AutoJudge-Problem-Difficulty-Prediction/
│
├── train_models.py          # Model training script
├── feature_utils.py         # Feature extraction & explainability
├── app_web.py               # Flask web application
├── models/                  # Saved trained models
├── templates/               # HTML templates
├── static/                  # CSS files
├── requirements.txt
└── README.md
```

## Production Readiness

The application uses Flask’s development server for local execution.
For production deployment, it can be served using a production WSGI server such as Gunicorn, with debug mode disabled, without changing application logic.

## Project Report

The detailed project report (PDF) is available here:

[report/report.pdf](report/report.pdf)


## Demo Video Link

https://youtu.be/zr92Ds8q6Mo


## Author

```
Name: Ronak Kumar
Enrollement No: 23113130
Department: Civil Engineering (3rd year)
GitHub: ronak-kumar06
```