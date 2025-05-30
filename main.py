from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask_cors import CORS
from sklearn import datasets as sklearn_datasets

app = Flask(__name__)
CORS(app)

# Dataset Config
datasets = {
    "Iris Dataset": "iris",
    "California Housing": "california_housing",
    "Diabetes": "diabetes",
    "Wine Quality": "wine",
    "Titanic": "titanic"
}

# Algorithms Config
algorithms = {
    "Random Forest Regressor": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "SVM": SVR(),
    "LightGBM": lgb.LGBMRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": xgb.XGBRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso()
}

# Dataset Descriptions
dataset_info = {
    "Iris Dataset": "Classification of iris flowers into three species.",
    "California Housing": "Predicts median house values based on census data.",
    "Diabetes": "Predicts diabetes progression based on medical data.",
    "Wine Quality": "Predicts wine quality based on physicochemical tests.",
    "Titanic": "Passenger survival data from the Titanic disaster."
}

@app.route('/get_dataset_info', methods=['GET'])
def get_dataset_info():
    return jsonify(dataset_info)

# Load dataset
def load_dataset(dataset_name):
    if dataset_name == "iris":
        data = sklearn_datasets.load_iris(as_frame=True)
        df = data.frame
    elif dataset_name == "california_housing":
        data = sklearn_datasets.fetch_california_housing(as_frame=True)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "diabetes":
        data = sklearn_datasets.load_diabetes(as_frame=True)
        df = data.frame
    elif dataset_name == "wine":
        data = sklearn_datasets.load_wine(as_frame=True)
        df = data.frame
    elif dataset_name == "titanic":
        df = sns.load_dataset("titanic").dropna()
    else:
        return None
    return df

@app.route('/get_datasets', methods=['GET'])
def get_datasets():
    return jsonify({"datasets": list(datasets.keys())})

@app.route('/get_algorithms', methods=['GET'])
def get_algorithms():
    return jsonify({"algorithms": list(algorithms.keys())})

@app.route('/dataset_sample', methods=['GET'])
def dataset_sample():
    dataset_name = request.args.get('dataset', '')
    df = load_dataset(datasets.get(dataset_name))
    if df is not None:
        return jsonify({"sample": df.head().to_dict(orient='records')})
    return jsonify({"error": "Dataset not found"}), 404

@app.route('/compare', methods=['POST'])
def compare_algorithms():
    try:
        data = request.json
        dataset_name = datasets.get(data.get('dataset'))
        algo1_name = data.get('algorithm1')
        algo2_name = data.get('algorithm2')

        if dataset_name not in datasets.values() or algo1_name not in algorithms or algo2_name not in algorithms:
            return jsonify({"error": "Invalid dataset or algorithm"}), 400

        df = load_dataset(dataset_name)
        if df is None:
            return jsonify({"error": "Dataset not found"}), 404

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model1 = algorithms[algo1_name]
        model2 = algorithms[algo2_name]

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)

        def metrics(y_true, y_pred):
            return mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)

        mae1, mse1, r2_1 = metrics(y_test, pred1)
        mae2, mse2, r2_2 = metrics(y_test, pred2)

        best_algo = algo1_name if mse1 < mse2 else algo2_name

        plt.figure(figsize=(6, 4))
        sns.barplot(x=[algo1_name, algo2_name], y=[mae1, mae2], palette="coolwarm")
        plt.title("Algorithm Performance Comparison")
        plt.ylabel("Mean Absolute Error")
        plt.xlabel("Algorithms")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        result = {
            "best_algorithm": best_algo,
            "metrics": {
                "MSE1": mse1,
                "MSE2": mse2,
                "MAE1": mae1,
                "MAE2": mae2,
                "R2_1": r2_1,
                "R2_2": r2_2
            },
            "comparison_graph": graph_base64
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tutorials', methods=['GET'])
def get_tutorials():
    tutorials = [
        {
            "title": "Clustering Algorithms",
            "description": "Understand how data can be grouped using K-Means and DBSCAN.",
            "content": "Clustering algorithms aim to group data points with similar characteristics. K-Means minimizes intra-cluster variance. DBSCAN detects dense regions..."
        },
        {
            "title": "Dimensionality Reduction",
            "description": "Learn how PCA and t-SNE simplify datasets for better visualization.",
            "content": "Dimensionality Reduction is used to compress features while preserving patterns. PCA focuses on variance, t-SNE on local similarities..."
        },
        {
            "title": "Anomaly Detection",
            "description": "Explore techniques to identify unusual data points using Isolation Forest, One-Class SVM.",
            "content": "Anomalies are data points that deviate significantly from the norm. Isolation Forest isolates anomalies by random feature slicing..."
        },
        {
            "title": "Linear Regression",
            "description": "Master the most basic predictive model for continuous outcomes.",
            "content": "Linear regression models the relationship between a dependent variable and one or more independent variables..."
        },
        {
            "title": "Logistic Regression",
            "description": "Predict binary outcomes using sigmoid-based probability estimation.",
            "content": "Logistic regression transforms linear output using the sigmoid function to estimate probabilities for classification..."
        }
    ]
    return jsonify({"tutorials": tutorials})


@app.route('/feature_importance', methods=['POST'])
def feature_importance():
    try:
        data = request.json
        dataset_name = datasets.get(data.get('dataset'))
        algorithm_name = data.get('algorithm')

        if dataset_name not in datasets.values() or algorithm_name not in algorithms:
            return jsonify({"error": "Invalid dataset or algorithm"}), 400

        df = load_dataset(dataset_name)
        if df is None:
            return jsonify({"error": "Dataset not found"}), 404

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        model = algorithms[algorithm_name]
        model.fit(X, y)

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = model.coef_
            if hasattr(importance, 'toarray'):
                importance = importance.toarray()[0]
        else:
            return jsonify({"error": f"{algorithm_name} does not support feature importance."}), 400

        features = X.columns
        sorted_idx = np.argsort(importance)[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis")
        plt.title(f"Feature Importance - {algorithm_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({"feature_importance_graph": graph_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
