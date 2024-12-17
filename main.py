from imports import *
from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
from matplotlib import pyplot
import os

from pyngrok import ngrok

app = Flask(__name__)

# Directory to save graphs
graph_dir = "static/graphs"
os.makedirs(graph_dir, exist_ok=True)

# Save and display graphs
def save_plot(fig, filename):
    filepath = os.path.join(graph_dir, filename)
    fig.savefig(filepath)
    return filepath

# Function to create and save plots for comparison
def create_algorithm_comparison_plots():
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    raw_plot_path = save_plot(fig, 'raw_algorithm_comparison.png')
    pyplot.close(fig)

    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
    pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
    pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
    pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
    fig = pyplot.figure()
    fig.suptitle('Scaled Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    scaled_plot_path = save_plot(fig, 'scaled_algorithm_comparison.png')
    pyplot.close(fig)

    ensembles = []
    ensembles.append(('AB', AdaBoostClassifier()))
    ensembles.append(('GBM', GradientBoostingClassifier()))
    ensembles.append(('RF', RandomForestClassifier()))
    ensembles.append(('ET', ExtraTreesClassifier()))
    results = []
    names = []
    for name, model in ensembles:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
    fig = pyplot.figure()
    fig.suptitle('Ensemble Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    ensemble_plot_path = save_plot(fig, 'ensemble_algorithm_comparison.png')
    pyplot.close(fig)

    return raw_plot_path, scaled_plot_path, ensemble_plot_path

# Create and save plots before starting Flask
raw_plot, scaled_plot, ensemble_plot = create_algorithm_comparison_plots()

ilename = 'sonar.all-data.csv'
dataset = read_csv(filename, header=None)

# Split dataset into features (X) and target (Y)
X = dataset.iloc[:, :5].values  # Only first 5 columns
Y = dataset.iloc[:, -1].values  # 60th column (Rock/Mine)

# Scale the data
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Train the SVM model
model = SVC(probability=True, random_state=42)
model.fit(X_scaled, Y)

# Example: Predict with new data (5 features)
new_data = [[0.1, 0.5, -0.3, 0.2, -0.1]]  # Example input with 5 features
new_data_scaled = scaler.transform(new_data)  # Scale the input data
prediction = model.predict(new_data_scaled)
print(f"Prediction: {prediction}")
@app.route('/')
def index():
    """Front Page with Navigation Options."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Algorithm Comparison Dashboard."""
    return render_template('dashboard.html',
                           raw_plot=url_for('static', filename='graphs/raw_algorithm_comparison.png'),
                           scaled_plot=url_for('static', filename='graphs/scaled_algorithm_comparison.png'),
                           ensemble_plot=url_for('static', filename='graphs/ensemble_algorithm_comparison.png'))

@app.route('/predict')
def predict():
    # Render the prediction form
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Extract user input from form submission
        input_data = request.form.to_dict(flat=True)
        input_values = np.array([float(value) for value in input_data.values()]).reshape(1, -1)

        # Ensure 5 inputs are provided
        if input_values.shape[1] != 5:
            return render_template('result.html', error='Please provide exactly 5 inputs.')



        # Make prediction
        prediction = model.predict(input_values)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == '__main__':
    # Authenticate ngrok

    app.run(host='0.0.0.0', port=5000)
