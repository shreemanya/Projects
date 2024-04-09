# Import necessary libraries
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
# import matplotlib
# # matplotlib.use('Agg')  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib  # For saving and loading the model

app = Flask(__name__)

def plot_to_base64(plt_figure):
    """Converts a matplotlib plot to a base64 string."""
    img = BytesIO()
    plt_figure.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    base64_img = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(plt_figure)  # Close the figure to prevent re-use
    return base64_img

def preprocess_and_train():
    # Load the dataset
    data_path = 'training_data.csv'  # Training data set path
    df = pd.read_csv(data_path) # read and store data in a panda variable

    # Drop non-numeric columns explicitly and handle the target column
    X = df.drop(['No.','Time','Source','Destination','Classification'], axis=1)  # Assuming 'No.','Time','Source','Destination','Classification' is a non-numeric column
    print(X.head())
    y = df['Classification']  # storing Classification column in y variable

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)

    # Train the RandomForestClassifier
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model = KNeighborsClassifier(n_neighbors=5)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y)

    # Saving the model and scaler
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

preprocess_and_train() # training and pre processing function declaration

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    if request.method == 'POST':
        f = request.files['file']
        if not f:
            return "No file uploaded", 400

        # Load the model and scaler
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')

        # Read the testing data set file
        df = pd.read_csv(f)
        # Drop 'hash', 'Protocol' column and directly use the model for predictions
        X = df.drop(['No.','Time','Source','Destination','Classification'], axis=1, errors='ignore')  # Use errors='ignore' to avoid issues
        X_scaled = scaler.transform(X)  # Scale features

        # Make predictions
        y_pred = model.predict(X_scaled)
        y_true = df['Classification'].values  # Assuming the uploaded file includes the true labels

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)

        # Additional Plots
        # Histograms for feature distributions
        num_features = X.shape[1]  # Get the number of features to determine grid size
        num_rows = num_features // 3 + (num_features % 3 > 0)  # Arrange subplots in a 3-column grid
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))  # Adjust the size as needed (3 columns here)
        axes = axes.flatten()

        # Plot histograms on their respective axes
        for i, col in enumerate(X.columns):
            sns.histplot(X[col], bins=20, kde=False, ax=axes[i])  # Adjust the number of bins as needed

        # Remove any empty subplots
        for i in range(num_features, num_rows * 3):
            fig.delaxes(axes[i])

        plt.tight_layout()
        histograms_plot = plt.gcf()  # Get the current figure
        histograms_base64 = plot_to_base64(histograms_plot)

        # Return the rendered template with plots
        return render_template(
            'dashboard.html',
            accuracy=accuracy,
            class_report=class_report,
            histograms=histograms_base64,
        )
def generate_confusion_matrix_plot(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.close()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

if __name__ == '__main__':
    app.run(debug=True, port=5001)