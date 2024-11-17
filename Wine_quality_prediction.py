import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_data(url):
    """Load the dataset from the given URL."""
    data = pd.read_csv(url, sep=';')
    return data

def visualize_data(data):
    """Visualize the distribution of wine quality."""
    sns.countplot(x='quality', data=data)
    plt.title('Wine Quality Distribution')
    plt.show()

def preprocess_data(data):
    """Preprocess the data: split features and target, normalize features."""
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # Normalize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    """Train the Random Forest model with SMOTE oversampling and class weights."""
    
    # Oversample the minority class using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print the classification report."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def predict_quality(model, scaler, input_features):
    """Predict the quality of wine based on user input."""
    # Scale the user input features before prediction
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)
    return prediction[0]

# Main execution flow
if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    # Load and visualize data
    data = load_data(url)
    visualize_data(data)

    # Preprocess data
    X, y, scaler = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # User input for prediction
    print("Enter the features for prediction (in the order of fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol):")
    
    # Ensure correct number of inputs (11 features)
    while True:
        try:
            # Attempt to read the user input as a list of floats
            user_input = list(map(float, input("Please enter 11 feature values separated by spaces: ").split()))
            
            # Check if the user has entered exactly 11 features
            if len(user_input) != 11:
                raise ValueError("Please provide exactly 11 features.")
            break
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
    
    try:
        # Make the prediction using the trained model
        predicted_quality = predict_quality(model, scaler, user_input)
        print(f'Predicted Wine Quality: {predicted_quality}')
    except Exception as e:
        print(f"Error: {e}.")
