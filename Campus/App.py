import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sn
import matplotlib.pyplot as plt
import json
from streamlit_lottie import st_lottie

# Function to load the dataset
def load_data():
    # Upload CSV
    uploaded_file = st.file_uploader("Upload your dataset", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("Please upload a dataset!")
        return None
def load_lottie_local(filepath: str):
    with open(filepath, "r") as file:
        return json.load(file)
# Preprocessing function
def preprocess_data(df):
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Features and target
    X = df.drop("Placement(Y/N)?", axis=1)
    y = df["Placement(Y/N)?"]

    return X, y

# Function for model training
def train_model(model_choice, X_train, y_train, X_test, y_test):
    if model_choice == "SVC":
        # Define pipeline for SVC
        pipeline = ImbPipeline(steps=[ 
            ('scaler', StandardScaler()), 
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC())
        ])

        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    
    elif model_choice == "Decision Tree":
        # Define pipeline for Decision Tree
        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        param_grid = {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [3, 5, 10, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

    elif model_choice == "CNN + LSTM":
        # Convert and preprocess
        scaler = StandardScaler()
        X_combined = pd.concat([X_train, X_test])
        X_scaled = scaler.fit_transform(X_combined)

        y_combined = np.concatenate((y_train, y_test))

        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        y_cat = to_categorical(y_combined)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores = []
        fold_models = []

        st.subheader("CNN + LSTM K-Fold Results")
        for i, (train_idx, val_idx) in enumerate(kfold.split(X_reshaped, y_combined)):
            X_fold_train, X_fold_val = X_reshaped[train_idx], X_reshaped[val_idx]
            y_fold_train, y_fold_val = y_cat[train_idx], y_cat[val_idx]

            model = Sequential([
                Conv1D(64, 2, activation='relu', input_shape=(X_fold_train.shape[1], 1)),
                MaxPooling1D(2),
                LSTM(64),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dense(y_cat.shape[1], activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_fold_train, y_fold_train, epochs=20, batch_size=16, verbose=0)
            loss, acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
            acc_scores.append(acc)
            fold_models.append(model)
            #st.write(f"Fold {i+1} Accuracy: {acc:.4f}")

        st.write("Average K-Fold Accuracy:", np.mean(acc_scores))

        model = fold_models[np.argmax(acc_scores)]  # Optionally use the best model

    return model

# Function to make predictions
def make_predictions(model, X_test, model_choice):
    y_pred = model.predict(X_test)

    if model_choice == "CNN + LSTM":
        # y_pred is probabilities → convert to labels
        y_pred = np.argmax(y_pred, axis=1)

    return y_pred

# Streamlit App Layout
def main():
    st.title("Placement Data Analysis & Prediction App")

    # Sidebar
    section = st.sidebar.selectbox("Select a Section", ["Home", "Data Loading", "Model Training", "Predictions", "Reports"])

    # Home Section
    if section == "Home":
        lottie_file_path = "Animation - 1728670264444.json"
        lottie_animation = load_lottie_local(lottie_file_path)
        st_lottie(lottie_animation, speed=1, width=700, height=400, key="home_animation")
        st.write("Campus recruitment is a strategy for sourcing, engaging and hiring young talent for internship and entry-level positions. College recruiting is typically a tactic for medium- to large-sized companies with high-volume recruiting needs, but can range from small efforts (like working with university career centers to source potential candidates) to large-scale operations (like visiting a wide array of colleges and attending recruiting events throughout the spring and fall semester). Campus recruitment often involves working with university career services centers and attending career fairs to meet in-person with college students and recent graduates.")


    # Data Loading Section
    if section == "Data Loading":
        st.header("Data Loading")
        df = load_data()
        if df is not None:
            st.write(df.head())
            X, y = preprocess_data(df)
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training Section
    if section == "Model Training":
        st.header("Model Training")
        
        # Select model
        model_choice = st.selectbox("Select a Model to Train", ["SVC", "Decision Tree", "CNN + LSTM"])

        if st.button("Train Model and Make Predictions"):
            if hasattr(st.session_state, 'X_train') and hasattr(st.session_state, 'y_train'):
                # Train the selected model
                model = train_model(model_choice, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
                st.session_state.model = model
                st.session_state.model_choice = model_choice  # Save the model choice in session state
                st.success(f"{model_choice} model trained successfully!")

                # Make predictions
                y_pred = make_predictions(model, st.session_state.X_test, model_choice)
                st.write("Predictions:", y_pred)
                st.write("Accuracy:", accuracy_score(st.session_state.y_test, y_pred))
                st.write("Classification Report:\n", classification_report(st.session_state.y_test, y_pred))
            else:
                st.warning("Please load and preprocess data first.")

    # Predictions Section\
    if section == "Predictions":
        st.header("Make Real-Time Prediction")

        if hasattr(st.session_state, 'model') and hasattr(st.session_state, 'X_train'):

            # Feature columns expected by the model
            feature_columns = [
                'Gender', '10th board', '10th marks', '12th board', '12th marks',
                'Stream', 'Cgpa', 'Internships(Y/N)', 'Training(Y/N)', 'Backlogs',
                'Innovative Project(Y/N)', 'Communication level', 'Technical Course(Y/N)',
                'codingComptation', 'Internships', 'Projects', 'AMCAT_Total_Marks'
            ]

            st.subheader("Enter Candidate Details:")

            # --- Collect user input ---
            input_data = {
                'Gender': st.selectbox("Gender", ["Male", "Female"]),
                '10th board': st.selectbox("10th board", ["CBSE", "ICSE", "State Board"]),
                '10th marks': st.number_input("10th marks", min_value=0.0, max_value=100.0),
                '12th board': st.selectbox("12th board", ["CBSE", "ICSE", "State Board"]),
                '12th marks': st.number_input("12th marks", min_value=0.0, max_value=100.0),
                'Stream': st.selectbox("Stream", ["CSE", "ECE", "EEE", "IT", "Mechanical Engineering", "Civil"]),
                'Cgpa': st.number_input("CGPA", min_value=0.0, max_value=10.0),
                'Internships(Y/N)': st.selectbox("Internships(Y/N)", ["Yes", "No"]),
                'Training(Y/N)': st.selectbox("Training(Y/N)", ["Yes", "No"]),
                'Backlogs': st.selectbox("Backlogs", ["Yes", "No"]),
                'Innovative Project(Y/N)': st.selectbox("Innovative Project(Y/N)", ["Yes", "No"]),
                'Communication level': st.slider("Communication level (1-5)", 1, 5),
                'Technical Course(Y/N)': st.selectbox("Technical Course(Y/N)", ["Yes", "No"]),
                'codingComptation': st.selectbox("Coding Competition Participation", ["Yes", "No"]),
                'Internships': st.number_input("No. of Internships", min_value=0),
                'Projects': st.number_input("No. of Projects", min_value=0),
                'AMCAT_Total_Marks': st.number_input("AMCAT Total Marks", min_value=0)
            }

            # Create DataFrame from input
            user_input_df = pd.DataFrame([input_data])

            # Combine with X_train for consistent encoding
            X_train = st.session_state.X_train.copy()
            full_df = pd.concat([X_train, user_input_df], ignore_index=True)

            # Encode categorical columns using fresh LabelEncoder per column
            for col in full_df.select_dtypes(include='object').columns:
                le = LabelEncoder()
                full_df[col] = le.fit_transform(full_df[col].astype(str))

            # Extract processed user input
            processed_input = full_df.iloc[[-1]].values

            # --- Make Prediction ---
            if st.button('Predict'):
                model = st.session_state.model
                model_choice = st.session_state.model_choice

                if model_choice == "CNN + LSTM":
                    processed_input = processed_input.reshape((1, processed_input.shape[1], 1))
                    pred_prob = model.predict(processed_input)
                    prediction = np.argmax(pred_prob, axis=1)[0]
                else:
                    prediction = model.predict(processed_input)[0]

                label_map = {0: "Not Placed", 1: "Placed"}
                st.success(f"🎯 Prediction applied using {model_choice} : {label_map.get(prediction, prediction)}")

        else:
            st.warning("Please train a model first.")

    # Reports Section
    # Reports Section
    if section == "Reports":
        st.header("Model Performance Report")
        
        # Check if training data exists in session_state
        if hasattr(st.session_state, 'X_train') and hasattr(st.session_state, 'y_train'):
            # List of models to evaluate
            models = ["SVC", "Decision Tree", "CNN + LSTM"]
            model_metrics = []

            for model_choice in models:
                if model_choice == "CNN + LSTM":
                    # Handle CNN + LSTM model separately
                    model = train_model(model_choice, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
                    
                    # Reshape X_test for CNN + LSTM prediction
                    X_test_reshaped = st.session_state.X_test.values.reshape((st.session_state.X_test.shape[0], st.session_state.X_test.shape[1], 1))
                    
                    # Make predictions
                    y_pred = model.predict(X_test_reshaped)
                    y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
                else:
                    # Train other models (SVC and Decision Tree)
                    model = train_model(model_choice, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
                    y_pred = make_predictions(model, st.session_state.X_test, model_choice)

                # Compute metrics
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                precision = classification_report(st.session_state.y_test, y_pred, output_dict=True)['1']['precision']
                recall = classification_report(st.session_state.y_test, y_pred, output_dict=True)['1']['recall']

                # Store the metrics for later plotting
                model_metrics.append({
                    "Model": model_choice,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall
                })

            # Convert the model metrics into a DataFrame for easy plotting
            metrics_df = pd.DataFrame(model_metrics)

            # Plot the metrics
            st.subheader("Model Comparison: Accuracy, Precision, and Recall")
            
            # Create subplots for the metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            
            # Accuracy Comparison
            metrics_df.set_index('Model')[['Accuracy']].plot(kind='bar', ax=axes[0], color='green')
            axes[0].set_title("Accuracy Comparison")
            axes[0].set_ylabel("Score")
            axes[0].set_ylim(0, 1)
            
            # Precision Comparison
            metrics_df.set_index('Model')[['Precision']].plot(kind='bar', ax=axes[1], color='orange')
            axes[1].set_title("Precision Comparison")
            axes[1].set_ylabel("Score")
            axes[1].set_ylim(0, 1)
            
            # Recall Comparison
            metrics_df.set_index('Model')[['Recall']].plot(kind='bar', ax=axes[2], color='red')
            axes[2].set_title("Recall Comparison")
            axes[2].set_ylabel("Score")
            axes[2].set_ylim(0, 1)

            # Show the plot
            st.pyplot(fig)

            # Show the detailed classification report for each model
            st.write("Detailed Classification Reports:")
            for model_choice in models:
                st.subheader(f"Classification Report for {model_choice}")
                if model_choice == "CNN + LSTM":
                    # Handle CNN + LSTM model separately
                    model = train_model(model_choice, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
                    X_test_reshaped = st.session_state.X_test.values.reshape((st.session_state.X_test.shape[0], st.session_state.X_test.shape[1], 1))
                    y_pred = model.predict(X_test_reshaped)
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    # Train other models (SVC and Decision Tree)
                    model = train_model(model_choice, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
                    y_pred = make_predictions(model, st.session_state.X_test, model_choice)

                st.write(classification_report(st.session_state.y_test, y_pred))




if __name__ == "__main__":
    main()
