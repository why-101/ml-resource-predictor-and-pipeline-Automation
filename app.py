import os
import io
import base64
import traceback
import re
import contextlib # For redirecting stdout

# --- FIX 1: Set Matplotlib backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
# --- END FIX 1 ---

from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import joblib


# Load environment variables from .env file (for API key)
load_dotenv()

# --- IMPORTANT: Configure your Google Generative AI API Key ---
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv('GENAI_API_KEY')) # Fetches from .env or environment variables
    sql_model = genai.GenerativeModel("gemini-1.5-flash")
    gemini_model_for_ml_code = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Warning: Could not initialize Google Generative AI. SQL to NoSQL and ML code generation will not work. Error: {e}")
    genai = None # Set to None if import fails

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'super_secret_and_complex_key_change_me_in_production_12345') # !!! CHANGE THIS IN PRODUCTION !!!
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16 MB upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global Resources (Loaded Once at App Startup) ---
time_model = None
memory_model = None
size_model = None
scalers = {}
le = None
model_load_success = True

try:
    time_model = joblib.load("models/RandomForest_time.pkl")
    memory_model = joblib.load("models/svr_model.pkl")
    size_model = joblib.load("models/RandomForest_space.pkl")

    scalers = {        'size_mb': joblib.load("scalers/size_mb_scaler.pkl"),
        'records': joblib.load("scalers/records_scaler.pkl"),
        'features': joblib.load("scalers/features_scaler.pkl"),
        'split_ratio': joblib.load("scalers/split_ratio_scaler.pkl"),
        'ram_gb': joblib.load("scalers/ram_gb_scaler.pkl")
    }
    le = joblib.load("encoders/model_encoder.pkl")
    print("✅ All ML models, scalers, and encoder loaded successfully.")
except FileNotFoundError as e:    
    print(f"❌ Error loading ML resources: {e}")
    print("Please ensure 'models/', 'scalers/', and 'encoders/' directories exist and contain the necessary .pkl files.")
    model_load_success = False
except Exception as e:
    print(f"❌ An unexpected error occurred during ML resource loading: {e}")
    model_load_success = False


# Define model options per task
model_options = {
    "classification": [
        "Logistic Regression", "Gaussian NB", "Random Forest", "Decision Tree", "AdaBoost",
        "KNN", "SVM"
    ],
    "regression": [
        "Linear Regression", "SVR", "Ridge Regression", "SGD Regression","Random Forest Regressor","Gradient Boosting"
    ],
    "clustering": [
        "KMeans", "DBSCAN", "Agglomerative"
    ]
}

# Define evaluation metrics for GenAI code generation
metric_options = {
    "classification": ["accuracy_score", "precision_score", "recall_score", "f1_score", "roc_auc_score", "classification_report"],
    "regression": ["mean_squared_error", "mean_absolute_error", "r2_score"],
    "clustering": ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"]
}

# --- Utility Functions ---

def plot_to_img_tag(fig):
    """Converts matplotlib figure to base64 img tag string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig) # Close the figure to free memory
    return f'data:image/png;base64,{img_base64}'

def scale_input_df(df_row):
    """Applies loaded scalers to the relevant columns in a DataFrame row."""
    scaled_df_row = df_row.copy()
    for col, scaler in scalers.items():
        if col in scaled_df_row.columns and not scaled_df_row[col].empty:
            if pd.api.types.is_numeric_dtype(scaled_df_row[col]):
                scaled_df_row[col] = scaler.transform(scaled_df_row[[col]])
            else:
                app.logger.warning(f"Column '{col}' is not numeric and cannot be scaled. Skipping.")
    return scaled_df_row

def clean_and_execute_ml_code(generated_code, dataset_path, df_columns, target_variable, task_type):
    """
    Cleans markdown from generated code and attempts to execute it in a controlled
    environment, capturing stdout and errors.

    WARNING: Executing arbitrary AI-generated code is a significant security risk.
    In a production environment, this should be done in a highly sandboxed, isolated
    environment (e.g., a separate container) or avoided entirely by parsing the AI's
    response into pre-defined, safe operations.
    """
    cleaned_code = re.sub(r"```(?:python)?\s*([\s\S]*?)```", r"\1", generated_code).strip()

    # Create a dictionary for the execution environment
    # Only expose necessary and safe modules/functions
    exec_globals = {
        'pd': pd,
        'np': np,
        'train_test_split': getattr(__import__('sklearn.model_selection'), 'train_test_split', None),
        'LabelEncoder': getattr(__import__('sklearn.preprocessing'), 'LabelEncoder', None),
        'StandardScaler': getattr(__import__('sklearn.preprocessing'), 'StandardScaler', None),
        'MinMaxScaler': getattr(__import__('sklearn.preprocessing'), 'MinMaxScaler', None),
        'RobustScaler': getattr(__import__('sklearn.preprocessing'), 'RobustScaler', None),
        # Dynamically import models
        'LogisticRegression': getattr(__import__('sklearn.linear_model'), 'LogisticRegression', None),
        'GaussianNB': getattr(__import__('sklearn.naive_bayes'), 'GaussianNB', None),
        'RandomForestClassifier': getattr(__import__('sklearn.ensemble'), 'RandomForestClassifier', None),
        'DecisionTreeClassifier': getattr(__import__('sklearn.tree'), 'DecisionTreeClassifier', None),
        'AdaBoostClassifier': getattr(__import__('sklearn.ensemble'), 'AdaBoostClassifier', None),
        'KNeighborsClassifier': getattr(__import__('sklearn.neighbors'), 'KNeighborsClassifier', None),
        'SVC': getattr(__import__('sklearn.svm'), 'SVC', None),
        'LinearRegression': getattr(__import__('sklearn.linear_model'), 'LinearRegression', None),
        'SVR': getattr(__import__('sklearn.svm'), 'SVR', None),
        'Ridge': getattr(__import__('sklearn.linear_model'), 'Ridge', None),
        'SGDRegressor': getattr(__import__('sklearn.linear_model'), 'SGDRegressor', None),
        'RandomForestRegressor': getattr(__import__('sklearn.ensemble'), 'RandomForestRegressor', None),
        'GradientBoostingRegressor': getattr(__import__('sklearn.ensemble'), 'GradientBoostingRegressor', None),
        'KMeans': getattr(__import__('sklearn.cluster'), 'KMeans', None),
        'DBSCAN': getattr(__import__('sklearn.cluster'), 'DBSCAN', None),
        'AgglomerativeClustering': getattr(__import__('sklearn.cluster'), 'AgglomerativeClustering', None),
        # Dynamically import metrics
        'accuracy_score': getattr(__import__('sklearn.metrics'), 'accuracy_score', None),
        'precision_score': getattr(__import__('sklearn.metrics'), 'precision_score', None),
        'recall_score': getattr(__import__('sklearn.metrics'), 'recall_score', None),
        'f1_score': getattr(__import__('sklearn.metrics'), 'f1_score', None),
        'roc_auc_score': getattr(__import__('sklearn.metrics'), 'roc_auc_score', None),
        'mean_squared_error': getattr(__import__('sklearn.metrics'), 'mean_squared_error', None),
        'mean_absolute_error': getattr(__import__('sklearn.metrics'), 'mean_absolute_error', None),
        'r2_score': getattr(__import__('sklearn.metrics'), 'r2_score', None),
        'silhouette_score': getattr(__import__('sklearn.metrics'), 'silhouette_score', None),
        'davies_bouldin_score': getattr(__import__('sklearn.metrics'), 'davies_bouldin_score', None),
        'calinski_harabasz_score': getattr(__import__('sklearn.metrics'), 'calinski_harabasz_score', None),
        'classification_report': getattr(__import__('sklearn.metrics'), 'classification_report', None),
        # Synthetic data generation
        'make_classification': getattr(__import__('sklearn.datasets'), 'make_classification', None),
        'make_regression': getattr(__import__('sklearn.datasets'), 'make_regression', None),
        'dataset_path': dataset_path,
        'target_variable': target_variable,
        'df_columns': df_columns,
        'task_type': task_type
    }

    output_buffer = io.StringIO()
    execution_success = False
    error_message = None
    output = ""  # Initialize output variable here to ensure it's always defined

    with contextlib.redirect_stdout(output_buffer):
        try:
            exec(cleaned_code, exec_globals)
            execution_success = True
        except Exception as e:
            error_message = f"Execution error: {e}"
            traceback.print_exc(file=output_buffer)
            app.logger.error(f"Error executing generated code: {e}", exc_info=True)
        # Retrieve the output after try-except so it's available whether exception occurred or not
        output = output_buffer.getvalue()

    return execution_success, output, error_message
# --- Routes ---

@app.route('/')
def home():
    """Renders the homepage with links to different tools."""    
    return render_template('home.html')

# In app.py

# ... (other imports and code) ...

# --- FIX 2: New route to explicitly clear ML session data ---
@app.route('/clear-ml-session')
def clear_ml_session():
    """Clears all session data related to the ML experimental pipeline."""
    session.pop('dataset_path', None)
    session.pop('records', None)    
    session.pop('features', None)
    session.pop('df_columns', None)
    session.pop('df_head_html', None)
    session.pop('eda_from_file', None)
    session.pop('last_experiment_results', None)
    session.pop('last_input_df_for_genai', None)
    session.pop('all_experiment_results', None)
    session.pop('prompt_history', None)
    session.modified = True
    flash("ML pipeline session cleared. You can now start a fresh experiment.", 'info')
    return redirect(url_for('ml_pipeline_start'))
# --- END FIX 2 ---

@app.route('/ml-pipeline-start')
def ml_pipeline_start():
    """Initial page for ML pipeline, allowing choice of input mode."""
    # --- FIX 2: Do NOT clear session here by default.
    #    Instead, check for existing experiments and let the user choose to clear explicitly.
    if 'all_experiment_results' not in session or not isinstance(session['all_experiment_results'], list):
        session['all_experiment_results'] = []        
        session.modified = True
    
    has_existing_experiments = bool(session['all_experiment_results'])

    return render_template('ml_pipeline_start.html', has_existing_experiments=has_existing_experiments)


@app.route('/upload-dataset', methods=['GET', 'POST'])
def upload_dataset():
    """Handles dataset upload, performs basic EDA, and stores metadata in session."""
    if request.method == 'POST':
        if 'dataset_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['dataset_file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            counter = 1
            original_filepath = filepath
            while os.path.exists(filepath):
                name, ext = os.path.splitext(original_filepath)
                filepath = f"{name}_{counter}{ext}"                
                counter += 1
            file.save(filepath)
            app.logger.info(f"File saved to: {filepath}")

            try:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    app.logger.warning("UTF-8 decode failed, trying 'ISO-8859-1'...")
                    df = pd.read_csv(filepath, encoding='ISO-8859-1')
                session['dataset_path'] = filepath

                info_buf = io.StringIO()
                df.info(buf=info_buf)
                info_text = info_buf.getvalue()
                description = df.describe(include='all').to_html(classes="table table-striped", border=0)
                nulls = df.isnull().sum().to_frame(name='Null Count').to_html(classes="table table-bordered", border=0)
                duplicates = df.duplicated().sum()
                
                # Unique Values per column
                unique_values = df.nunique().to_frame(name='Unique Count').to_html(classes="table table-bordered", border=0)

                # Correlation Matrix (table)
                corr_matrix_table = None
                numeric_df_for_corr = df.select_dtypes(include=[np.number])
                if not numeric_df_for_corr.empty and len(numeric_df_for_corr.columns) > 1:
                    corr_matrix_table = numeric_df_for_corr.corr().to_html(classes="table table-bordered", border=0)

                # Correlation Heatmap (image)
                corr_img = None
                if not numeric_df_for_corr.empty and len(numeric_df_for_corr.columns) > 1:
                    fig_corr = plt.figure(figsize=(15, 10))
                    sns.heatmap(numeric_df_for_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
                    plt.title("Correlation Heatmap")
                    plt.tight_layout()
                    corr_img = plot_to_img_tag(fig_corr)
                else:
                    app.logger.info("No sufficient numeric columns for correlation heatmap.")

                # Boxplot of Numerical Features
                boxplot_img = None
                boxplot_info_message = None
                numeric_cols_for_boxplot = df.select_dtypes(include=[np.number]).columns
                
                # Filter out columns that are constant (have only one unique value)
                non_constant_numeric_cols = [col for col in numeric_cols_for_boxplot if df[col].nunique() > 1]

                if not non_constant_numeric_cols:
                    app.logger.info("No non-constant numerical columns found for boxplot.")
                    boxplot_info_message = "No numerical columns with varying values found to generate a boxplot."
                elif len(non_constant_numeric_cols) > 20:
                    app.logger.warning("Too many non-constant numerical columns for a readable boxplot. Skipping boxplot generation.")
                    boxplot_info_message = "Too many numerical columns to generate a readable boxplot. Consider plotting a subset or using a different visualization."
                else:
                    try:
                        fig_boxplot = plt.figure(figsize=(20, 10))
                        plt.boxplot(df[non_constant_numeric_cols].values, vert=True, patch_artist=True, medianprops=dict(color='red'))
                        plt.xticks(ticks=np.arange(1, len(non_constant_numeric_cols)+1),
                                   labels=non_constant_numeric_cols, rotation=45, ha='right')
                        plt.title("Boxplot of Numerical Features (Non-Constant Columns)")
                        plt.ylabel("Value")
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        boxplot_img = plot_to_img_tag(fig_boxplot)
                    except Exception as e:
                        app.logger.error(f"Error generating boxplot: {e}", exc_info=True)                        
                        boxplot_info_message = f"Error generating boxplot: {e}. Please check numerical data or try a different dataset."

                session['records'] = df.shape[0]
                session['features'] = df.shape[1]
                session['df_columns'] = df.columns.tolist()
                session['df_head_html'] = df.head().to_html(classes="table table-hover", border=0)
                
                session['eda_from_file'] = True

                flash('Dataset loaded successfully and EDA performed!', 'success')
                return render_template('dataset_summary.html',                    head=session['df_head_html'],
                    info=info_text,
                    description=description,
                    nulls=nulls,
                    duplicates=duplicates,
                    unique_values=unique_values,
                    corr_matrix_table=corr_matrix_table,
                    corr_img=corr_img,
                    boxplot_img=boxplot_img,
                    boxplot_info_message=boxplot_info_message
                    )
            except Exception as e:
                flash(f"Error processing dataset: {e}. Please ensure it's a valid CSV.", 'danger')
                app.logger.error(f"Error during dataset processing: {e}", exc_info=True)                
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.', 'danger')
            return redirect(request.url)
    return render_template('upload_dataset.html')


@app.route('/run-experiment', methods=['GET', 'POST'])
def run_experiment():
    """Allows users to input parameters for ML experiment prediction."""
    if not model_load_success:
        flash("ML prediction models are not loaded. This feature is disabled.", 'danger')
        return redirect(url_for('home'))

    initial_data = {
        'size_mb': 0.0,
        'records': 0,
        'features': 0,
        'ram_gb': 8.00,        
        'split_ratio': 0.3, # Default for manual entry
        'dataset_path_info': 'Enter metadata manually for research.'
    }
    
    if session.get('eda_from_file'):
        try:
            dataset_path = session.get('dataset_path')
            if dataset_path and os.path.exists(dataset_path):
                initial_data['size_mb'] = round(os.path.getsize(dataset_path) / 1_000_000, 2)
                initial_data['dataset_path_info'] = f"Metadata auto-filled from: {os.path.basename(dataset_path)}"
            initial_data['records'] = session.get('records', 0)
            initial_data['features'] = session.get('features', 0)
            initial_data['split_ratio'] = 0.3
        except Exception as e:
            flash(f"Could not auto-fill metadata from previous upload: {e}", 'warning')
            app.logger.warning(f"Error auto-filling from session: {e}")
            session.pop('eda_from_file', None)

    if request.method == 'POST':        
        try:
            size_mb = float(request.form['size_mb'])
            records = int(request.form['records'])
            features = int(request.form['features'])
            split_ratio = float(request.form['split_ratio'])
            ram_gb = float(request.form['ram_gb'])
            task_type = request.form['task_type'].lower()
            
            if task_type not in model_options:
                flash('Invalid task type selected.', 'danger')
                return redirect(request.url)
            algo = request.form['algorithm']
            if algo not in model_options[task_type]:
                flash('Invalid algorithm selected for this task type.', 'danger')
                return redirect(request.url)            
            processor_flag = request.form.get('processor_flag', 'off') == 'on'

            try:
                algo_encoded = le.transform([algo])[0]
            except Exception as e:
                flash(f"Error encoding algorithm '{algo}': {e}. Ensure encoder is loaded and algorithm is valid.", 'danger')
                app.logger.error(f"Error encoding algorithm: {e}", exc_info=True)
                return redirect(request.url)

            input_data_for_df = {
                "size_mb": size_mb,
                "records": records,
                "features": features,
                "split_ratio": split_ratio,
                "ram_gb": ram_gb,
                "task_classification": task_type == "classification",
                "task_clustering": task_type == "clustering",
                "task_regression": task_type == "regression",
                "processor_Intel64 Family 6 Model 165 Stepping 2, GenuineIntel": processor_flag,
                "algo_encoded": algo_encoded
            }

            input_df_row = pd.DataFrame([input_data_for_df])
            input_df_scaled = scale_input_df(input_df_row)

            expected_cols = [
                "size_mb", "records", "features", "split_ratio", "ram_gb",
                "task_classification", "task_clustering", "task_regression",
                "processor_Intel64 Family 6 Model 165 Stepping 2, GenuineIntel",
                "algo_encoded"
            ]
            input_df_final = input_df_scaled.reindex(columns=expected_cols, fill_value=0)            
            pred_time = time_model.predict(input_df_final)[0]
            pred_memory = memory_model.predict(input_df_final)[0]
            pred_size = size_model.predict(input_df_final)[0]

            current_experiment_results = {
                "model_name": algo,
                "size_mb": size_mb, "records": records, "features": features,
                "split_ratio": split_ratio, "ram_gb": ram_gb, "task_type": task_type,
                "algorithm": algo, "processor_flag": processor_flag,                "TrainingTime_s": round(pred_time, 3),
                "MemoryUsage_MB": round(pred_memory, 3),
                "ModelSize_MB": round(pred_size, 3)
            }

            if 'all_experiment_results' not in session or session['all_experiment_results'] is None:
                session['all_experiment_results'] = []
            
            experiment_id = len(session['all_experiment_results']) + 1
            current_experiment_results['Model'] = f"Model_{experiment_id}"
            
            session['all_experiment_results'].append(current_experiment_results)
            session.modified = True

            session['mode_choice_manual'] = not session.get('eda_from_file', False)

            flash('Experiment simulation completed successfully!', 'success')
            return render_template('experiment_results.html', 
                                   current_result=current_experiment_results,
                                   total_experiments=len(session['all_experiment_results']))

        except ValueError as ve:
            flash(f"Input Error: {ve}. Please check your numerical inputs.", 'danger')
            app.logger.error(f"Input validation error: {ve}")
            return redirect(request.url)
        except Exception as e:
            flash(f"An unexpected error occurred during experiment prediction: {e}", 'danger')
            app.logger.error(f"Error during experiment prediction: {e}", exc_info=True)
            return redirect(request.url)

    return render_template('run_experiment.html',
                           model_options=model_options,
                           initial_data=initial_data)


@app.route('/experiment-results')
def experiment_results():
    """Displays the result of the most recently run experiment."""
    if 'all_experiment_results' not in session or not session['all_experiment_results']:
        flash("No experiments have been run yet.", 'info')
        return redirect(url_for('run_experiment'))
    
    current_result = session['all_experiment_results'][-1]
    total_experiments = len(session['all_experiment_results'])
    
    return render_template('experiment_results.html',
                           current_result=current_result,
                           total_experiments=total_experiments)


@app.route('/all-experiments-summary')
def all_experiments_summary():
    """Displays a summary table and visualizations of all run experiments."""
    if 'all_experiment_results' not in session or not session['all_experiment_results']:
        flash("No experiments have been run yet to summarize.", 'info')
        return redirect(url_for('run_experiment'))

    results_data = session['all_experiment_results']
    results_df = pd.DataFrame(results_data)    
    plot_images = {}

    fig_time = plt.figure(figsize=(10, 6))    
    sns.barplot(data=results_df, x="Model", y="TrainingTime_s")
    plt.title("Training Time (seconds) per Model")
    plt.ylabel("Time (s)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_images['time'] = plot_to_img_tag(fig_time)

    fig_memory = plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Model", y="MemoryUsage_MB")
    plt.title("Memory Usage (MB) per Model")
    plt.ylabel("Memory (MB)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_images['memory'] = plot_to_img_tag(fig_memory)

    fig_size = plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Model", y="ModelSize_MB")
    plt.title("Model Size (MB) per Model")
    plt.ylabel("Size (MB)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_images['size'] = plot_to_img_tag(fig_size)    
    return render_template('all_experiments_summary.html',
                           results_table=results_df.to_html(classes="table table-striped table-hover", index=False, border=0),
                           plot_images=plot_images,
                           experiments=results_data)

@app.route('/sql-to-nosql', methods=['GET', 'POST'])
def sql_to_nosql():
    """Converts a MySQL query to a target NoSQL operation using a generative AI model."""
    if genai is None:
        flash("Google Generative AI is not initialized. SQL to NoSQL conversion is unavailable.", 'danger')
        return redirect(url_for('home'))

    converted_code = None
    error = None
    if request.method == 'POST':
        sql_query = request.form['sql_query']        
        target_db = request.form.get('target_db', 'MongoDB')

        if not sql_query.strip():
            flash("Please enter a SQL query.", 'warning')
            return redirect(request.url)        
        try:
            prompt = f"""
You are a code conversion assistant. Convert the following MySQL query to its equivalent {target_db} NoSQL operation.

MySQL Query:
{sql_query}

Important:- Use idiomatic syntax of {target_db}.
- Provide full code, not just a partial conversion.
- If a join is not directly possible in NoSQL, simulate with aggregation or nested queries.

Only output the code (no explanation).
"""
            response = sql_model.generate_content(prompt)
            raw_code = response.text.strip()
            cleaned_code = re.sub(r"```(?:javascript)?\s*([\s\S]*?)```", r"\1", raw_code).strip()
            converted_code = cleaned_code

            flash('SQL query converted successfully!', 'success')
        except Exception as e:
            error = str(e)
            flash(f"Error during conversion: {e}", 'danger')
            app.logger.error(f"Error during SQL to NoSQL conversion: {e}", exc_info=True)

    return render_template('sql_to_nosql.html', code=converted_code, error=error)

@app.route('/generate-ml-code/<int:experiment_idx>', methods=['GET', 'POST'])
def generate_ml_code(experiment_idx):
    # --- Debugging: Comprehensive logging at the start of the request ---
    app.logger.debug(f"Entering generate_ml_code for index {experiment_idx}")
    app.logger.debug(f"Current session keys: {list(session.keys())}")

    if 'all_experiment_results' in session:
        app.logger.debug(f"session['all_experiment_results'] content (first 2 items): {session['all_experiment_results'][:2]}")
        app.logger.debug(f"Length of session['all_experiment_results']: {len(session['all_experiment_results'])}")
    else:
        app.logger.debug("session['all_experiment_results'] is NOT present in session.")

    # --- Consolidated and robust validation at the very beginning ---
    if genai is None:
        flash("Google Generative AI is not initialized. ML code generation is unavailable.", 'danger')
        app.logger.warning("GenAI not initialized, redirecting to home.")
        return redirect(url_for('home'))

    if 'all_experiment_results' not in session or not isinstance(session['all_experiment_results'], list) or not session['all_experiment_results']:
        flash("No previous experiment found. Please run an ML experiment first.", 'danger')
        app.logger.warning(
            "Validation failed: 'all_experiment_results' missing/not a list/or empty. Redirecting to run_experiment."
        )
        return redirect(url_for('run_experiment'))

    if experiment_idx < 0 or experiment_idx >= len(session['all_experiment_results']):
        flash("Selected experiment not found or index is out of range. Please try again.", 'danger')
        app.logger.warning(
            f"Validation failed: Experiment index {experiment_idx} is out of range. "
            f"List length: {len(session['all_experiment_results'])}. Redirecting to all_experiments_summary."
        )
        return redirect(url_for('all_experiments_summary'))
    # --- End Consolidated Validation ---

    selected_experiment = session['all_experiment_results'][experiment_idx]
    app.logger.debug(f"Successfully selected experiment: {selected_experiment.get('Model', 'N/A')}")

    dataset_path = session.get('dataset_path')
    df_columns = session.get('df_columns', [])
    mode_choice_manual = session.get('mode_choice_manual', False)

    available_metrics = metric_options.get(selected_experiment['task_type'], [])
    app.logger.debug(f"Available metrics for task '{selected_experiment['task_type']}': {available_metrics}")

    generated_code = None
    execution_output = None
    execution_success = False
    error = None

    # Ensure 'prompt_history' structure
    if 'prompt_history' not in session or not isinstance(session['prompt_history'], dict):
        session['prompt_history'] = {}
        session.modified = True
        app.logger.debug("Initialized 'prompt_history' in session.")

    experiment_history_key = str(experiment_idx)
    if experiment_history_key not in session['prompt_history'] or not isinstance(session['prompt_history'][experiment_history_key], list):
        session['prompt_history'][experiment_history_key] = []
        session.modified = True
        app.logger.debug(f"Initialized 'prompt_history[{experiment_history_key}]' in session.")

    # Read current form values, fallback defaults for GET requests
    current_target_variable = request.form.get('target_variable', '')
    current_preprocessing = request.form.get('preprocessing', 'None')
    current_metrics = request.form.getlist('metrics') if request.method == 'POST' else []

    app.logger.debug(f"Form values - target_variable: {current_target_variable}, preprocessing: {current_preprocessing}, metrics: {current_metrics}")

    if request.method == 'POST':
        action = request.form.get('action', '')
        app.logger.debug(f"POST action requested: '{action}'")

        # Re-fetch form fields on POST to ensure updated values for generate/improve
        current_target_variable = request.form.get('target_variable', '').strip()
        current_preprocessing = request.form.get('preprocessing', 'None').strip()
        current_metrics = request.form.getlist('metrics')

        app.logger.debug(f"Updated form data - target_variable: {current_target_variable}, preprocessing: {current_preprocessing}, metrics: {current_metrics}")

        # Safely join metrics string for prompt
        if not isinstance(current_metrics, list):
            app.logger.warning(f"Expected current_metrics as list but got {type(current_metrics)}. Resetting to empty list.")
            current_metrics = []
        metrics_str = ', '.join(current_metrics) if current_metrics else 'none'

        if action == 'generate':
            # Validation on the target variable
            if not current_target_variable:
                flash("Please enter a target variable.", "warning")
                app.logger.warning("Target variable empty on code generation request.")
                return render_template(
                    'generate_ml_code.html',
                    experiment_idx=experiment_idx,
                    selected_experiment=selected_experiment,
                    last_results=selected_experiment,
                    df_columns=df_columns,
                    available_metrics=available_metrics,
                    generated_code=None,
                    execution_output=None,
                    execution_success=False,
                    error=None,
                    current_target_variable=current_target_variable,
                    current_preprocessing=current_preprocessing,
                    current_metrics=current_metrics,
                )

            # Compose the prompt template
            prompt_template = f"""
You are an expert ML assistant. Generate Python code using scikit-learn to train and evaluate a {selected_experiment['task_type']} model.

Details:
- use UTF-8 or ISO-8859-1 while reading csv data
- Dataset Path: {dataset_path if dataset_path else 'None'}
- Model Algorithm: {selected_experiment['algorithm']}
- Target Variable: {current_target_variable}
- Number of Records: {selected_experiment['records']}
- Number of Features: {selected_experiment['features']}
- Dataset Columns: {df_columns} (use these for feature selection and handling)
- look for null values if there are any remove them first
- Make sure to handle non-numeric features (e.g., using pd.get_dummies() or LabelEncoder) before model training.
- Preprocessing: {current_preprocessing} (apply this to all feature columns except the target, if not 'None')

{"Generate a synthetic dataset using `sklearn.datasets.make_classification` or `make_regression` before training, as no dataset path was provided. The synthetic dataset should mimic the given 'Number of Records' and 'Number of Features'." if mode_choice_manual else "Load the dataset from the provided Dataset Path. If 'dataset_path' is 'None', generate synthetic data as a fallback."}

Required Steps:
- Split data using `train_test_split` with a test_size of {1 - selected_experiment['split_ratio']:.2f}.
- Apply the specified preprocessing.
- Train the model.
- Predict on the test set.
- Calculate and print the following evaluation metrics: {metrics_str}.
- If it's a classification task, also print a `classification_report`.

Only generate executable Python code. Do not include any explanations, markdown code fences, or extra text outside the code.
"""
            # Save prompt in session history
            session['prompt_history'][experiment_history_key] = [prompt_template]
            session.modified = True
            app.logger.debug(f"Generated new prompt template:\n{prompt_template}")

        elif action == 'improve':
            feedback_output = request.form.get('execution_output', '').strip()
            current_target_variable = request.form.get('target_variable', '').strip()
            current_preprocessing = request.form.get('preprocessing', 'None').strip()
            current_metrics = request.form.getlist('metrics')
            if not isinstance(current_metrics, list):
                current_metrics = []
            metrics_str = ', '.join(current_metrics) if current_metrics else 'none'

            app.logger.debug(f"Improve action with feedback output length {len(feedback_output)} chars.")
            if not session['prompt_history'][experiment_history_key]:
                flash("No previous code generation found to improve upon for this experiment.", "warning")
                app.logger.warning(f"No prompt history found to improve for experiment {experiment_history_key}.")
                return redirect(url_for('generate_ml_code', experiment_idx=experiment_idx))

            last_prompt = session['prompt_history'][experiment_history_key][-1]
            app.logger.debug(f"Last prompt length: {len(last_prompt)} characters.")

            prompt_template = f"""
{last_prompt}

The previous code execution resulted in the following output:
--- BEGIN EXECUTION OUTPUT ---
{feedback_output}
--- END EXECUTION OUTPUT ---
Please regenerate the code with improvements. Focus on enhancing the model's performance (e.g., by refining preprocessing, hyperparameter tuning, or trying alternative approaches) to achieve better {metrics_str}.

Only generate updated executable Python code. Do not include any explanations, markdown code fences, or extra text outside the code.
"""
            session['prompt_history'][experiment_history_key].append(prompt_template)
            session.modified = True
            app.logger.debug("Appended improved prompt template to prompt history.")

        else:
            flash("Invalid action.", "danger")
            app.logger.warning(f"Invalid action received in generate_ml_code route: '{action}'")
            return redirect(url_for('generate_ml_code', experiment_idx=experiment_idx))

        try:
            app.logger.debug("Sending prompt to Gemini model for ML code generation...")
            response = gemini_model_for_ml_code.generate_content(prompt_template)
            generated_code = response.text.strip()
            app.logger.debug(f"Received generated code of length {len(generated_code)} characters.")

            exec_success, exec_output, exec_error = clean_and_execute_ml_code(
                generated_code, dataset_path, df_columns, current_target_variable, selected_experiment['task_type']
            )
            execution_output = exec_output
            execution_success = exec_success
            error = exec_error

            if execution_success:
                flash("Code generated and executed successfully!", "success")
                app.logger.info("ML code generation and execution succeeded.")
            else:
                flash(f"Code execution failed: {error}. Please review the output.", "danger")
                app.logger.error(f"Code execution failed with error: {error}")

            session.modified = True

        except Exception as e:
            error = str(e)
            flash(f"Error generating or executing code: {error}", "danger")
            app.logger.error(f"Exception in ML code generation/execution: {error}", exc_info=True)

    # Render the page with all context variables to restore form state, output and errors
    return render_template(
        'generate_ml_code.html',
        experiment_idx=experiment_idx,
        selected_experiment=selected_experiment,
        last_results=selected_experiment,  # for template logic and display
        df_columns=df_columns,
        available_metrics=available_metrics,
        generated_code=generated_code,
        execution_output=execution_output,
        execution_success=execution_success,
        error=error,
        current_target_variable=current_target_variable,
        current_preprocessing=current_preprocessing,
        current_metrics=current_metrics,
    )

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(413) # Payload too large
def file_too_large(e):
    flash('File too large (max 16MB). Please upload a smaller CSV.', 'danger')
    return redirect(url_for('upload_dataset'))

@app.errorhandler(Exception)
def handle_general_exception(e):
    app.logger.error(f"An unhandled error occurred: {e}", exc_info=True)
    flash(f"An unexpected error occurred: {e}. Our apologies! Please try again or contact support.", 'danger')
    return render_template('error.html', error_message=str(e)), 500

# --- Run App ---
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('scalers', exist_ok=True)
    os.makedirs('encoders', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True)
