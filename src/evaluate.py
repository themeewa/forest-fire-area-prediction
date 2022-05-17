# author: Gautham Pughazhendhi
# date: 2021-Nov-24

"""Reads in the train and test data, and outputs a tuned model and cross-validation results

Usage: evaluate.py --test_data=<test_data> --results_path=<results_path>
evaluate.py --test_data=../data/processed/test.csv --results_path=results_kr --algo=kr
 
Options: 
--test_data=<test_data>          Testing data path
--results_path=<results_path>    Output path for test data evaluation results
"""
import config
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from preprocess_n_tune import root_mean_squared_error, read_data
from docopt import docopt
from scipy.integrate import simps
# from slickml.metrics import RegressionMetrics

def load_model(path_prefix):
    """
    Loads the model from the path prefix

    Parameters
    ----------
    path_prefix : str
        Path prefix of the model

    Returns
    ----------
        Pipeline: saved model
    """

    with open(f"{path_prefix}_{config.result_prefix}_tuned_model.pickle", "rb") as f:
        model = pickle.load(f)
        
    return model

def evaluate_model(model, algo, X_test, y_test, path_prefix):
    """
    Evaluates the best fit model on the test data

    Parameters
    ----------
    model: Pipeline
        the best fit model
    X_test : pandas DataFrame
        X in the test data
    y_test : numpy array
        y in the test data 

    Returns
    ----------
        DataFrame: best fi model's results on the test data
    """
    
    test_scores = {}

    predictions = model.predict(X_test)
#     test_scores[f"SVR_Optimized_MAE"] = mean_absolute_error(y_test, predictions)
#     test_scores[f"SVR_Optimized_RMSE"] = root_mean_squared_error(y_test, predictions)
    test_scores[f"Optimized_MAE"] = mean_absolute_error(y_test, predictions)
    test_scores[f"Optimized_RMSE"] = root_mean_squared_error(y_test, predictions)
#     REC CURVE
#     reg_metrics = RegressionMetrics(y_test, predictions)
#     reg_metrics.plot()
    plot_REC(y_test, predictions, path_prefix)
    plot_comparision(y_test, predictions, path_prefix)

    return predictions, pd.DataFrame(test_scores, index=["Test Score"])

def store_results(results, path_prefix):
    """
    Saves the test results as an image

    Parameters
    ----------
    results : DataFrame
        test results dataframe
    path_prefix: str
        output path prefix
    """
    
    dfi.export(results, f"{path_prefix}_{config.result_prefix}_test_results.png",table_conversion='matplotlib')
    

def plot_predictions(y_true, y_pred, path_prefix):
    """
    Saves the comparison plot of actual vs predicted values as an image

    Parameters
    ----------
    y_true : numpy array
        actual values
    y_pred : numpy array
        predicted values
    path_prefix: str
        output path prefix
    """
    
    plt.scatter(y_true, y_pred)
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(y_pred), max(y_true))
    p2 = min(min(y_pred), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.axis('equal')

    plt.xlabel("Log True Area(ha)")
    plt.ylabel("Log Predicted Area(ha)")
    plt.savefig(f"{path_prefix}_{config.result_prefix}_predictions.png")

# Function for Regression Error Characteritic Curve

def REC(y_true , y_pred):

    # initilizing the lists
    Accuracy = []

    # initializing the values for Epsilon
    Begin_Range = 0
    End_Range = 1.5
    Interval_Size = 0.01

    # List of epsilons
    Epsilon = np.arange(Begin_Range , End_Range , Interval_Size)

    # Main Loops
    for i in range(len(Epsilon)):
        count = 0.0
        for j in range(len(y_true)):
            if np.linalg.norm(y_true[j] - y_pred[j]) / np.sqrt( np.linalg.norm(y_true[j]) **2 + np.linalg.norm(y_pred[j])**2 ) < Epsilon[i]:
                count = count + 1

        Accuracy.append(count/len(y_true))

    # Calculating Area Under Curve using Simpson's rule
    AUC = simps(Accuracy , Epsilon ) / End_Range

    # returning epsilon , accuracy , area under curve
    return Epsilon , Accuracy , AUC
#

def plot_comparision(y_true , y_pred, path_prefix):
    RR = r2_score(y_true , y_pred)
    xx = np.arange(1, y_pred.size + 1, 1)
    plt.figure(figsize=(6 , 6))
    plt.scatter(xx, y_true, linewidths=.5, marker ="s", label="real")
    plt.scatter(xx, y_pred, linewidths=.5, alpha = 0.5, label="predicted")
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
#     plt.plot([y_true.min(), y_true.max()], [xx.min(), xx.max()], 'k--', lw=1)
    plt.text(45, -5, r"$R^2 = %0.4f$" %RR , fontsize=15)
    plt.legend(loc='upper center', shadow=True, fontsize='small')
    plt.savefig(f"{path_prefix}_{config.result_prefix}_predictions_comparison.png")
    plt.show()


def plot_REC(y_true , y_pred, path_prefix):
    Deviation , Accuracy, AUC = REC(y_true , y_pred)
    plt.figure(figsize=(6 , 6))
    plt.title("Regression Error Characteristic (REC)")
    plt.plot(Deviation, Accuracy, "--b",lw =1)
    plt.xlabel("Deviation")
    plt.ylabel("Accuracy (%)")
    plt.text(1.1, 0.07, "AUC = %0.4f" %AUC , fontsize=15)
    plt.savefig(f"{path_prefix}_{config.result_prefix}_REC.png")
#     plt.show()

def main(opt):
    algo = opt["--algo"]
    model = load_model(opt["--results_path"])
    X_test, y_test = read_data(opt["--test_data"])
    
    predictions, results = evaluate_model(model,algo, X_test, y_test, opt["--results_path"])
    assert(isinstance(results, pd.DataFrame))

    store_results(results, opt["--results_path"])
    plot_predictions(y_test, predictions, opt["--results_path"])
    
if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt)