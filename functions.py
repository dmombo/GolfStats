import plotly.express as px
import plotly.subplots as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats

def visualize_column_distribution(df, list_of_cols):
    """
    Given a dataframe and a list of column names, 
    Plots all the distributions in a 3-column grid layout.
    """
    num_plots = len(list_of_cols)
    cols_per_row = 3
    rows = (num_plots // cols_per_row) + (num_plots % cols_per_row > 0)

    # Create subplots
    fig = sp.make_subplots(rows=rows, cols=cols_per_row, subplot_titles=list_of_cols)

    for i, col in enumerate(list_of_cols):
        row = i // cols_per_row + 1
        col_num = i % cols_per_row + 1
        histogram = px.histogram(df, x=col)
        for trace in histogram.data:
            fig.add_trace(trace, row=row, col=col_num)

    fig.update_layout(title="Column Distributions", height=300 * rows)
    fig.show()
        
def qq_plot(df, list_of_cols):
    num_plots = len(list_of_cols)
    cols_per_row = 3
    rows = (num_plots // cols_per_row) + (num_plots % cols_per_row > 0)

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))

    # Flatten axes if multiple rows, otherwise, keep as a list
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, column in enumerate(list_of_cols):
        stats.probplot(df[column], dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot for {column}")

    # Hide unused subplots if there are extra spaces
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
        
def cap_outliers_zScore(data, column):
    """
    
    """
    upper_limit = data[column].mean() + 3*data[column].std()
    lower_limit = data[column].mean() - 3*data[column].std()
    data[column] = np.where(
        data[column] > upper_limit,
        upper_limit,
        np.where(
            data[column] < lower_limit,
            lower_limit,
            data[column]
        )
    )
    return data

def cap_outliers_iqr(data, column):
    """
    Given a dataset and a column name, caps outliers to the 
    upper and lower limits using IQR method.
    
    Can be used for skewed and bimodal* distributions.
    """

    
    percentile25 = data[column].quantile(0.25)
    percentile75 = data[column].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr

    data[column] = np.where(
        data[column] > upper_limit,
        upper_limit,
        np.where(
            data[column] < lower_limit,
            lower_limit,
            data[column]
        )
    )

    return data

def remove_outliers_zScore(data, column):
    """
    Given a dataset and a column name, removes outliers
    using zScore method.
    
    Can be used for normal distributions.
    """
    upper_limit = data[column].mean() + 3*data[column].std()
    lower_limit = data[column].mean() - 3*data[column].std()
    data = data[data[column] < upper_limit]
    data = data[data[column] > lower_limit]
    return data

def remove_outliers_iqr(data, column):
    """
    Given a dataset and a column name, removes
    outliers using IQR method.
    
    Can be used for skewed and bimodal* distributions.
    """
    percentile25 = data[column].quantile(0.25)
    percentile75 = data[column].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    data = data[data[column] < upper_limit]
    data = data[data[column] > lower_limit]

    return data


def fill_mean(df, column):
    """
    Given a df and a column, fills nulls in the column with the mean
    """
    
    df[column].fillna(df[column].mean(), inplace=True)
    
    return df


def fill_median(df, column):
    """
    Given a df and a column, fills nulls in the column with the median
    """
    
    df[column].fillna(df[column].median(), inplace=True)
    
    return df


def fill_mode(df, column):
    """
    Given a df and a column, fills nulls in the column with the mode
    """
    
    df.loc[:, column] = df[column].fillna(df[column].mode().iloc[0])
    
    return df

def calculate_stats(tn, fp, fn, tp):
    """
    Given the four parts of the matrix:
        tn = true negative
        fp = false positive
        fn = false negative
        tp = true positive,
    returns the four accuracy rates of the individual parts
    e.g. tn rate = tn/total cases
    """
    total = tn + fp + fn + tp
    tn_perc = round((tn/total) * 100, 2)
    fp_perc = round((fp/total) * 100, 2)
    fn_perc = round((fn/total) * 100, 2)
    tp_perc = round((tp/total) * 100, 2)
    
    print('False Negative %: ' + str(fn_perc))
    print('False Positive %: ' + str(fp_perc))
    print('True Negative %: ' + str(tn_perc))
    print('True Positive %: ' + str(tp_perc))
    
    return

def set_threshold(y_pred_proba, threshold=0.24):
    """
    Given a threshold and prediction probabilities,
    transforms the predictions to adhere to the threshold
    """
    return (y_pred_proba[:, 1] > threshold).astype(int)

def calculate_col_percentage(df, column):
    return (df[column] / df[column].sum())*100