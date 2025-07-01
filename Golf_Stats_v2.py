import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st
from streamlit_plotly_events import plotly_events  # Ensure this is installed: pip install streamlit-plotly-events
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import statsmodels.api as sm
import matplotlib.image as mpimg
from scipy.stats import chi2
import unicodedata

# Force the default template to "plotly"
pio.templates.default = "plotly"

sns.set_theme(style="darkgrid")

#-----------------------  has to be called first
# Set the page layout to wide
st.set_page_config(layout="wide")

# TO RUN THIS, USE TERMINAL
#                        streamlit run C:\Users\dmomb\OneDrive\Python\Projects\GolfDec24\Golf_Stats_v2.py
# Turn on 'Always re-run' option on your app (in the web browser), then every time you save code changes, they'll automatically show in the app

# File and folder path
fol = ''
#fol = 'C:/Users/dmomb/OneDrive/Documents/Golf/'
fn = 'FS_Golf_DB.xlsx'

# Caching data loading
@st.cache_data
def load_data(filepath):
    return pd.read_excel(filepath, engine='openpyxl')

df = load_data(fol + fn)

#### Helper functions etc  ################################################################
numcols = [
    'Ball_mph', 'Club_mph', 'Smash_Factor', 'Carry_yds', 'Total_yds', 'Roll_yds',
    'Swing_H', 'Spin_rpm', 'Height_ft', 'Time_s', 'AOA', 'Spin_Loft', 'Swing_V', 
    'Spin_Axis', 'Lateral_yds', 'FTP', 'FTT', 'Dynamic_Loft', 'Club_Path', 'Launch_H',
    'Launch_V', 'Low_Point_ftin', 'DescentV', 'Curve_Dist_yds', 'Lateral_Impact_in', 'Vertical_Impact_in'
]

col_display = [
    'Club', 'Time', 'Golfer', 'Shot_Type', 'Carry_yds', 'Total_yds', 'Roll_yds', 'Lateral_yds',
    'Club_mph', 'Ball_mph', 'Smash_Factor', 'Swing_H', 'Swing_V', 'AOA', 'FTP', 'FTT', 'Low_Point_ftin',
    'Lateral_Impact_in', 'Vertical_Impact_in', 'Spin_rpm', 'Spin_Axis'
]

clubs = [
    'Driver', '3 Wood', '5 Wood', '4 Iron', '5 Iron', '6 Iron', '7 Iron',
    '8 Iron', '9 Iron', 'Pitching Wedge', 'Gap Wedge', 'Sand Wedge', 'Lob Wedge'
]

# Ensure numeric conversion explicitly
def ensure_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.error(f"Column '{col}' not found in the dataframe.")

# Convert directional values with error handling
def convert_value(value):
    try:
        import unicodedata
        value = str(value).strip()
        value = unicodedata.normalize("NFKC", value)  # Normalize Unicode
 
        value = value.replace("\xa0", "").replace("\u200b", "").replace(" ", "")  # Remove non-breaking spaces & zero-width spaces

        if len(value) < 2 or not value[-1].isalpha():
            return None  # Ensure valid format before processing
        
        number, direction = value[:-1], value[-1].upper()
        number = float(number)
        return -number if direction == 'R' else number if direction == 'L' else None
    except (ValueError, AttributeError):
        return None

# Convert specified columns that contain directional indicators
def convert_directional_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_value)
        else:
            st.error(f"Column '{col}' is missing from the data.")
    return df


# Clean column names function
def clean_column_names(df):
    df.columns = (
        df.columns
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace('\xa0', ' ')
        .str.strip()
        .str.replace(' ', '_')
    )
# Add Smash_Factor_catgeory column
bins = [0, 1.0, 1.1, 1.2, 1.3, 1.4, float('inf')]  # Define the bins
labels = ['<1.0', '1.0–1.1', '1.1–1.2', '1.2–1.3', '1.3–1.4', '>1.4']
# Create the categorical variable
df['Smash_Factor_Category'] = pd.cut(df['Smash_Factor'], bins=bins, labels=labels, right=False)
# Define ordered categories
sf_cat_type = CategoricalDtype(categories=['<1.0', '1.0–1.1', '1.1–1.2', '1.2–1.3', '1.3–1.4', '>1.4'],ordered=True)
# Assign the categorical type to the column
df['Smash_Factor_Category'] = df['Smash_Factor_Category'].astype(sf_cat_type)

# Main processing function
def process_df(df,numcols=numcols):
    clean_column_names(df)

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        
        # Remove NaT (null) values before using .dt.strftime()
        df = df.dropna(subset=['Time'])

        df['Session'] = df['Time'].dt.strftime('%Y %b %d %I:%M %p')

        # Directly sort by 'Time'
        df = df.sort_values('Time').copy()

        # Remove NaNs from unique session list before using in pd.Categorical
        sorted_sessions = [s for s in df['Session'].unique() if pd.notna(s)]
        df['Session'] = pd.Categorical(df['Session'], categories=sorted_sessions, ordered=True)
    else:
        st.error("Column 'Time' is missing from the data.")

    directional_cols = ['Swing_H', 'Spin_Axis', 'Lateral_yds', 'FTP', 'FTT', 'Club_Path', 'Launch_H']
    df = convert_directional_columns(df, directional_cols)

    ensure_numeric(df, numcols)
    
    if 'Shot_Type' in df.columns:
        df['Shot_Type'] = df['Shot_Type'].astype(str)
    else:
        st.error("Column 'Shot_Type' is missing from the data.")
      
    # Pivot with Sessions
    df_pivot = df.pivot_table(
        index='Club',
        columns=['Golfer', 'Session'],
        values=numcols, 
        aggfunc='mean',
        observed=True  # Avoid future warnings from pandas)
    )
    # Reindex columns to ensure Session is in ascending chronological order
    df_sessions = df_pivot.reindex(columns=sorted_sessions, level='Session')

    df_golfer = df.pivot_table(
        index='Club',
        columns=['Golfer'],
        values=numcols, 
        aggfunc='mean')
    df_golfer = df_golfer.reindex(clubs)       # Puts the clubs in the order of the list 'clubs'
    df_sessions = df_sessions.reindex(clubs)   # Puts the clubs in the order of the list 'clubs'
    return df,df_sessions,df_golfer

########################################################   PROCESSING COMPLETE ##########################################
#################    PROCESS THE DF FILE.  CALC AVGS BY SESSION AND GOLFER TOTAL AFTER FILTERING ##############################

df, df_sessions, df_golfer = process_df(df)       # #  Club in rows, Golfer, Session in Columns as MultiIndex  (all metrics)

# Calculate counts grouped by Golfer and Club
shot_counts = df.groupby(['Golfer', 'Club'])['Carry_yds'].count().reset_index()
shot_counts.rename(columns={'Carry_yds': 'Shot_Count'}, inplace=True)
# Pivot table for display
counts_golfer = shot_counts.pivot(index='Club', columns='Golfer', values='Shot_Count')

dfall = df.copy()

# Sidebar Setup -------------------------------------------------------------------------------------------
# Sidebar helper functions

def return_filtered_df(df, col, search_term):
    if search_term != "All":
        df = df[df[col] == search_term].copy()
    return df
def remove_outliers_iqr(group, cols):
    """
    Given a DataFrame 'group' (already filtered by a particular
    Golfer, Session, Club), remove outliers in each column in 'cols'
    using the 1.5*IQR rule.
    Returns the group without outliers.
    """
    for col in cols:
        if col not in group.columns:
            # skip if the col doesn't exist in group
            continue

        # Only consider valid (non-NaN) data
        col_data = group[col].dropna()
        if col_data.empty:
            # no data for this group in this column
            continue

        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Create a boolean mask for outliers
        outlier_mask = (group[col] < lower_bound) | (group[col] > upper_bound)

        # Remove outliers by setting them to NaN or dropping them
        # Option A: If you want to *drop* rows with outliers:
        group = group[~outlier_mask]

        # Option B (Alternative):
        # If you want to keep the row but set the value to NaN:
        # group.loc[outlier_mask, col] = float('NaN')

    return group
# 2. Apply the function groupwise to calculate the df_no_outliers on any column
df_no_outliers = (
    df
    .groupby(['Golfer', 'Session', 'Club'], group_keys=False,observed=False)
    .apply(lambda grp: remove_outliers_iqr(grp, numcols))
)
def remove_outliers_iqr_single_column(group, column):
    """
    Remove outliers based on the IQR method for a single column.
    Returns the group without outliers for the specified column.
    """
    if column not in group.columns:
        # Skip if the column doesn't exist in the group
        st.error(f"Column '{column}' not found in the dataframe.")
        return group

    # Only consider valid (non-NaN) data
    col_data = group[column].dropna()
    if col_data.empty:
        # No data for this group in this column
        return group

    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter rows based on the outlier condition for the specific column
    return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

st.sidebar.title("Filter Shots")
st.sidebar.write("Choose Desired Options")

filter_choice = st.sidebar.selectbox('Choose filtering of Data', ['All', 'IQR_All', 'IQR_Carry_yds'])

if filter_choice == 'All':
    df = dfall.copy()
elif filter_choice == 'IQR_All':
    df = df_no_outliers.copy()
elif filter_choice == 'IQR_Carry_yds':
    # Apply IQR filtering only on the 'Carry_yds' column, starting with the full dataset
    df = remove_outliers_iqr_single_column(dfall, 'Carry_yds')

####  THIS SECTION FILTERS THE DATA BASED ON THE SIDEBAR SELECTIONS  ############################################
col = 'Time'
choices = ['All'] + df[col].unique().tolist()
sb_time = st.sidebar.selectbox('Select ' + col, choices)
df = return_filtered_df(df, col, sb_time)

col = 'Golfer'
choices = ['All'] + dfall[col].unique().tolist()
sb_golfer = st.sidebar.selectbox('Select ' + col, choices)
df = return_filtered_df(df, col, sb_golfer)

col = 'Club'
choices = ['All'] + df[col].unique().tolist()
sb_club = st.sidebar.selectbox('Select ' + col, choices)
df = return_filtered_df(df, col, sb_club)

# Dark grey line separator
st.sidebar.markdown("<hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Choose what to color on 
choices = ['Time', 'Golfer', 'Club', 'Shot_Type','Smash_Factor_Category']
color_on = st.sidebar.selectbox('Select ColorOn', choices)
########### df is now the filtered data on the code below   #######################################################################


#----------------------------------------------------------------------------------------------------------------
hov_data = ['Time', 'Club', 'Golfer', 'Shot_Type','Total_yds']





###### Golf Analysis and Plots ##########################################################
df['Shot_Type'] = df['Shot_Type'].astype(str)

def calculate_confidence_radius(confidence_percent):
    """Calculate confidence radius for any confidence percentage."""
    alpha = 1 - (confidence_percent / 100)
    chi2_val = chi2.ppf(1 - alpha, df=2)  # Chi-squared value with 2 degrees of freedom
    return np.sqrt(chi2_val)

#######################  FIGURES FOR TAB1 ##########################################################################################
##### fig1 #####
def create_fig1(df,x_choice):
    # print(f"Total rows before dropna: {len(df)}")
    # print(f"NaN count in {x_choice}: {df[x_choice].isna().sum()}")
    # print(f"NaN count in Lateral_yds: {df['Lateral_yds'].isna().sum()}")    
    x_max = df[x_choice].max() * 1.25
    max_abs_y = max(abs(df["Lateral_yds"].min()), abs(df["Lateral_yds"].max()))
    scale_fac = 1.5
    y_min, y_max = -max_abs_y*scale_fac, max_abs_y*scale_fac

    fig1 = px.scatter(df, x=x_choice, y='Lateral_yds', color=color_on, title="Dispersion Field", color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)
    fig1.update_xaxes(range=[0, x_max])
    fig1.update_yaxes(range=[y_min, y_max])
    fig1.update_layout(yaxis_scaleanchor="x")  # Makes the scale correct so x & y distances are the same on the plot

    # Add confidence ellipse to fig1
    x = df[x_choice].dropna()
    y = df['Lateral_yds'].dropna()
    if len(x) > 1 and len(y) > 1:
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)

        theta = np.degrees(np.arctan2(*v[:, 0][::-1]))
        ellipse_x = np.linspace(0, 2 * np.pi, 100)
        # np.sqrt(2.71) corrresponds to 90% confidence, np.sqrt(5.99) corresponds to 95% confidence,np.sqrt(1.64) corresponds to 80% confidence
        confidence_radius = calculate_confidence_radius(80)    #######  SET CONFIDENCE RADIUS HERE in %  ########
        ellipse_coords = np.array([2 * np.sqrt(confidence_radius) * lambda_[0] * np.cos(ellipse_x),
                                    2 * np.sqrt(confidence_radius) * lambda_[1] * np.sin(ellipse_x)])
        rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                    [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        ellipse_coords = rotation_matrix @ ellipse_coords
        ellipse_coords[0] += x.mean()
        ellipse_coords[1] += y.mean()

        fig1.add_trace(go.Scatter(x=ellipse_coords[0], y=ellipse_coords[1], mode='lines', name='80% CI Ellipse',
                                line=dict(color='red')))
        ##############
        confidence_radius = calculate_confidence_radius(1)    #######  SET CONFIDENCE RADIUS HERE in %  ########
        ellipse_coords = np.array([2 * np.sqrt(confidence_radius) * lambda_[0] * np.cos(ellipse_x),
                                    2 * np.sqrt(confidence_radius) * lambda_[1] * np.sin(ellipse_x)])
        rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                    [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        ellipse_coords = rotation_matrix @ ellipse_coords
        ellipse_coords[0] += x.mean()
        ellipse_coords[1] += y.mean()

        fig1.add_trace(go.Scatter(x=ellipse_coords[0], y=ellipse_coords[1], mode='lines', name='1% CI Ellipse',
                                line=dict(color='red', dash='dash')))
    return fig1

##### fig2 #####

fig2 = px.scatter(df, x="Carry_yds", y='Height_ft', color=color_on, title="Height vs Carry (yds)", color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)

##### fig3 #####


fig3 = px.scatter(df, x='Club_mph', y='Ball_mph', color=color_on, title="Smash Factor by 'Color'",
                  color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)

##### fig4 #####
df['AOA'] = pd.to_numeric(df['AOA'], errors='coerce')

fig4 = px.scatter(df, x='AOA', y='Height_ft', color=color_on, title="Height of Ball Flight vs Angle of Attack",
                  color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)

##### fig5 #####
# xvar_choice = 'Roll_yds'
# yvar_choice = 'Spin_rpm'
# df[xvar_choice] = pd.to_numeric(df[xvar_choice], errors='coerce')

# fig5 = px.scatter(df, x=xvar_choice, y=yvar_choice, color=color_on, title="Roll vs Spin",
#                   color_discrete_sequence=px.colors.qualitative.Bold, hover_data=['Shot_Type', 'Club'])
# fig5.update_yaxes(range=[0, None])
# fig5.update_layout(
#     xaxis=dict(showline=True, mirror=True, linecolor='black'),
#     yaxis=dict(showline=True, mirror=True, linecolor='black')
# )
df['Smash_Factor'] = pd.to_numeric(df['Smash_Factor'], errors='coerce')
df['Carry_yds'] = pd.to_numeric(df['Carry_yds'], errors='coerce')

fig5,ax5 = plt.subplots()
sns.scatterplot(data=df,x='Smash_Factor',y='Carry_yds',hue=color_on,ax=ax5)


#########################  FIGURES FOR TAB2 ######################################################################

def create_fig6(df, var_choice, colvar, range_mode='normal'):
    """
    Creates a box plot for the selected variable over different sessions, with the option 
    to color by a categorical variable and display median annotations.

    Parameters:
    df (pd.DataFrame): The dataset containing session data.
    var_choice (str): The numerical variable to be plotted on the y-axis.
    colvar (str): The categorical variable used for coloring the box plot.
    range_mode (str, optional): The y-axis range mode. Defaults to 'normal'.
        - 'normal': Allows natural scaling.
        - 'tozero': Forces the y-axis to start at 0.

    Returns:
    plotly.graph_objects.Figure: A box plot with median annotations.
    """
    # Drop NA values only for the selected variable and 'Session'
    df_filtered = df.dropna(subset=['Session', var_choice])
    
    # Convert 'Session' to an ordered categorical to maintain chronological order
    sorted_sessions = df_filtered['Session'].dropna().unique()
    df_filtered['Session'] = pd.Categorical(
        df_filtered['Session'],
        categories=sorted_sessions,
        ordered=True
    )
    
    # Create a box plot
    fig6 = px.box(df_filtered, x='Session', y=var_choice, points='all', color=colvar, hover_data=hov_data)
    
    # Compute median values for annotation
    mean_values = df_filtered.groupby('Session', observed=True)[var_choice].median().reset_index()
    
    # Add the median as text annotations
    for i, row in mean_values.iterrows():
        fig6.add_annotation(
            x=row['Session'],
            y=row[var_choice],
            text=f"{row[var_choice]:.1f}",
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor="white"
        )
    
    # Ensure the y-axis follows the correct range setting
    if range_mode == 'tozero':
        min_y = 0
    else:
        min_y = df_filtered[var_choice].min()
    max_y = df_filtered[var_choice].max()
    
    fig6.update_layout(yaxis=dict(range=[min_y, max_y]))
    
    return fig6

###################################################################################################################
def create_fig8(df, fig8_type='kde', background_image_path=None, image_scale=1.5):  # Image is the golf club
    if fig8_type in ['kde', 'scatter', 'hex', 'reg', 'resid', 'hist']:
        num_bins = 5

        if 'Total_yds' not in df.columns:
            raise ValueError("Total_yds column is missing from the DataFrame.")

        df['Total_yds_bin_label'] = np.nan

        if 'Club' in df.columns:
            for club, group in df.groupby('Club'):
                try:
                    _, bin_edges = pd.qcut(group['Total_yds'], num_bins, retbins=True, duplicates='drop')
                    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])} yds" for i in range(len(bin_edges)-1)]
                    df.loc[group.index, 'Total_yds_bin_label'] = pd.cut(
                        group['Total_yds'], bins=bin_edges, labels=bin_labels, include_lowest=True
                    )
                except ValueError:
                    df.loc[group.index, 'Total_yds_bin_label'] = "N/A"
        else:
            _, bin_edges = pd.qcut(df['Total_yds'], num_bins, retbins=True, duplicates='drop')
            bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])} yds" for i in range(len(bin_edges)-1)]
            df['Total_yds_bin_label'] = pd.cut(df['Total_yds'], bins=bin_edges, labels=bin_labels, include_lowest=True)

        cmap = plt.get_cmap("viridis", num_bins)

        g = sns.jointplot(
            data=df,
            x='Lateral_Impact_in',
            y='Vertical_Impact_in',
            kind="scatter",
            height=4
        )
        # Ensure square aspect ratio for x and y axes
        g.ax_joint.set_aspect('equal')
        g.ax_joint.set_xlim(-2, 2)
        g.ax_joint.invert_xaxis()
        background_image_path = 'golf_club.jpg'
        if background_image_path:
            img = mpimg.imread(background_image_path)

            width = 4 * image_scale
            height =  3 * image_scale  #(g.ax_joint.get_ylim()[1] - g.ax_joint.get_ylim()[0]) * 1.5 * image_scale
            x_center = 0.75  # Shift image left by approximately 1/8 of the frame
            y_center = 0.75 +(g.ax_joint.get_ylim()[0] + g.ax_joint.get_ylim()[1]) / 2
            g.ax_joint.imshow(img, aspect='auto',
                              extent=[x_center - width/2, x_center + width/2,
                                      y_center - height/2, y_center + height/2],
                              origin='upper', alpha=0.3, zorder=0)

        if fig8_type == 'kde':
            g.plot_joint(sns.kdeplot, fill=True, levels=20)
            scatter = g.plot_joint(
                sns.scatterplot,
                hue=df['Total_yds_bin_label'],
                palette=cmap.colors,
                alpha=0.6
            )
            legend = g.ax_joint.legend(title="Total Yards")
            for text in legend.get_texts():
                text.set_fontsize(8)

        g.figure.set_size_inches(6, 4)
        return g.figure
import plotly.express as px

def create_bar_chart(df, session, y_variable):
    """
    Creates a bar chart for a given session showing the selected y_variable,
    ordered by Club and Shot sequence.

    Parameters:
    - df: DataFrame containing golf shot data
    - session: Selected session to filter the data
    - y_variable: The variable to plot (e.g., 'Total_yds', 'Carry_yds', 'Ball_mph')

    Returns:
    - A Plotly figure object
    """

    # Filter data for the selected session
    df_session = df[df['Session'] == session].copy()

    if df_session.empty:
        st.error(f"No data available for session: {session}")
        return None

    # Ensure club order is maintained
    df_session['Club'] = pd.Categorical(df_session['Club'], categories=clubs, ordered=True)

    # Sort by Club and then by Shot sequence
    df_session = df_session.sort_values(['Club', 'Shot'])

        # Define columns to show in hover tooltip
    hover_columns = ['Shot_Type','Club_mph', 'Smash_Factor', 'AOA', 'Spin_Loft', 'Swing_V', 'FTP',
                     'Dynamic_Loft', 'Club_Path', 'Launch_V', 'Low_Point_ftin',
                     'Lateral_Impact_in', 'Vertical_Impact_in']

    # Create bar chart
    fig = px.bar(df_session, x='Shot', y=y_variable, title=f"{y_variable} for {session}",
                 color='Club', text=y_variable, height=500, facet_col='Club', facet_col_wrap=4,
                 hover_data=hover_columns)

    # Format bars
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')

    # Improve layout
    fig.update_layout(xaxis_title="Shot Sequence", yaxis_title=y_variable, showlegend=True)

    return fig

# Function for the fixed reference bar chart
def create_fixed_bar_chart(df, session, reference_variable="Total_yds"):
    """
    Creates a bar chart for Total Yds or Carry Yds for a given session,
    always displayed beneath the user-selected variable chart.
    """

    df_session = df[df['Session'] == session].copy()

    if df_session.empty:
        return None

    df_session['Club'] = pd.Categorical(df_session['Club'], categories=clubs, ordered=True)
    df_session = df_session.sort_values(['Club', 'Shot'])
    # Ensure Smash_Factor_Category has all categories
    all_cats = df['Smash_Factor_Category'].cat.categories
    present_cats = df_session['Smash_Factor_Category'].dropna().unique()
    # Add dummy rows for missing categories ##################
    missing_cats = [cat for cat in all_cats if cat not in present_cats] 
    dummy_rows = pd.DataFrame({
    'Shot': [np.nan] * len(missing_cats),
    'Club': [df_session['Club'].iloc[0]] * len(missing_cats),  # Assign an existing Club just to avoid error
    'Session': [session] * len(missing_cats),
    'Smash_Factor_Category': pd.Categorical(missing_cats, categories=all_cats, ordered=True),
    'Total_yds': [np.nan] * len(missing_cats)  # Or whatever reference_variable is
    })
    # Append and reset
    df_session = pd.concat([df_session, dummy_rows], ignore_index=True)
    ###########################################################
    df_session['Smash_Factor_Category'] = df_session['Smash_Factor_Category'].cat.set_categories(all_cats)

    # Explicit color mapping for Smash_Factor_Category with red as highest
    color_discrete_map = {
        '<1.0': 'indigo',
        '1.0–1.1': 'blue',
        '1.1–1.2': 'green',
        '1.2–1.3': 'yellow',
        '1.3–1.4': 'orange',
        '>1.4': 'red'
    }
    
    fig = px.bar(
        df_session, 
        x='Shot', 
        y=reference_variable, 
        title=f"{reference_variable} for {session}",
        color='Smash_Factor_Category', 
        text=reference_variable, 
        height=400,  
        facet_col='Club', 
        facet_col_wrap=4,
        color_discrete_map=color_discrete_map
    )

    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Shot Sequence",
        yaxis_title=reference_variable,
        showlegend=True,
        legend_title_text='',  # Remove legend title
        margin=dict(t=60, b=40, l=20, r=20)
    )

    # Clean facet labels (e.g., remove "Club=")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def make_stacked_bar_charts(
    df, session, xvar, yvar_list, yvar_fixed,
    facet_col='Club', facet_color=None, fixed_color=None,
    category_order=None, color_map=None
):
    df_session = df[df['Session'] == session].copy()

    if df_session.empty:
        return None

    # Ensure consistent facet column ordering
    if category_order:
        df_session[facet_col] = pd.Categorical(df_session[facet_col], categories=category_order, ordered=True)
    df_session = df_session.sort_values([facet_col, xvar])

    num_rows = len(yvar_list) + 1  # Each yvar gets a row + 1 for fixed
    clubs = df_session[facet_col].unique()

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=yvar_list + [yvar_fixed]
    )

    for i, yvar in enumerate(yvar_list, start=1):
        for club in clubs:
            club_df = df_session[df_session[facet_col] == club]
            fig.add_trace(
                go.Bar(
                    x=club_df[xvar],
                    y=club_df[yvar],
                    name=str(club),
                    marker_color=color_map.get(club) if color_map else None,
                    showlegend=(i == 1)
                ),
                row=i,
                col=1
            )

    # Add the fixed reference variable
    for cat in df_session[facet_col].unique():
        club_df = df_session[df_session[facet_col] == cat]
        fig.add_trace(
            go.Bar(
                x=club_df[xvar],
                y=club_df[yvar_fixed],
                name=str(cat),
                marker_color=fixed_color.get(club_df['Smash_Factor_Category'].iloc[0]) if fixed_color else None,
                showlegend=False
            ),
            row=num_rows,
            col=1
        )

    fig.update_layout(
        height=300 * num_rows,
        barmode='group',
        showlegend=True,
        title_text=f"Session Comparison for {session}",
        margin=dict(t=60, b=40)
    )

    return fig



#######################################################################################################################

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9,tab10,tab11 = st.tabs(["4 Plots", "BoxPlots","Stats","Plotchoice","Seaborn","Parameters","Distances","Impact","All Plots","Selected Data","Shots in Order"])

with tab1:                                                                          ## TAB 1 4 Plots  ##
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        #st.write("Swing_H,Spin_Axis,Lateral_yds,FTP,FTT,Club_Path,Launch_H")
        st.plotly_chart(create_fig1(df,"Carry_yds"), use_container_width=True, key="T1C1R1")
    with row1_col2:
        #st.write("Title: Col 2, Row 1")
        st.plotly_chart(fig2, use_container_width=True, key="T1C2R1")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        #st.write("Title: Col 1, Row 2")
        st.plotly_chart(fig5, use_container_width=True, key="T1C1R2")
    with row2_col2:
        #st.write("Title: Col 2, Row 2")
        st.plotly_chart(fig3, use_container_width=True, key="T1C2R2")

with tab2:                                                                          ## TAB 2 BoxPlots ###################
    with st.container():

        row1 = st.columns([1,6])

        with row1[0]:
            disp_x = st.selectbox('Select X axis', ["Carry_yds","Total_yds"])

        with row1[1]:
            st.plotly_chart(create_fig1(df,disp_x), use_container_width=True, key="T2C1R1")

        bottom_row = st.columns([1,6])

        with bottom_row[0]:
            boxplot_metric = st.selectbox('Select Metric for Boxplot', numcols)
            # Change the range of the y-axis
            range_choice = st.selectbox('Select Y-axis Range', ['normal', 'tozero'])

        with bottom_row[1]:
            fig6 = create_fig6(df,boxplot_metric,color_on,range_choice)

            st.write("Box Plot for "+boxplot_metric)
            st.plotly_chart(fig6, use_container_width=True, key="T2C3R3")


with tab3:                                                                          ## TAB 3 Stats #####################
    col1_3, col2_3, col3_3 = st.columns(3)
    with col1_3:
        st.write("### Average Carry Yds")
        st.dataframe(df_golfer['Carry_yds'].round(1),height=600)
    with col2_3:
        st.write("### Average Total Yds")
        st.dataframe(df_golfer['Total_yds'].round(1),height=600)
    with col3_3:
        st.write("### Shot Counts")
        st.dataframe(counts_golfer,height=600)

with tab4:  # TAB 4 Plotchoice
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Use larger text with st.markdown
    st.markdown("""
        <h2 style="font-size:22px; font-weight:bold;">
        For Left/ Right Variables : Swing_H, Spin_Axis, Lateral_yds, FTP, FTT, Club_Path, Launch_H  
        Left is +ve, Right is -ve
        </h2>
    """, unsafe_allow_html=True)

    with col1:
        ycol = st.selectbox('Y-axis column', numcols, index=4)
    with col2:
        xcol = st.selectbox('X-axis column', numcols, index=1)
    with col3:
        color_on2 = st.selectbox('Color By', ['Time', 'Golfer', 'Club', 'Shot_Type'])
    with col4:
        filter_col = st.selectbox('Filter Column', numcols, index=0)  # Select numeric column to filter
    with col5:
        min_val, max_val = st.slider(f'Select {filter_col} range', 
                                     float(df[filter_col].min()), 
                                     float(df[filter_col].max()), 
                                     (float(df[filter_col].min()), float(df[filter_col].max())))

    # Apply filtering to the data
    filtered_df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]

    row2 = st.columns([1, 6])
    with row2[0]:
        st.write("#### Dates Shown")
        df_unique_times = pd.DataFrame(filtered_df['Time'].unique(), columns=[''])  
        st.dataframe(df_unique_times, height=600)

    with row2[1]:
        # Scatter plot with filtered data
        fig7 = px.scatter(filtered_df, x=xcol, y=ycol, color=color_on2, 
                          title=f"{ycol} vs {xcol} (Filtered by {filter_col})",
                          color_discrete_sequence=px.colors.qualitative.Bold, 
                          hover_data=hov_data, trendline='ols')

        st.plotly_chart(fig7, use_container_width=True, key="T4C1R1")

with tab5:                                                                          ## TAB 5  Seaborn ####################
    ###################################### Seaborn Statistical Plots ###################################
    df['Smash_Factor'] = pd.to_numeric(df['Smash_Factor'], errors='coerce')
    df['Carry_yds'] = pd.to_numeric(df['Carry_yds'], errors='coerce')

    # fig8,ax5 = plt.subplots(figsize=(10, 5))  # Adjust figsize as needed)
    # sns.scatterplot(data=df,x='Smash_Factor',y='Carry_yds',hue=color_on,ax=ax5)
    # st.pyplot(fig8)


    col_a,col_b = st.columns(2)

    with col_a:
        # Create an lmplot (no ax parameter)
        
        sns_plot = sns.lmplot(data=df, x='Smash_Factor', y='Carry_yds', hue=color_on, height=4, aspect=2)
        # Display the lmplot in Streamlit
        st.pyplot(sns_plot.figure) 
    with col_b:
        sns_plot = sns.relplot(data=df, x='Club_mph', y='Carry_yds',size='Smash_Factor',hue='Golfer',style='Time',height=4, aspect=2)
        st.pyplot(sns_plot.figure)     

    col_c,col_d = st.columns(2)

    with col_c:
            # Create 2 columns for the inputs to be contained
            colc1, colc2 = st.columns(2)
            with colc1:
                # Select which variable to plot
                xcol2 = st.selectbox('Select column for distribution', numcols)
            with colc2:
                plotchoice = st.selectbox('Select type of plot', ['histogram','kernel density'])
            if plotchoice == 'histogram':
                plot_type = 'hist'
            else:
                plot_type = 'kde'

            sns_plot = sns.displot(data=df, x=xcol2, hue='Golfer',kind=plot_type,fill=True, height=4, aspect=2)
            # Display the lmplot in Streamlit
            st.pyplot(sns_plot.figure) 
    with col_d:
        sns_plot = sns.displot(data=df, x='Carry_yds', hue='Golfer',kind='kde',fill=True, height=4, aspect=2)
        st.pyplot(sns_plot.figure)     

with tab6:                                                                          ## TAB 6 Parameters (Description) ##
        ##########################  Images that show the Mevo+ Data Parameters and their Meaning  ###################################
        #  Assumes these images are in the top folder
        st.write("# Flightscope Mevo+ Club and Ball Parameters")
        st.write("### NOTE: Left is positive, Right is Negative when looking at club path etc.")
        st.image("Club Data 1.jpg")
        st.image("Club Data 2.jpg")
        st.image("Ball Data 1.jpg")
        st.image("Ball Data 2.jpg")
        st.image("Ball Data 3.jpg")
        st.image("Ball-Flight.jpg")

with tab7:                                                                         ## TAB 7 Distances ##
    ####################### Line Plot for Golfer ######################################################################################
    st.write("### Carry Distance by Session for Specific Golfer")

    # Filter data for a specific golfer and metric
    col1_7, col2_7 = st.columns(2)
    with col1_7:
        metric_name = st.selectbox("Select Metric", numcols)
        metric_golfer_sessions = df_sessions[metric_name]   #  Club in rows, Golfer, Session in Columns as MultiIndex
    with col2_7:
        golfer_name = st.selectbox("Select Golfer", df['Golfer'].unique())    
        filtered_df = metric_golfer_sessions[golfer_name]

    # --- 2) Melt the DataFrame from wide to long ---
    df_melt = filtered_df.reset_index().melt(
        id_vars='Club',             # 'Club' stays as its own column
        var_name='Session',         # The old column names (session strings) become this column
        value_name='Metric'         # The cell values become 'Metric'
    )

    # --- 3) Parse Session to a datetime so we can plot chronologically
    df_melt['Session_dt'] = pd.to_datetime(
        df_melt['Session'],
        format='%Y %b %d %I:%M %p',
        errors='coerce'
    )

    # --- 4) Create the line plot using Plotly Express
    fig_golfer = px.line(
        data_frame=df_melt,
        x='Session_dt',   # time on the x-axis
        y='Metric',       # the numeric metric on the y-axis
        color='Club',     # one line per club
        markers=True      # optional: show markers for data points
    )
    # Force a known set of discrete colors
    color_discrete_sequence=px.colors.qualitative.Plotly

    # --- 5) Customize as needed ---
    fig_golfer.update_layout(
        xaxis_title='Session',
        yaxis_title='Metric',
        title='Metric by Club over Sessions'
    )
    fig_golfer.update_traces(connectgaps=True)
   
    st.plotly_chart(fig_golfer, use_container_width=True)

    with tab8:                                                                          ## TAB 8 Impact ##
        figchoices = ['scatter', 'hex', 'kde', 'contour', 'reg', 'resid', 'hist']

        with st.container():
            bottomrow = st.columns([4, 4])
            with bottomrow[0]:
                st.write("### Golf Ball Impact Data (on the clubface)")
                fig_to_plot = st.selectbox('Select Chart Type', figchoices, index=2)

            toprow = st.columns([4, 4])
            with toprow[0]:
                st.write("#### Vertical and Lateral Impact Data")
                fig8 = create_fig8(df, fig8_type=fig_to_plot)
                if fig_to_plot == 'contour':
                    st.plotly_chart(fig8)
                else:
                    st.pyplot(fig8)

with tab9:                                                                         ## TAB 9 All Plots ##
    # Select Y axis
    ycol2 = st.selectbox('Select Y axis', ["Carry_yds", "Total_yds"], key="xcol2_selectbox")

    # Loop over numcols in chunks of 3
    for i in range(0, len(numcols), 3):
        # Create a new row with 3 columns
        cols = st.columns(3)
        # Iterate over each column in the current chunk
        for j, xcol2 in enumerate(numcols[i:i+3]):
            # Skip if xcol2 is the same as ycol2
            if xcol2 == ycol2:
                continue  # Skip this iteration            
            # Create the scatter plot
            fig9 = px.scatter(
                df,
                x=xcol2,
                y=ycol2,
                color=color_on,
                title=f"{ycol2} versus {xcol2}",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hover_data=hov_data,
                trendline='ols',
                trendline_scope="overall"  # Ensures only one trendline across all data
            )
            # Set the chart height
            fig9.update_layout(height=400)
            # Place the chart in the appropriate column
            with cols[j]:
                st.plotly_chart(fig9, use_container_width=True, key="T9" + xcol2)

with tab10:                                                                      ## TAB 10 Selected Data ##         
    with st.container():
        disp_x = st.selectbox('Select X axis', ["Carry_yds", "Total_yds"], key="selectbox_tab10")
        
        fig1 = create_fig1(df,disp_x)
        
        try:
            selected_points = plotly_events(fig1, click_event=True, select_event=True, key="T10C1R1")
        except NameError:
            st.error("Error: plotly_events not found. Please install it using 'pip install streamlit-plotly-events'.")
            selected_points = []
        
    # Display Filtered Data Table Based on Selection
    if selected_points:
        selected_ids = [pt['pointIndex'] for pt in selected_points if 'pointIndex' in pt]
        if selected_ids:
            filtered_df = df.iloc[selected_ids]
            st.dataframe(filtered_df,column_order=col_display)
        else:
            st.write("No valid points selected.")
    else:
        st.write("Select points on the chart to view their details.")
# Columns Available: ['Mombo_ShotID', 'Club', 'Time', 'Golfer', 'Shot', 'Video', 'Ball mph','Club mph', 'Smash_Factor', 'Carry yds', 'Total yds', 'Roll yds',
#       'Swing H', 'Spin rpm', 'Height ft', 'Time s', 'AOA', 'Spin Loft','Swing V', 'Spin Axis', 'Lateral yds', 'Shot Type', 'FTP', 'FTT',
#       'Dynamic Loft', 'Club Path', 'Launch H', 'Launch V', 'Low Point ftin','DescentV', 'Curve Dist yds', 'Lateral Impact in', 'Vertical Impact in',
#       'Mode', 'Location', 'Unnamed_35', 'Unnamed_36', 'Unnamed_37','Unnamed_38', 'Unnamed_39', 'Unnamed_40', 'Comment', 'User1', 'User2','Exclude', 'Session']

with tab11:
    st.write("### Session Bar Chart")

    session_choice = st.selectbox("Select Session", df["Session"].unique())
    y_variable_choice = st.selectbox("Select Variable", numcols, index=4)

    fig_bar = create_bar_chart(df, session_choice, y_variable_choice)

    if fig_bar:
        st.plotly_chart(fig_bar, use_container_width=True)   
  
    st.markdown("---")  # Horizontal rule

    # Generate the fixed "Total Yds" or "Carry Yds" chart (bottom
    fig_ref = create_fixed_bar_chart(df, session_choice, reference_variable="Total_yds")  
    if fig_ref:
        st.plotly_chart(fig_ref, use_container_width=True)

    fig = make_stacked_bar_charts(
    df,
    session='2025 May 10 03:11 PM',
    xvar='Shot',
    yvar_list=['AOA'],
    yvar_fixed='Total_yds',
    facet_col='Club',
    category_order=clubs,
    color_map={
        '8 Iron': 'red',
        'Pitching Wedge': 'orange'
    },
    fixed_color={
        '<1.0': 'indigo',
        '1.0–1.1': 'blue',
        '1.1–1.2': 'green',
        '1.2–1.3': 'yellow',
        '1.3–1.4': 'orange',
        '>1.4': 'red'
    }
)

    st.plotly_chart(fig, use_container_width=True)