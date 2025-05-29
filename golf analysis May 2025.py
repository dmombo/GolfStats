import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pandas.api.types import CategoricalDtype
from scipy.stats import chi2

# TO RUN THIS, USE TERMINAL
#                        streamlit run C:\Users\dmomb\OneDrive\Python\Projects\GolfDec24\golf analysis May 2025.py

# Plotting configuration
pio.templates.default = "plotly"
st.set_page_config(layout="wide")

# File path configuration
DATA_FOLDER = ''
FILENAME = 'FS_Golf_DB.xlsx'

# Load data with caching
@st.cache_data
def load_data(filepath):
    return pd.read_excel(filepath, engine='openpyxl')

df = load_data(DATA_FOLDER + FILENAME)

# Constants
NUMERIC_COLS = [
    'Ball_mph', 'Club_mph', 'Smash_Factor', 'Carry_yds', 'Total_yds', 'Roll_yds',
    'Swing_H', 'Spin_rpm', 'Height_ft', 'Time_s', 'AOA', 'Spin_Loft', 'Swing_V',
    'Spin_Axis', 'Lateral_yds', 'FTP', 'FTT', 'Dynamic_Loft', 'Club_Path', 'Launch_H',
    'Launch_V', 'Low_Point_ftin', 'DescentV', 'Curve_Dist_yds', 'Lateral_Impact_in', 'Vertical_Impact_in'
]

CLUB_ORDER = [
    'Driver', '3 Wood', '5 Wood', '4 Iron', '5 Iron', '6 Iron', '7 Iron',
    '8 Iron', '9 Iron', 'Pitching Wedge', 'Gap Wedge', 'Sand Wedge', 'Lob Wedge'
]

# Cleaning and conversion functions
def ensure_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.error(f"Missing numeric column: {col}")

def convert_directional_value(val):
    try:
        val = str(val).strip()
        val = val.replace("\xa0", "").replace("\u200b", "").replace(" ", "")
        if len(val) < 2 or not val[-1].isalpha():
            return None
        number, direction = float(val[:-1]), val[-1].upper()
        return -number if direction == 'R' else number if direction == 'L' else None
    except:
        return None

def convert_directional_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_directional_value)
        else:
            st.error(f"Missing directional column: {col}")
    return df

def clean_column_names(df):
    df.columns = (df.columns
                  .str.replace(r'[^\w\s]', '', regex=True)
                  .str.replace('\xa0', ' ')
                  .str.strip()
                  .str.replace(' ', '_'))

# Process and enrich DataFrame
def process_df(df, numcols=NUMERIC_COLS):
    clean_column_names(df)

    if 'Time' not in df.columns:
        st.error("Missing 'Time' column.")
        return df, None, None

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.dropna(subset=['Time'], inplace=True)
    df.sort_values('Time', inplace=True)
    df['Session'] = df['Time'].dt.strftime('%Y %b %d %I:%M %p')
    df['Session'] = pd.Categorical(df['Session'], categories=sorted(df['Session'].unique()), ordered=True)

    df = convert_directional_columns(df, ['Swing_H', 'Spin_Axis', 'Lateral_yds', 'FTP', 'FTT', 'Club_Path', 'Launch_H'])
    ensure_numeric(df, numcols)

    if 'Shot_Type' not in df.columns:
        st.error("Missing 'Shot_Type' column.")
    else:
        df['Shot_Type'] = df['Shot_Type'].astype(str)

    # Add Smash Factor category
    bins = [0, 1.0, 1.1, 1.2, 1.3, 1.4, float('inf')]
    labels = ['<1.0', '1.0–1.1', '1.1–1.2', '1.2–1.3', '1.3–1.4', '>1.4']
    cat_type = CategoricalDtype(categories=labels, ordered=True)
    df['Smash_Factor_Category'] = pd.cut(df['Smash_Factor'], bins=bins, labels=labels, right=False).astype(cat_type)

    # Pivot by Golfer + Session and Golfer only
    df_sessions = df.pivot_table(index='Club', columns=['Golfer', 'Session'], values=numcols, aggfunc='mean', observed=True)
    df_sessions = df_sessions.reindex(CLUB_ORDER)
    df_golfer = df.pivot_table(index='Club', columns='Golfer', values=numcols, aggfunc='mean')
    df_golfer = df_golfer.reindex(CLUB_ORDER)

    return df, df_sessions, df_golfer

# Run processing
df, df_sessions, df_golfer = process_df(df)

# Sidebar filters
selected_golfer = st.sidebar.selectbox("Select Golfer", sorted(df['Golfer'].dropna().unique()))
sessions = df[df['Golfer'] == selected_golfer]['Session'].dropna().unique()
selected_session = st.sidebar.selectbox("Select Session", sorted(sessions))

# Filtered dataframe
filtered_df = df[(df['Golfer'] == selected_golfer) & (df['Session'] == selected_session)]

# Supporting data for plotting
color_on = 'Club'
hov_data = ['Time', 'Shot_Type', 'Club', 'Carry_yds', 'Lateral_yds']

# Confidence ellipse radius calculator
def calculate_confidence_radius(percent):
    return chi2.ppf(percent/100, df=2)

# Scatter plot with confidence ellipse
def create_fig1(df, x_choice):
    x_max = df[x_choice].max() * 1.25
    max_abs_y = max(abs(df["Lateral_yds"].min()), abs(df["Lateral_yds"].max()))
    scale_fac = 1.5
    y_min, y_max = -max_abs_y * scale_fac, max_abs_y * scale_fac

    fig1 = px.scatter(df, x=x_choice, y='Lateral_yds', color=color_on, title="Dispersion Field", color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)
    fig1.update_xaxes(range=[0, x_max])
    fig1.update_yaxes(range=[y_min, y_max])
    fig1.update_layout(yaxis_scaleanchor="x")

    x_vals = df[x_choice].dropna()
    y_vals = df['Lateral_yds'].dropna()
    if len(x_vals) > 1 and len(y_vals) > 1:
        cov = np.cov(x_vals, y_vals)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)

        theta = np.degrees(np.arctan2(*v[:, 0][::-1]))
        ellipse_x = np.linspace(0, 2 * np.pi, 100)

        for confidence in [80, 1]:
            radius = calculate_confidence_radius(confidence)
            ellipse_coords = np.array([2 * np.sqrt(radius) * lambda_[0] * np.cos(ellipse_x),
                                       2 * np.sqrt(radius) * lambda_[1] * np.sin(ellipse_x)])
            rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                        [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
            ellipse_coords = rotation_matrix @ ellipse_coords
            ellipse_coords[0] += x_vals.mean()
            ellipse_coords[1] += y_vals.mean()
            fig1.add_trace(go.Scatter(x=ellipse_coords[0], y=ellipse_coords[1], mode='lines',
                                      name=f'{confidence}% CI Ellipse',
                                      line=dict(color='red', dash='dash' if confidence == 1 else None)))
    return fig1

# Tab layout
with st.tabs(["Dispersion"])[0]:
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(create_fig1(filtered_df, "Carry_yds"), use_container_width=True, key="T1C1R1")

