import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pandas.api.types import CategoricalDtype
from scipy.stats import chi2
from io import BytesIO
from fpdf import FPDF
import os

# TO RUN THIS, USE TERMINAL
#                        streamlit run C:\Users\dmomb\OneDrive\Python\Projects\GolfDec24\golf_analysis_May_2025.py

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

    df['Time'] = pd.to_datetime(df['Time'], format="%Y%m%d  %I:%M:%S %p", errors='raise')  #  vs 'coerce

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
golfer_list = sorted(df['Golfer'].dropna().unique())
default_index = golfer_list.index("Dave") if "Dave" in golfer_list else 0
selected_golfer = st.sidebar.selectbox("Select Golfer", golfer_list, index=default_index)
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

    # Map club name to color for matching ellipse color
    club_colors = {}
    for d in fig1.data:
        if 'name' in d and 'marker' in d:
            club_colors[d.name] = d.marker.color

    for club in df['Club'].dropna().unique():
        df_club = df[df['Club'] == club]
        x_vals = df_club[x_choice].dropna()
        y_vals = df_club['Lateral_yds'].dropna()
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
                color = club_colors.get(club, 'red')
                fig1.add_trace(go.Scatter(
                    x=ellipse_coords[0], y=ellipse_coords[1], mode='lines',
                    name=f'{club} {confidence}% CI',
                    line=dict(color=color, dash='dash' if confidence == 1 else None)
            ))
                
    return fig1

# Bar chart generator

def create_bar_chart(df, session, y_variable,club):
    df_session = df[(df['Session'] == session) & (df['Club'] == club)].copy()
    if df_session.empty:
        st.error(f"No data available for session: {session}")
        return None

    df_session['Club'] = pd.Categorical(df_session['Club'], categories=CLUB_ORDER, ordered=True)
    df_session = df_session.sort_values(['Club', 'Shot'])

    hover_columns = ['Shot_Type','Club_mph', 'Smash_Factor', 'AOA', 'Spin_Loft', 'Swing_V', 'FTP',
                     'Dynamic_Loft', 'Club_Path', 'Launch_V', 'Low_Point_ftin',
                     'Lateral_Impact_in', 'Vertical_Impact_in']

    fig = px.bar(df_session, x='Shot', y=y_variable, title=f"{y_variable} for {session}",
                 color='Club', text=y_variable, height=500,hover_data=hover_columns)

    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(xaxis_title="Shot Sequence", yaxis_title=y_variable, showlegend=True)
    return fig

# Reference chart below main bar chart

def create_fixed_bar_chart(df, session, reference_variable="Total_yds",club=None):
    df_session = df[(df['Session'] == session) & (df['Club'] == club)].copy()
    if df_session.empty:
        return None

    df_session['Club'] = pd.Categorical(df_session['Club'], categories=CLUB_ORDER, ordered=True)
    df_session = df_session.sort_values(['Club', 'Shot'])

    all_cats = df['Smash_Factor_Category'].cat.categories
    present_cats = df_session['Smash_Factor_Category'].dropna().unique()
    missing_cats = [cat for cat in all_cats if cat not in present_cats]
    dummy_rows = pd.DataFrame({
        'Shot': [np.nan] * len(missing_cats),
        'Club': [df_session['Club'].iloc[0]] * len(missing_cats),
        'Session': [session] * len(missing_cats),
        'Smash_Factor_Category': pd.Categorical(missing_cats, categories=all_cats, ordered=True),
        'Total_yds': [np.nan] * len(missing_cats)
    })
    df_session = pd.concat([df_session, dummy_rows], ignore_index=True)
    df_session['Smash_Factor_Category'] = df_session['Smash_Factor_Category'].cat.set_categories(all_cats)

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
        facet_col=None,
        facet_col_wrap=4,
        color_discrete_map=color_discrete_map
    )

    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Shot Sequence",
        yaxis_title=reference_variable,
        showlegend=True,
        legend_title_text='',
        margin=dict(t=60, b=40, l=20, r=20)
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

# Helper to write a plotly figure to an in-memory PNG image
def fig_to_png_bytes(fig,scale=3):  # Increase scale for higher resolution
    buf = BytesIO()
    fig.write_image(buf, format="png", scale = scale)
    buf.seek(0)
    return buf

# PDF generation function

def generate_pdf(fig_bar, fig_ref, fig_xy, fig_hist):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    figs = [
        (fig_bar, "Bar Chart"),
        (fig_hist, "Histogram"),
        (fig_ref, "Reference Chart"),
        (fig_xy, "XY Plot")
    ]

    for fig, title in figs:
        if fig:
            image_stream = fig_to_png_bytes(fig,scale=3)
            temp_path = f"temp_{title.replace(' ', '_')}.png"
            with open(temp_path, "wb") as f:
                f.write(image_stream.read())
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, title, ln=True)
            pdf.image(temp_path, x=10, y=20, w=190)
            os.remove(temp_path)

    pdf.output("golf_report.pdf")

# Tab layout
tab1, tab2, tab3 = st.tabs(["Dispersion", "Session Bars", "XY Plots"])
# Tab 1 – Dispersion Field
with tab1:
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(create_fig1(filtered_df, "Carry_yds"), use_container_width=True, key="T1C1R1")

# Tab 2 – Session Bar Charts
with tab2:
    st.write("### Session Bar Chart")

    session_clubs = filtered_df['Club'].dropna().unique().tolist()
    session_clubs.sort(key=lambda x: CLUB_ORDER.index(x) if x in CLUB_ORDER else 999)
    selected_club = st.selectbox("Select Club", session_clubs)
    df_club = filtered_df[filtered_df['Club'] == selected_club]

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        y_variable_choice = st.selectbox("Select Variable", NUMERIC_COLS, index=4)
        fig_bar = create_bar_chart(df_club, selected_session, y_variable_choice,selected_club)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

    with row1_col2:
        fig_hist = px.histogram(df_club, x=y_variable_choice, nbins=20,
                                title=f"Histogram of {y_variable_choice}",
                                color='Club',
                                color_discrete_sequence=px.colors.qualitative.Bold)
        fig_hist.update_layout(margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        ref_choice = st.selectbox("Reference Chart Variable", ["Total_yds", "Carry_yds"], index=0)
        fig_ref = create_fixed_bar_chart(df_club, selected_session, reference_variable=ref_choice,club=selected_club)
        if fig_ref:
            st.plotly_chart(fig_ref, use_container_width=True)

    with row2_col2:
        fig_xy = px.scatter(df_club, x=y_variable_choice, y=ref_choice,
                            title=f"{ref_choice} vs {y_variable_choice}",
                            color='Club',
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            hover_data=['Time', 'Shot_Type'])
        fig_xy.update_layout(margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig_xy, use_container_width=True)
        st.markdown("---")

    if st.button("Generate PDF Report"):
        try:
            generate_pdf(fig_bar, fig_ref, fig_xy, fig_hist)
            with open("golf_report.pdf", "rb") as f:
                st.download_button("Download PDF Report", f.read(), file_name="golf_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Error generating report: {e}")

# List of XY pairs for plotting
def get_xy_pairs():
    return [
        ("Smash_Factor", "Carry_yds"),
        ("Club_Path", "Launch_H"),
        ("Roll_yds", "DescentV"),
        ("Carry_yds", "Total_yds"),
        ("Club_Path", "Smash_Factor"),
        ("Launch_H", "Smash_Factor"),
        ("AOA", "Smash_Factor"),
        ("Spin_rpm", "Carry_yds"),
        ("Club_mph", "Ball_mph")
    ]

# Generate individual scatter plot

def create_xy_plot(df, x_var, y_var):
    fig = px.scatter(df, x=x_var, y=y_var, color='Club',
                     title=f"{y_var} vs {x_var}",
                     height=300,
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     hover_data=['Time', 'Shot_Type', 'Club'])
    fig.update_layout(margin=dict(t=40, b=20, l=10, r=10))
    return fig

# Render Tab 3
with tab3:
    st.write("### XY Plots")
    xy_pairs = get_xy_pairs()
    for i in range(0, len(xy_pairs), 3):
        cols = st.columns(3)
        for j, (x, y) in enumerate(xy_pairs[i:i+3]):
            with cols[j]:
                st.plotly_chart(create_xy_plot(filtered_df, x, y), use_container_width=True)