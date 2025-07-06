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

# Custom CSS for larger fonts throughout the app
st.markdown("""
<style>
    /* Increase base font size */
    .main .block-container {
        font-size: 18px;
    }
    
    /* Sidebar font sizes */
    .sidebar .sidebar-content {
        font-size: 16px;
    }
    
    /* Selectbox and input labels */
    .stSelectbox label, .stMultiSelect label {
        font-size: 16px !important;
        font-weight: bold;
    }
    
    /* Metric labels and values */
    .metric-container .metric-label {
        font-size: 16px !important;
    }
    .metric-container .metric-value {
        font-size: 24px !important;
    }
    
    /* Headers */
    h1 {
        font-size: 32px !important;
    }
    h2 {
        font-size: 26px !important;
    }
    h3 {
        font-size: 22px !important;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px !important;
    }
    
    /* Dataframe text */
    .dataframe {
        font-size: 16px !important;
    }
    
    /* Warning and info messages */
    .stAlert {
        font-size: 16px !important;
    }
    
    /* Debug text */
    .stMarkdown p {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

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

# Color options for different attributes
COLOR_ATTRIBUTE_OPTIONS = {
    'Club': 'Club',
    'Shot Type': 'Shot_Type',
    'Ball Speed': 'Ball_mph',
    'Club Speed': 'Club_mph',
    'Smash Factor': 'Smash_Factor',
    'Spin Rate': 'Spin_rpm',
    'Launch Angle': 'Launch_V',
    'Launch Direction': 'Launch_H',
    'Club Path': 'Club_Path',
    'Attack Angle': 'AOA',
    'Dynamic Loft': 'Dynamic_Loft',
    'Face to Path (FTP)': 'FTP',
    'Face to Target (FTT)': 'FTT',
    'Carry Distance': 'Carry_yds',
    'Total Distance': 'Total_yds',
    'Height': 'Height_ft',
    'Descent Angle': 'DescentV'
}

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
        return -number if direction == 'L' else number if direction == 'R' else None
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

# Get sessions for the selected golfer and sort by actual datetime
golfer_sessions = df[df['Golfer'] == selected_golfer][['Session', 'Time']].drop_duplicates()
golfer_sessions = golfer_sessions.sort_values('Time', ascending=False)  # Most recent first
sessions_ordered = golfer_sessions['Session'].tolist()

selected_sessions = st.sidebar.multiselect("Select Sessions", sessions_ordered, default=[sessions_ordered[0]] if len(sessions_ordered) > 0 else [])

# Club filter - this is crucial for meaningful analysis
golfer_data = df[df['Golfer'] == selected_golfer]
clubs = sorted(golfer_data['Club'].dropna().unique())
selected_club = st.sidebar.selectbox("Select Club", clubs)

# Filtered dataframe - now filtered by golfer, sessions, AND club
filtered_df = df[(df['Golfer'] == selected_golfer) & (df['Session'].isin(selected_sessions)) & (df['Club'] == selected_club)]

# Supporting data for plotting
hov_data = ['Time', 'Shot_Type', 'Club', 'Carry_yds', 'Lateral_yds', 'Ball_mph', 'Smash_Factor']

# Confidence ellipse radius calculator
def calculate_confidence_radius(percent):
    return chi2.ppf(percent/100, df=2)

# Enhanced scatter plot with configurable color attribute
def create_fig1(df, x_choice, color_choice='Club', x_axis_start='From Min Value'):
    if df.empty:
        st.warning("No data available for the selected filters.")
        return None
    
    # Determine if color choice is numeric or categorical
    if color_choice in df.columns:
        color_data = df[color_choice]
        is_numeric = pd.api.types.is_numeric_dtype(color_data)
        
        # For numeric data, use continuous color scale
        if is_numeric:
            color_scale = px.colors.sequential.Viridis
        else:
            color_scale = px.colors.qualitative.Bold
    else:
        st.warning(f"Column '{color_choice}' not found in data.")
        return None
    
    # Calculate appropriate axis ranges - with dynamic x-axis start
    if not df[x_choice].empty:
        data_min = df[x_choice].min()
        data_max = df[x_choice].max()
        
        # Dynamic x-axis start based on user selection
        if x_axis_start == "From Zero":
            x_min = 0
        else:  # "From Min Value"
            x_min = data_min * 0.95
        
        x_max = data_max * 1.25  # Changed back to 1.25 as requested
        
        # Debug info
        st.write(f"Debug: {x_choice} range: {data_min:.1f} to {data_max:.1f}, plot range: {x_min:.1f} to {x_max:.1f}")
    else:
        x_min, x_max = 0, 100
    
    # For lateral distance, use much wider range to show ellipses properly
    if not df["Lateral_yds"].empty:
        y_min_val = df["Lateral_yds"].min()
        y_max_val = df["Lateral_yds"].max()
        y_range = max(abs(y_min_val), abs(y_max_val))
        # Use much wider range - at least 50 yards total width, or 3x the data range
        min_range = 50  # minimum 50 yards total width (±25 yards)
        plot_range = max(min_range/2, y_range * 2.0)  # Double the data range or minimum 25 yards each side
        y_min, y_max = -plot_range, plot_range
        # Debug info
        st.write(f"Debug: Lateral range: {y_min_val:.1f} to {y_max_val:.1f}, plot range: {y_min:.1f} to {y_max:.1f}")
    else:
        y_min, y_max = -25, 25

    # Create scatter plot
    fig1 = px.scatter(
        df, 
        x=x_choice, 
        y='Lateral_yds', 
        color=color_choice, 
        title=f"Dispersion Field (Colored by {color_choice})",
        color_continuous_scale=color_scale if is_numeric else None,
        color_discrete_sequence=color_scale if not is_numeric else None,
        hover_data=hov_data,
        labels={
            x_choice: x_choice.replace('_', ' ').title(),
            'Lateral_yds': 'Lateral Distance (yds)',
            color_choice: color_choice.replace('_', ' ').title()
        }
    )
    
    fig1.update_xaxes(range=[x_min, x_max], title=x_choice.replace('_', ' ').title(), title_font_size=18, tickfont_size=14)
    fig1.update_yaxes(
        range=[y_min, y_max], 
        title="Lateral Distance (yds)", 
        title_font_size=18, 
        tickfont_size=14,
        autorange="reversed"  # Flip y-axis so negative (left) goes up
    )
    fig1.update_layout(
        yaxis_scaleanchor=None,  # Remove aspect ratio constraint to allow wider field
        height=600,
        showlegend=True,
        title_font_size=20,
        legend_font_size=14,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        )
    )
    
    # Add confidence ellipses - with better error handling
    # First, ensure we have matching data points (no NaN in either x or y)
    valid_mask = df[[x_choice, 'Lateral_yds']].notna().all(axis=1)
    valid_data = df[valid_mask]
    
    if len(valid_data) > 2:  # Need at least 3 points for meaningful ellipse
        try:
            x_vals = valid_data[x_choice].values
            y_vals = valid_data['Lateral_yds'].values
            
            # Check if we have enough variation in both dimensions
            if len(np.unique(x_vals)) > 1 and len(np.unique(y_vals)) > 1:
                # Create covariance matrix
                data_matrix = np.vstack([x_vals, y_vals])
                cov = np.cov(data_matrix)
                
                # Calculate eigenvalues and eigenvectors
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(np.abs(lambda_))  # Take absolute value to avoid complex numbers
                
                if np.all(lambda_ > 0):  # Ensure positive eigenvalues
                    theta = np.degrees(np.arctan2(*v[:, 0][::-1]))
                    ellipse_x = np.linspace(0, 2 * np.pi, 100)

                    for confidence in [80, 50]:
                        radius = calculate_confidence_radius(confidence)
                        ellipse_coords = np.array([
                            2 * np.sqrt(radius) * lambda_[0] * np.cos(ellipse_x),
                            2 * np.sqrt(radius) * lambda_[1] * np.sin(ellipse_x)
                        ])
                        
                        # Apply rotation
                        rotation_matrix = np.array([
                            [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                            [np.sin(np.radians(theta)), np.cos(np.radians(theta))]
                        ])
                        ellipse_coords = rotation_matrix @ ellipse_coords
                        
                        # Translate to data center
                        ellipse_coords[0] += x_vals.mean()
                        ellipse_coords[1] += y_vals.mean()
                        
                        fig1.add_trace(go.Scatter(
                            x=ellipse_coords[0], 
                            y=ellipse_coords[1], 
                            mode='lines',
                            name=f'{confidence}% Confidence Ellipse',
                            line=dict(color='red', dash='dash' if confidence == 50 else 'solid', width=2),
                            showlegend=True
                        ))
        except Exception as e:
            st.warning(f"Could not calculate confidence ellipses: {str(e)}")
    else:
        if len(valid_data) > 0:
            st.info(f"Only {len(valid_data)} valid data points - need at least 3 for confidence ellipses.")
    
    return fig1

# Enhanced statistics table
def create_stats_table(df):
    if df.empty:
        return None
    
    # Calculate key statistics
    stats_data = []
    
    # Overall statistics
    stats_data.append({
        'Metric': 'Total Shots',
        'Value': len(df),
        'Unit': 'shots'
    })
    
    # Key performance metrics
    key_metrics = {
        'Average Carry': ('Carry_yds', 'yds'),
        'Average Ball Speed': ('Ball_mph', 'mph'),
        'Average Smash Factor': ('Smash_Factor', ''),
        'Average Spin Rate': ('Spin_rpm', 'rpm'),
        'Average Launch Angle': ('Launch_V', '°'),
        'Average Launch Direction': ('Launch_H', '°'),
        'Average Club Path': ('Club_Path', '°'),
        'Lateral Accuracy (±)': ('Lateral_yds', 'yds')
    }
    
    for metric_name, (col_name, unit) in key_metrics.items():
        if col_name in df.columns:
            if col_name == 'Lateral_yds':
                # For lateral accuracy, use absolute value
                value = df[col_name].abs().mean()
            else:
                value = df[col_name].mean()
            
            if not pd.isna(value):
                stats_data.append({
                    'Metric': metric_name,
                    'Value': f"{value:.1f}" if unit != '' else f"{value:.2f}",
                    'Unit': unit
                })
    
    return pd.DataFrame(stats_data)

# Tab layout
tab1, tab2, tab3 = st.tabs(["Dispersion", "Statistics", "Club Performance"])

with tab1:
    st.header("Shot Dispersion Analysis")
    
    # Controls for the dispersion plot
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_axis_choice = st.selectbox(
            "X-Axis Variable:", 
            ["Carry_yds", "Total_yds", "Ball_mph", "Club_mph"],
            index=0
        )
    
    with col2:
        color_choice = st.selectbox(
            "Color Points By:",
            list(COLOR_ATTRIBUTE_OPTIONS.keys()),
            index=0
        )
        color_column = COLOR_ATTRIBUTE_OPTIONS[color_choice]
    
    with col3:
        x_axis_start = st.selectbox(
            "X-Axis Start:",
            ["From Zero", "From Min Value"],
            index=1  # Default to "From Min Value" (0.95 * min)
        )
    
    with col4:
        st.metric("Selected Club", selected_club)
        st.metric("Total Shots", len(filtered_df))
    
    # Create and display the dispersion plot
    fig1 = create_fig1(filtered_df, x_axis_choice, color_column, x_axis_start)
    if fig1:
        st.plotly_chart(fig1, use_container_width=True, key="dispersion_plot")

with tab2:
    st.header(f"Session Statistics - {selected_club}")
    
    # Display statistics table - constrained to left side with larger text
    stats_df = create_stats_table(filtered_df)
    if stats_df is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(
                stats_df, 
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Metric": st.column_config.TextColumn(width="medium"),
                    "Value": st.column_config.TextColumn(width="small"),
                    "Unit": st.column_config.TextColumn(width="small")
                }
            )
    
    # Show session details for the selected club
    if len(selected_sessions) > 0:
        st.subheader(f"Session Details - {selected_club}")
        session_info = []
        for session in selected_sessions:
            session_data = filtered_df[filtered_df['Session'] == session]
            if not session_data.empty:
                session_info.append({
                    'Session': session,
                    'Shots': len(session_data),
                    'Avg Carry': f"{session_data['Carry_yds'].mean():.1f} yds" if 'Carry_yds' in session_data.columns else 'N/A',
                    'Avg Ball Speed': f"{session_data['Ball_mph'].mean():.1f} mph" if 'Ball_mph' in session_data.columns else 'N/A',
                    'Avg Smash': f"{session_data['Smash_Factor'].mean():.2f}" if 'Smash_Factor' in session_data.columns else 'N/A',
                    'Accuracy (±)': f"{session_data['Lateral_yds'].abs().mean():.1f} yds" if 'Lateral_yds' in session_data.columns else 'N/A'
                })
        
        if session_info:
            st.dataframe(
                pd.DataFrame(session_info), 
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Session": st.column_config.TextColumn(width="large"),
                    "Shots": st.column_config.NumberColumn(width="small"),
                    "Avg Carry": st.column_config.TextColumn(width="medium"),
                    "Avg Ball Speed": st.column_config.TextColumn(width="medium"),
                    "Avg Smash": st.column_config.TextColumn(width="medium"),
                    "Accuracy (±)": st.column_config.TextColumn(width="medium")
                }
            )
    
    # Add club-specific performance ranges for context
    if not filtered_df.empty:
        st.subheader(f"Performance Range - {selected_club}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Carry_yds' in filtered_df.columns:
                carry_min = filtered_df['Carry_yds'].min()
                carry_max = filtered_df['Carry_yds'].max()
                st.metric("Carry Range", f"{carry_min:.0f} - {carry_max:.0f} yds")
        
        with col2:
            if 'Ball_mph' in filtered_df.columns:
                speed_min = filtered_df['Ball_mph'].min()
                speed_max = filtered_df['Ball_mph'].max()
                st.metric("Ball Speed Range", f"{speed_min:.0f} - {speed_max:.0f} mph")
        
        with col3:
            if 'Lateral_yds' in filtered_df.columns:
                lateral_range = filtered_df['Lateral_yds'].abs().max()
                st.metric("Max Lateral Miss", f"{lateral_range:.1f} yds")

with tab3:
    st.header(f"Club Performance - {selected_club}")
    
    if not filtered_df.empty:
        # Session-by-session comparison for the selected club
        if len(selected_sessions) > 1:
            st.subheader("Session Comparison")
            session_comparison = []
            for session in selected_sessions:
                session_data = filtered_df[filtered_df['Session'] == session]
                if not session_data.empty:
                    session_comparison.append({
                        'Session': session,
                        'Shots': len(session_data),
                        'Avg Carry': session_data['Carry_yds'].mean() if 'Carry_yds' in session_data.columns else None,
                        'Std Carry': session_data['Carry_yds'].std() if 'Carry_yds' in session_data.columns else None,
                        'Avg Ball Speed': session_data['Ball_mph'].mean() if 'Ball_mph' in session_data.columns else None,
                        'Avg Smash': session_data['Smash_Factor'].mean() if 'Smash_Factor' in session_data.columns else None,
                        'Accuracy (±)': session_data['Lateral_yds'].abs().mean() if 'Lateral_yds' in session_data.columns else None
                    })
            
            if session_comparison:
                comparison_df = pd.DataFrame(session_comparison)
                st.dataframe(comparison_df.round(2), use_container_width=True, hide_index=True)
                
                # Performance trend chart for this club
                if len(comparison_df) > 1:
                    fig_trend = px.line(
                        comparison_df, 
                        x='Session', 
                        y='Avg Carry',
                        title=f"Carry Distance Trend - {selected_club}",
                        markers=True
                    )
                    fig_trend.update_layout(
                        xaxis_tickangle=-45,
                        title_font_size=18,
                        xaxis_title_font_size=16,
                        yaxis_title_font_size=16,
                        xaxis_tickfont_size=12,
                        yaxis_tickfont_size=12
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
        
        # Overall performance summary for this club
        st.subheader(f"Overall Performance Summary - {selected_club}")
        summary_stats = {
            'Total Shots': len(filtered_df),
            'Sessions': len(selected_sessions),
            'Avg Carry': f"{filtered_df['Carry_yds'].mean():.1f} yds" if 'Carry_yds' in filtered_df.columns else 'N/A',
            'Carry Consistency (StdDev)': f"{filtered_df['Carry_yds'].std():.1f} yds" if 'Carry_yds' in filtered_df.columns else 'N/A',
            'Best Carry': f"{filtered_df['Carry_yds'].max():.1f} yds" if 'Carry_yds' in filtered_df.columns else 'N/A',
            'Avg Ball Speed': f"{filtered_df['Ball_mph'].mean():.1f} mph" if 'Ball_mph' in filtered_df.columns else 'N/A',
            'Avg Smash Factor': f"{filtered_df['Smash_Factor'].mean():.2f}" if 'Smash_Factor' in filtered_df.columns else 'N/A',
            'Accuracy (±)': f"{filtered_df['Lateral_yds'].abs().mean():.1f} yds" if 'Lateral_yds' in filtered_df.columns else 'N/A'
        }
        
        # Display in a nice format
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Shots", summary_stats['Total Shots'])
            st.metric("Sessions", summary_stats['Sessions'])
        with col2:
            st.metric("Avg Carry", summary_stats['Avg Carry'])
            st.metric("Best Carry", summary_stats['Best Carry'])
        with col3:
            st.metric("Carry Consistency", summary_stats['Carry Consistency (StdDev)'])
            st.metric("Avg Ball Speed", summary_stats['Avg Ball Speed'])
        with col4:
            st.metric("Avg Smash Factor", summary_stats['Avg Smash Factor'])
            st.metric("Accuracy (±)", summary_stats['Accuracy (±)'])
    else:
        st.warning(f"No data available for {selected_club} in the selected sessions.")

# Add session summary in sidebar
if len(selected_sessions) > 0:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {selected_club} Summary")
    st.sidebar.metric("Total Shots", len(filtered_df))
    if not filtered_df.empty:
        st.sidebar.metric("Average Carry", f"{filtered_df['Carry_yds'].mean():.1f} yds")
        st.sidebar.metric("Best Carry", f"{filtered_df['Carry_yds'].max():.1f} yds")
        st.sidebar.metric("Accuracy (±)", f"{filtered_df['Lateral_yds'].abs().mean():.1f} yds")
        st.sidebar.metric("Avg Smash Factor", f"{filtered_df['Smash_Factor'].mean():.2f}")