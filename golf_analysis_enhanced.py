import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pandas.api.types import CategoricalDtype
from scipy.stats import chi2
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TO RUN THIS, USE TERMINAL:
# streamlit run "path/to/your/golf_analysis_enhanced.py"

# Configuration
pio.templates.default = "plotly_white"
st.set_page_config(
    page_title="Golf Performance Analysis",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

# File path configuration
DATA_FOLDER = ''
FILENAME = 'FS_Golf_DB.xlsx'

# Enhanced constants
NUMERIC_COLS = [
    'Ball_mph', 'Club_mph', 'Smash_Factor', 'Carry_yds', 'Total_yds', 'Roll_yds',
    'Swing_H', 'Spin_rpm', 'Height_ft', 'Time_s', 'AOA', 'Spin_Loft', 'Swing_V',
    'Spin_Axis', 'Lateral_yds', 'FTP', 'FTT', 'Dynamic_Loft', 'Club_Path', 'Launch_H',
    'Launch_V', 'Low_Point_ftin', 'DescentV', 'Curve_Dist_yds', 'Lateral_Impact_in', 'Vertical_Impact_in'
]

CLUB_ORDER = [
    'Driver', '3 Wood', '5 Wood', '2 Iron', '3 Iron', '4 Iron', '5 Iron', '6 Iron', '7 Iron',
    '8 Iron', '9 Iron', 'Pitching Wedge', 'Gap Wedge', 'Sand Wedge', 'Lob Wedge'
]

# Key performance metrics by club type
CLUB_PERFORMANCE_METRICS = {
    'Driver': ['Ball_mph', 'Total_yds', 'Smash_Factor', 'Launch_H', 'Launch_V', 'Spin_rpm'],
    'Woods': ['Ball_mph', 'Total_yds', 'Smash_Factor', 'Launch_H', 'Launch_V', 'Spin_rpm'],
    'Irons': ['Ball_mph', 'Carry_yds', 'Smash_Factor', 'Launch_H', 'Launch_V', 'Spin_rpm'],
    'Wedges': ['Ball_mph', 'Carry_yds', 'Smash_Factor', 'Launch_H', 'Launch_V', 'Spin_rpm']
}

# Load data with caching
@st.cache_data
def load_data(filepath):
    try:
        return pd.read_excel(filepath, engine='openpyxl', sheet_name='DB')
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Enhanced data cleaning functions
def clean_column_names(df):
    """Clean column names by removing special characters and normalizing"""
    df.columns = (df.columns
                  .str.replace(r'[^\w\s]', '', regex=True)
                  .str.replace('\xa0', ' ')
                  .str.strip()
                  .str.replace(' ', '_'))
    return df

def convert_directional_value(val):
    """Convert directional values like '1.3 L' to -1.3, '1.3 R' to 1.3"""
    if pd.isna(val):
        return None
    
    try:
        val = str(val).strip()
        val = val.replace("\xa0", "").replace("\u200b", "").replace(" ", "")
        
        if len(val) < 2 or not val[-1].isalpha():
            return float(val) if val.replace('.', '').replace('-', '').isdigit() else None
            
        number_str = val[:-1]
        direction = val[-1].upper()
        
        if not number_str.replace('.', '').replace('-', '').isdigit():
            return None
            
        number = float(number_str)
        
        # Convention: Left = negative, Right = positive
        return -number if direction == 'L' else number if direction == 'R' else None
        
    except (ValueError, IndexError):
        return None

def convert_directional_columns(df, columns):
    """Convert directional columns to numeric"""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_directional_value)
        else:
            st.warning(f"Column '{col}' not found in data")
    return df

def ensure_numeric(df, columns):
    """Ensure specified columns are numeric"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.warning(f"Numeric column '{col}' not found in data")
    return df

def categorize_clubs(club_name):
    """Categorize clubs into types"""
    if pd.isna(club_name):
        return 'Unknown'
    
    club_name = str(club_name).strip()
    
    if 'Driver' in club_name:
        return 'Driver'
    elif 'Wood' in club_name:
        return 'Woods'
    elif any(x in club_name for x in ['Iron', '2 Iron', '3 Iron', '4 Iron', '5 Iron', '6 Iron', '7 Iron', '8 Iron', '9 Iron']):
        return 'Irons'
    elif any(x in club_name for x in ['Wedge', 'PW', 'GW', 'SW', 'LW']):
        return 'Wedges'
    else:
        return 'Other'

def process_dataframe(df):
    """Enhanced data processing with better error handling"""
    if df is None:
        return None, None, None
    
    # Clean column names
    df = clean_column_names(df)
    
    # Check for essential columns
    required_columns = ['Time', 'Golfer', 'Club']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return None, None, None
    
    # Process time column
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df = df.sort_values('Time')
    
    # Create session identifiers
    df['Date'] = df['Time'].dt.date
    df['Session'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M')
    df['Session_Date'] = df['Time'].dt.strftime('%Y-%m-%d')
    df['Session_Time'] = df['Time'].dt.strftime('%H:%M')
    
    # Convert directional columns
    directional_cols = ['Swing_H', 'Spin_Axis', 'Lateral_yds', 'FTP', 'FTT', 'Club_Path', 'Launch_H']
    df = convert_directional_columns(df, directional_cols)
    
    # Ensure numeric columns
    df = ensure_numeric(df, NUMERIC_COLS)
    
    # Add club categories
    df['Club_Type'] = df['Club'].apply(categorize_clubs)
    
    # Add performance categories
    if 'Smash_Factor' in df.columns:
        bins = [0, 1.0, 1.1, 1.2, 1.3, 1.4, float('inf')]
        labels = ['<1.0', '1.0-1.1', '1.1-1.2', '1.2-1.3', '1.3-1.4', '>1.4']
        df['Smash_Factor_Category'] = pd.cut(df['Smash_Factor'], bins=bins, labels=labels, right=False)
    
    # Add shot quality metrics
    if 'Lateral_yds' in df.columns:
        df['Accuracy'] = df['Lateral_yds'].abs()
        df['Accuracy_Category'] = pd.cut(df['Accuracy'], 
                                       bins=[0, 5, 10, 20, 50, float('inf')],
                                       labels=['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor'])
    
    # Create pivot tables for analysis
    numeric_cols_available = [col for col in NUMERIC_COLS if col in df.columns]
    
    try:
        df_sessions = df.pivot_table(
            index='Club', 
            columns=['Golfer', 'Session_Date'], 
            values=numeric_cols_available, 
            aggfunc='mean'
        )
        
        df_golfer = df.pivot_table(
            index='Club', 
            columns='Golfer', 
            values=numeric_cols_available, 
            aggfunc='mean'
        )
        
        # Reorder clubs if they exist in the data
        available_clubs = [club for club in CLUB_ORDER if club in df_sessions.index]
        if available_clubs:
            df_sessions = df_sessions.reindex(available_clubs)
            df_golfer = df_golfer.reindex(available_clubs)
    
    except Exception as e:
        st.warning(f"Error creating pivot tables: {str(e)}")
        df_sessions = None
        df_golfer = None
    
    return df, df_sessions, df_golfer

# Enhanced visualization functions
def create_dispersion_plot(df, x_col='Carry_yds', y_col='Lateral_yds', color_col='Club'):
    """Create enhanced dispersion plot with confidence ellipses"""
    if df.empty:
        st.warning("No data available for dispersion plot")
        return None
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=f"Shot Dispersion: {x_col} vs {y_col}",
        hover_data=['Time', 'Shot_Type', 'Club', 'Ball_mph', 'Smash_Factor'],
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=df[x_col].mean(), line_dash="dash", line_color="red", opacity=0.5)
    
    # Customize layout
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        height=600,
        showlegend=True
    )
    
    return fig

def create_performance_trends(df, metric='Carry_yds', club_filter=None):
    """Create performance trend over time"""
    if df.empty:
        return None
    
    if club_filter:
        df = df[df['Club'] == club_filter]
    
    if df.empty:
        return None
    
    fig = px.line(
        df, 
        x='Time', 
        y=metric,
        color='Club',
        title=f"{metric.replace('_', ' ').title()} Performance Over Time",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric.replace('_', ' ').title(),
        height=500
    )
    
    return fig

def create_club_comparison(df, metric='Carry_yds'):
    """Create club comparison chart"""
    if df.empty:
        return None
    
    club_stats = df.groupby('Club')[metric].agg(['mean', 'std', 'count']).reset_index()
    club_stats = club_stats.sort_values('mean', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Average',
        x=club_stats['Club'],
        y=club_stats['mean'],
        error_y=dict(type='data', array=club_stats['std']),
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f"Club Comparison - {metric.replace('_', ' ').title()}",
        xaxis_title="Club",
        yaxis_title=metric.replace('_', ' ').title(),
        height=500
    )
    
    return fig

def create_correlation_heatmap(df, metrics=None):
    """Create correlation heatmap for key metrics"""
    if df.empty or metrics is None:
        return None
    
    available_metrics = [col for col in metrics if col in df.columns]
    
    if len(available_metrics) < 2:
        return None
    
    corr_matrix = df[available_metrics].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix - Key Metrics",
        color_continuous_scale='RdBu_r'
    )
    
    return fig

# Main application
def main():
    st.markdown('<h1 class="main-header">⛳ Golf Performance Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data(DATA_FOLDER + FILENAME)
    
    if df is None:
        st.error("Could not load data. Please check the file path and format.")
        return
    
    # Process data
    df, df_sessions, df_golfer = process_dataframe(df)
    
    if df is None:
        st.error("Error processing data. Please check the data format.")
        return
    
    # Sidebar filters
    st.sidebar.markdown("## 🎯 Filters")
    
    # Golfer selection
    golfers = sorted(df['Golfer'].dropna().unique())
    selected_golfer = st.sidebar.selectbox("Select Golfer", golfers)
    
    # Filter data by golfer
    golfer_data = df[df['Golfer'] == selected_golfer]
    
    # Session selection
    sessions = sorted(golfer_data['Session_Date'].dropna().unique(), reverse=True)
    selected_sessions = st.sidebar.multiselect(
        "Select Sessions", 
        sessions, 
        default=sessions[:5] if len(sessions) > 5 else sessions
    )
    
    # Club selection
    clubs = sorted(golfer_data['Club'].dropna().unique())
    selected_clubs = st.sidebar.multiselect("Select Clubs", clubs, default=clubs)
    
    # Filter data
    filtered_df = golfer_data[
        (golfer_data['Session_Date'].isin(selected_sessions)) &
        (golfer_data['Club'].isin(selected_clubs))
    ]
    
    # Show data summary
    st.sidebar.markdown("## 📊 Data Summary")
    st.sidebar.metric("Total Shots", len(filtered_df))
    st.sidebar.metric("Sessions", len(selected_sessions))
    st.sidebar.metric("Clubs", len(selected_clubs))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Overview", 
        "🎯 Shot Dispersion", 
        "📊 Club Analysis", 
        "🔄 Trends", 
        "🔍 Detailed Analysis"
    ])
    
    with tab1:
        st.subheader("Performance Overview")
        
        if not filtered_df.empty:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_carry = filtered_df['Carry_yds'].mean()
                st.metric("Average Carry", f"{avg_carry:.1f} yds" if not pd.isna(avg_carry) else "N/A")
            
            with col2:
                avg_ball_speed = filtered_df['Ball_mph'].mean()
                st.metric("Average Ball Speed", f"{avg_ball_speed:.1f} mph" if not pd.isna(avg_ball_speed) else "N/A")
            
            with col3:
                avg_smash = filtered_df['Smash_Factor'].mean()
                st.metric("Average Smash Factor", f"{avg_smash:.2f}" if not pd.isna(avg_smash) else "N/A")
            
            with col4:
                avg_accuracy = filtered_df['Accuracy'].mean()
                st.metric("Average Accuracy", f"{avg_accuracy:.1f} yds" if not pd.isna(avg_accuracy) else "N/A")
            
            # Performance by club type
            st.subheader("Performance by Club Type")
            club_type_stats = filtered_df.groupby('Club_Type').agg({
                'Carry_yds': 'mean',
                'Ball_mph': 'mean',
                'Smash_Factor': 'mean',
                'Accuracy': 'mean'
            }).round(2)
            
            st.dataframe(club_type_stats, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab2:
        st.subheader("Shot Dispersion Analysis")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", ['Carry_yds', 'Total_yds', 'Ball_mph'], key='disp_x')
            
            with col2:
                color_by = st.selectbox("Color by", ['Club', 'Club_Type', 'Shot_Type'], key='disp_color')
            
            fig = create_dispersion_plot(filtered_df, x_col=x_axis, color_col=color_by)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for dispersion analysis")
    
    with tab3:
        st.subheader("Club Analysis")
        
        if not filtered_df.empty:
            metric_options = ['Carry_yds', 'Ball_mph', 'Smash_Factor', 'Spin_rpm', 'Launch_H', 'Launch_V']
            available_metrics = [m for m in metric_options if m in filtered_df.columns]
            
            selected_metric = st.selectbox("Select Metric", available_metrics)
            
            fig = create_club_comparison(filtered_df, metric=selected_metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("Metric Correlations")
            corr_fig = create_correlation_heatmap(filtered_df, available_metrics)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.warning("No data available for club analysis")
    
    with tab4:
        st.subheader("Performance Trends")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                trend_metric = st.selectbox("Select Metric", ['Carry_yds', 'Ball_mph', 'Smash_Factor', 'Accuracy'], key='trend_metric')
            
            with col2:
                club_filter = st.selectbox("Filter by Club", ['All'] + list(filtered_df['Club'].unique()), key='trend_club')
            
            club_filter = None if club_filter == 'All' else club_filter
            
            fig = create_performance_trends(filtered_df, metric=trend_metric, club_filter=club_filter)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for trend analysis")
    
    with tab5:
        st.subheader("Detailed Analysis")
        
        if not filtered_df.empty:
            # Shot quality distribution
            st.subheader("Shot Quality Distribution")
            if 'Accuracy_Category' in filtered_df.columns:
                accuracy_dist = filtered_df['Accuracy_Category'].value_counts()
                fig = px.pie(
                    values=accuracy_dist.values,
                    names=accuracy_dist.index,
                    title="Shot Accuracy Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed statistics
            st.subheader("Detailed Statistics")
            numeric_cols = [col for col in NUMERIC_COLS if col in filtered_df.columns]
            detailed_stats = filtered_df[numeric_cols].describe().round(2)
            st.dataframe(detailed_stats, use_container_width=True)
            
            # Raw data export
            st.subheader("Raw Data")
            if st.checkbox("Show raw data"):
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name=f"golf_data_{selected_golfer}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No data available for detailed analysis")

if __name__ == "__main__":
    main()
