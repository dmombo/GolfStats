import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

#-----------------------  has to be called first
# Set the page layout to wide
st.set_page_config(layout="wide")

# TO RUN THIS, USE TERMINAL
# streamlit run e:\iCloudDrive\Drop\Python_dropbox\Python\2024\Streamlit_Golf_Stats.py
# streamlit run C:\Users\dmomb\OneDrive\Python\Projects\GolfDec24\Golf_Stats_v2.py
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

# Clean column names to ensure consistency
df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace('\xa0', ' ').str.strip().str.replace(' ', '_')

# Ensure Time column is in datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Convert value with error handling
def convert_value(value):
    """
    Converts any value like '40R' or '20L' to numerical form:
    - '40R' becomes -40
    - '20L' becomes 20
    Returns None for invalid entries.
    """
    try:
        value = str(value).strip()  # Ensure value is a string and remove spaces
        number, direction = value[:-1], value[-1].upper()  # Split number and direction
        number = float(number)
        return -number if direction == 'R' else number if direction == 'L' else None
    except (ValueError, AttributeError):
        return None  # Return None for invalid entries

# Convert entire column and handle missing column errors
def convert_column(df, col):
    """
    Converts all values in a column using convert_value.
    Handles missing column errors gracefully.
    """
    if col in df.columns:
        df[col] = df[col].apply(convert_value)
        print(f"Column '{col}' successfully converted.")
    else:
        st.error(f"The column '{col}' is missing from the data.")
# Convert all the columns that have L & R in the data        
lrcols = ['Swing_H','Spin_Axis','Lateral_yds','FTP','FTT','Club_Path','Launch_H']
for col in lrcols:
    convert_column(df,col)

# Create a categorical 'Session' using a conversion of Time to strings
df['Session'] = df['Time'].dt.strftime('%Y %b %d %I:%M %p')  # Or any simpler string representation
df['Session'] = df['Session'].astype('category')

# Function to remove outliers based on IQR
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    return data[data[column] >= lower_bound]

# Remove outliers for 'Carry_yds' column
df = remove_outliers(df, 'Carry_yds')

# Calculate average yardage grouped by Golfer and Club
average_yardage = df.groupby(['Golfer', 'Club'])['Carry_yds'].mean().reset_index()
average_yardage.rename(columns={'Carry_yds': 'Average_Carry_yds'}, inplace=True)
average_yardage['Average_Carry_yds'] = average_yardage['Average_Carry_yds'].round(1)

# Pivot table for display
pivot_table = average_yardage.pivot(index='Club', columns='Golfer', values='Average_Carry_yds')

dfall = df.copy()

# Columns Available: ['Mombo_ShotID', 'Club', 'Time', 'Golfer', 'Shot', 'Video', 'Ball mph','Club mph', 'Smash_Factor', 'Carry yds', 'Total yds', 'Roll yds',
#       'Swing H', 'Spin rpm', 'Height ft', 'Time s', 'AOA', 'Spin Loft','Swing V', 'Spin Axis', 'Lateral yds', 'Shot Type', 'FTP', 'FTT',
#       'Dynamic Loft', 'Club Path', 'Launch H', 'Launch V', 'Low Point ftin','DescentV', 'Curve Dist yds', 'Lateral Impact in', 'Vertical Impact in',
#       'Mode', 'Location', 'Unnamed_35', 'Unnamed_36', 'Unnamed_37','Unnamed_38', 'Unnamed_39', 'Unnamed_40', 'Comment', 'User1', 'User2','Exclude', 'Session']

numcols = ['Ball_mph','Club_mph','Smash_Factor','Carry_yds','Total_yds','Roll_yds',
           'Swing_H','Spin_rpm','Height_ft','Time_s','AOA','Spin_Loft','Swing_V','Spin_Axis','Lateral_yds','FTP','FTT',
           'Dynamic_Loft','Club_Path','Launch_H','Launch_V','Low_Point_ftin','DescentV','Curve_Dist_yds','Lateral_Impact_in','Vertical_Impact_in']

# Sidebar description -------------------------------------------------------------------------------------------
st.sidebar.title("Filter Shots")
st.sidebar.write("Choose Time/Golfer/Club")
def return_filtered_df(df, col, search_term):
    if search_term != "All":
        df = df[df[col] == search_term].copy()
    return df

col = 'Time'
choices = ['All'] + df[col].unique().tolist()
search_term = st.sidebar.selectbox('Select ' + col, choices)
df = return_filtered_df(df, col, search_term)

col = 'Golfer'
choices = ['All'] + dfall[col].unique().tolist()
search_term = st.sidebar.selectbox('Select ' + col, choices)
df = return_filtered_df(df, col, search_term)

col = 'Club'
choices = ['All'] + df[col].unique().tolist()
search_term = st.sidebar.selectbox('Select ' + col, choices)
df = return_filtered_df(df, col, search_term)

# Dark grey line separator
st.sidebar.markdown("<hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Choose what to color on 
choices = ['Time', 'Golfer', 'Club', 'Shot_Type']
color_on = st.sidebar.selectbox('Select ColorOn', choices)

df['Shot_Type'] = df['Shot_Type'].astype(str)

#----------------------------------------------------------------------------------------------------------------
hov_data = ['Time', 'Club', 'Golfer', 'Shot_Type']
###### Golf Analysis and Plots ##########################################################
df['Shot_Type'] = df['Shot_Type'].astype(str)

#######################  FIGURES FOR TAB1 #############################################################################################################
##### fig1 #####
x_max = df["Carry_yds"].max() * 1.25
max_abs_y = max(abs(df["Lateral_yds"].min()), abs(df["Lateral_yds"].max()))
scale_fac = 1.5
y_min, y_max = -max_abs_y*scale_fac, max_abs_y*scale_fac

fig1 = px.scatter(df, x="Carry_yds", y='Lateral_yds', color=color_on, title="Dispersion Field", color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)
fig1.update_xaxes(range=[0, x_max])
fig1.update_yaxes(range=[y_min, y_max])
fig1.update_layout(yaxis_scaleanchor="x")

# Add confidence ellipse to fig1
x = df['Carry_yds'].dropna()
y = df['Lateral_yds'].dropna()
if len(x) > 1 and len(y) > 1:
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    theta = np.degrees(np.arctan2(*v[:, 0][::-1]))
    ellipse_x = np.linspace(0, 2 * np.pi, 100)
    # np.sqrt(2.71) corrresponds to 90% confidence, np.sqrt(5.99) corresponds to 95% confidence,np.sqrt(1.64) corresponds to 80% confidence
    ellipse_coords = np.array([2 * np.sqrt(1.64) * lambda_[0] * np.cos(ellipse_x),
                                2 * np.sqrt(1.64) * lambda_[1] * np.sin(ellipse_x)])
    rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                 [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    ellipse_coords = rotation_matrix @ ellipse_coords
    ellipse_coords[0] += x.mean()
    ellipse_coords[1] += y.mean()

    fig1.add_trace(go.Scatter(x=ellipse_coords[0], y=ellipse_coords[1], mode='lines', name='80% CI Ellipse',
                              line=dict(color='red', dash='dash')))


##### fig2 #####
fig2 = px.scatter(df, x="Carry_yds", y='Height_ft', color=color_on, title="Height vs Carry(yds)", color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)

##### fig3 #####
df['Club_mph'] = pd.to_numeric(df['Club_mph'], errors='coerce')
df['Ball_mph'] = pd.to_numeric(df['Ball_mph'], errors='coerce')

fig3 = px.scatter(df, x='Club_mph', y='Ball_mph', color=color_on, title="Smash Factor by 'Color'",
                  color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)

##### fig4 #####
df['AOA'] = pd.to_numeric(df['AOA'], errors='coerce')

fig4 = px.scatter(df, x='AOA', y='Height_ft', color=color_on, title="Height of Ball Flight vs Angle of Attack",
                  color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)

##### fig5 #####
xvar_choice = 'Roll_yds'
yvar_choice = 'Spin_rpm'
df[xvar_choice] = pd.to_numeric(df[xvar_choice], errors='coerce')

fig5 = px.scatter(df, x=xvar_choice, y=yvar_choice, color=color_on, title="Roll vs Spin",
                  color_discrete_sequence=px.colors.qualitative.Bold, hover_data=['Shot_Type', 'Club'])
fig5.update_yaxes(range=[0, None])
fig5.update_layout(
    xaxis=dict(showline=True, mirror=True, linecolor='black'),
    yaxis=dict(showline=True, mirror=True, linecolor='black')
)

#########################  FIGURES FOR TAB2 ######################################################################
# Create a box plot
fig6 = px.box(df, x='Session', y="Carry_yds", points='all', color='Golfer')
mean_values = df.groupby('Session')['Carry_yds'].median().reset_index()

# Add the median as text annotations
for i, row in mean_values.iterrows():
    fig6.add_annotation(
        x=row['Session'],
        y=row['Carry_yds'],
        text=f"{row['Carry_yds']:.1f}",
        showarrow=False,  # Avoid cluttering with arrows
        font=dict(size=12, color='black'),
        bgcolor="white"
    )

###################################################################################################################

tab1, tab2, tab3, tab4 = st.tabs(["4 Plots", "BoxPlots","Stats","Plotchoice"])

with tab1:
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.write("Title: Col 1, Row 1")
        st.plotly_chart(fig1, use_container_width=True, key="T1C1R1")
    with row1_col2:
        st.write("Title: Col 2, Row 1")
        st.plotly_chart(fig2, use_container_width=True, key="T1C2R1")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.write("Title: Col 1, Row 2")
        st.plotly_chart(fig5, use_container_width=True, key="T1C1R2")
    with row2_col2:
        st.write("Title: Col 2, Row 2")
        st.plotly_chart(fig3, use_container_width=True, key="T1C2R2")

with tab2:
    with st.container():
        row1 = st.columns([1, 7, 1])
        with row1[1]:
            st.plotly_chart(fig1, use_container_width=True, key="T2C1R1")

    bottom_row = st.columns([3, 1, 5])

    with bottom_row[0]:
        sl_img = fol + "Golf_Logo.jpeg"
        st.write("Bottom Row - Box 2")
        st.image(sl_img, caption="Logo", use_container_width=True)

    with bottom_row[1]:
        st.write("Bottom Row - Box 1")
        st.metric(label="Sales", value="$1,200", delta="+15%")

    with bottom_row[2]:
        st.write("Box Plot")
        st.plotly_chart(fig6, use_container_width=True, key="T2C3R3")
with tab3:
    st.write("### Average Yardage by Golfer and Club")
    st.dataframe(pivot_table,height=600)
with tab4:
    # Create two columns
    col1, col2, col3, col4 = st.columns(4)
    # Place the first selectbox in the first column
    with col1:
        ycol = st.selectbox('Select column for y axis', numcols) 
    # Place the second selectbox in the second column
    with col2:
        xcol = st.selectbox('Select column for x axis', numcols)
    with col3:
        # Choose what to color on 
        choices = ['Time', 'Golfer', 'Club', 'Shot_Type']
        color_on2 = st.selectbox('Select ColorOn 2', choices)
    with col4:
        # Allow the user to set the chart height dynamically
        chart_height = st.slider('Set chart height (in pixels)', min_value=400, max_value=1200, value=800, step=50)

##### fig7 #####
    ##### fig7 #####
    fig7 = px.scatter(df, x=xcol, y=ycol, color=color_on2, title=ycol+" versus "+xcol, color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data)
    # Adjust the chart's height using update_layout
    fig7.update_layout( height=chart_height )  # Set your desired height here

    st.plotly_chart(fig7, use_container_width=True, key="T4C1R1")