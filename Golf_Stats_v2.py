import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import statsmodels.api as sm
# Force the default template to "plotly"
pio.templates.default = "plotly"

sns.set_theme(style="darkgrid")

#-----------------------  has to be called first
# Set the page layout to wide
st.set_page_config(layout="wide")

# TO RUN THIS, USE TERMINAL
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

#### Helper functions   ################################################################
# Convert value with error handling
def convert_value(value):
    """
    Converts any value like '40R' or '20L' to numerical form:
    - '40R' becomes -40
    - '20L' becomes 20
    Returns None for invalid entries.
    ## USED in convert_column ##
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
    ## USED in process_df ##
    """
    if col in df.columns:
        df[col] = df[col].apply(convert_value)
        #print(f"Column '{col}' successfully converted.")
    else:
        st.error(f"The column '{col}' is missing from the data.")
######################################################################################################################################
# Cleaning the column names and getting df ready for the rest of the application
# Clean column names to ensure consistency
df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace('\xa0', ' ').str.strip().str.replace(' ', '_')

# Ensure Time column is in datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Convert Time to strings for Session variable
df['Session'] = df['Time'].dt.strftime('%Y %b %d %I:%M %p')  # Or any simpler string representation

# Convert all the columns that have L & R in the data        
lrcols = ['Swing_H','Spin_Axis','Lateral_yds','FTP','FTT','Club_Path','Launch_H']
for col in lrcols:
    convert_column(df,col)

# 1. Parse & sort
df['Session_dt'] = pd.to_datetime(df['Session'], format='%Y %b %d %I:%M %p')
df = df.sort_values('Session_dt').copy()
# 2. Convert Session to ordered categorical
sorted_sessions = df['Session'].unique()
df['Session'] = pd.Categorical(
    df['Session'],
    categories=sorted_sessions,
    ordered=True             )

df['Shot_Type'] = df['Shot_Type'].astype(str)

# Columns Available: ['Mombo_ShotID', 'Club', 'Time', 'Golfer', 'Shot', 'Video', 'Ball mph','Club mph', 'Smash_Factor', 'Carry yds', 'Total yds', 'Roll yds',
#       'Swing H', 'Spin rpm', 'Height ft', 'Time s', 'AOA', 'Spin Loft','Swing V', 'Spin Axis', 'Lateral yds', 'Shot Type', 'FTP', 'FTT',
#       'Dynamic Loft', 'Club Path', 'Launch H', 'Launch V', 'Low Point ftin','DescentV', 'Curve Dist yds', 'Lateral Impact in', 'Vertical Impact in',
#       'Mode', 'Location', 'Unnamed_35', 'Unnamed_36', 'Unnamed_37','Unnamed_38', 'Unnamed_39', 'Unnamed_40', 'Comment', 'User1', 'User2','Exclude', 'Session']

numcols = ['Ball_mph','Club_mph','Smash_Factor','Carry_yds','Total_yds','Roll_yds',
           'Swing_H','Spin_rpm','Height_ft','Time_s','AOA','Spin_Loft','Swing_V','Spin_Axis','Lateral_yds','FTP','FTT',
           'Dynamic_Loft','Club_Path','Launch_H','Launch_V','Low_Point_ftin','DescentV','Curve_Dist_yds','Lateral_Impact_in','Vertical_Impact_in']

###################################################################################################################################################################

def process_df(df):
    clubs = ['Driver','3 Wood','5 Wood','4 Iron','5 Iron','6 Iron','7 Iron', '8 Iron','9 Iron', 'Pitching Wedge',  'Gap Wedge','Sand Wedge' , 'Lob Wedge']
    dimensions = ['Club','Golfer','Session','Shot_Type','Mode']
    num_columns = ['Ball_mph', 'Club_mph', 'Smash_Factor', 'Carry_yds','Total_yds', 'Roll_yds', 'Swing_H', 'Height_ft', 'Time_s', 'AOA',
       'Spin_Loft', 'Swing_V', 'Spin_Axis', 'Lateral_yds', 'FTP', 'FTT','Dynamic_Loft', 'Club_Path', 'Launch_H', 'Launch_V', 'DescentV',
       'Curve_Dist_yds', 'Lateral_Impact_in', 'Vertical_Impact_in']
    

    # 3. Pivot with Sessions
    df_pivot = df.pivot_table(
        index='Club',
        columns=['Golfer', 'Session'],
        values=num_columns, 
        aggfunc='mean')
    # 4. (Optional) Reindex columns to ensure Session is in ascending chronological order
    df_sessions = df_pivot.reindex(columns=sorted_sessions, level='Session')

    df_golfer = df.pivot_table(
        index='Club',
        columns=['Golfer'],
        values=num_columns, 
        aggfunc='mean')
    df_golfer = df_golfer.reindex(clubs)       # Puts the clubs in the order of the list 'clubs'
    df_sessions = df_sessions.reindex(clubs)   # Puts the clubs in the order of the list 'clubs'
    
    return df,df_sessions,df_golfer



# Ensure numeric conversion for specified columns
def ensure_numeric(df, columns):
    """
    Converts specified columns to numeric, coercing errors to NaN.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.error(f"Column '{col}' not found in the dataframe.")

# Apply numeric conversion to the specified numeric columns
ensure_numeric(df, numcols)

########################################################   PROCESSING COMPLETE ##########################################

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
choices = ['Time', 'Golfer', 'Club', 'Shot_Type']
color_on = st.sidebar.selectbox('Select ColorOn', choices)



#----------------------------------------------------------------------------------------------------------------
hov_data = ['Time', 'Club', 'Golfer', 'Shot_Type','Total_yds']

#################    PROCESS THE DF FILE.  CALC AVGS BY SESSION AND GOLFER TOTAL AFTER FILTERING ##############################

df, df_sessions, df_golfer = process_df(df)       # #  Club in rows, Golfer, Session in Columns as MultiIndex  (all metrics)



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

def create_fig6(var_choice,colvar):
    # Create a box plot
    #  var_choice = "Carry_yds"   (Example)
    fig6 = px.box(df, x='Session', y=var_choice, points='all', color=colvar,hover_data=hov_data)
    mean_values = df.groupby('Session')[var_choice].median().reset_index()

    # Add the median as text annotations
    for i, row in mean_values.iterrows():
        fig6.add_annotation(
            x=row['Session'],
            y=row[var_choice],
            text=f"{row[var_choice]:.1f}",
            showarrow=False,  # Avoid cluttering with arrows
            font=dict(size=12, color='black'),
            bgcolor="white"
        )
    return fig6

###################################################################################################################

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["4 Plots", "BoxPlots","Stats","Plotchoice","Seaborn","Parameters","Distances"])

with tab1:                                                                          ## TAB 1 ##
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

with tab2:                                                                          ## TAB 2 ##
    with st.container():
        row1 = st.columns([1, 7, 1])
        with row1[1]:
            st.plotly_chart(fig1, use_container_width=True, key="T2C1R1")

    bottom_row = st.columns([1,6])

    with bottom_row[0]:
        boxplot_metric = st.selectbox('Select Metric for Boxplot', numcols)

    with bottom_row[1]:
        fig6 = create_fig6(boxplot_metric,color_on)
        st.write("Box Plot for "+boxplot_metric)
        st.plotly_chart(fig6, use_container_width=True, key="T2C3R3")
with tab3:                                                                          ## TAB 3 ##
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
with tab4:                                                                          ## TAB 4 ##
    # Create 4 columns for the inputs to be contained
    col1, col2, col3, col4 = st.columns(4)
    # Place the first selectbox in the first column
    with col1:
        ycol = st.selectbox('Select column for y axis', numcols) 
    # Place the second selectbox in the second column
    with col2:
        xcol = st.selectbox('Select column for x axis', numcols)
    with col3:
        # Choose what to color on separately for this tab (ie color_on2)
        choices = ['Time', 'Golfer', 'Club', 'Shot_Type']
        color_on2 = st.selectbox('Select ColorOn 2', choices)
    with col4:
        # Allow the user to set the chart height dynamically
        chart_height = st.slider('Set chart height (in pixels)', min_value=400, max_value=1200, value=800, step=50)

##### fig7 #####
    ##### fig7 #####
    fig7 = px.scatter(df, x=xcol, y=ycol, color=color_on2, title=ycol+" versus "+xcol, color_discrete_sequence=px.colors.qualitative.Bold, hover_data=hov_data,trendline='ols')
    # Adjust the chart's height using update_layout
    fig7.update_layout( height=chart_height )  # Set your desired height here
    st.plotly_chart(fig7, use_container_width=True, key="T4C1R1")

with tab5:                                                                          ## TAB 5 ##
    ###################################### Seaborn Statistical Plots ##########################################################################
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

with tab6:                                                                          ## TAB 6 ##
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

with tab7:
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