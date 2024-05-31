import streamlit as st
import pandas as pd
from azure.data.tables import TableServiceClient
import plotly.express as px

# Function to retrieve data from Azure Table Storage
def fetch_data_from_azure_table(connection_string, table_name):
    service = TableServiceClient.from_connection_string(conn_str=connection_string)
    table_client = service.get_table_client(table_name=table_name)
    
    entities = table_client.list_entities()
    data = []
    for entity in entities:
        data.append(entity)
    return pd.DataFrame(data)

# Azure Table Storage information
connection_string = 'DefaultEndpointsProtocol=https;AccountName=515team2;AccountKey=+wc53G0GKd551uGI/gn+ow5YcrqralBanMwl+MqJoxReUPwSHwBE6wu4Eoh3awBwxR4za3qlC0hQ+AStlJ2PmA==;EndpointSuffix=core.windows.net'
table_name = 'mile3'

# Fetch the data
data_df = fetch_data_from_azure_table(connection_string, table_name)

# Convert the Date column to datetime
data_df['Date'] = pd.to_datetime(data_df['Date'])

# Sidebar for date range selection
st.sidebar.caption("Filter by Date Range")
start_date = st.sidebar.date_input("Start Date", value=data_df['Date'].min())
end_date = st.sidebar.date_input("End Date", value=data_df['Date'].max())

# Filter the data based on the selected date range
filtered_df = data_df[(data_df['Date'] >= pd.to_datetime(start_date)) & (data_df['Date'] <= pd.to_datetime(end_date) + pd.DateOffset(days=1) - pd.Timedelta(seconds=1))]

# Set up the Streamlit app layout
st.header("📊 FarmBeats Data Visualization")

# Create tabs
tab1, tab2 = st.tabs(["See Trends", "See Relationship"])

with tab1:
    # 1. Percentage-date变化趋势
    fig_percentage = px.line(filtered_df, x='Date', y='Percentage', title='Percentage of Rust Over Time')
    st.plotly_chart(fig_percentage)

    # 2. 三个因素（Temperature，Humidity，Pressure）综合的变化趋势
    fig_combined = px.line(filtered_df, x='Date', y=['TemperatureC', 'Humidity', 'Pressure'], 
                           title='Temperature, Humidity, and Pressure Over Time',
                           labels={'value': 'Measurement', 'variable': 'Parameter'})
    st.plotly_chart(fig_combined)

    # 3. 三个因素个字的变化趋势（三个图展示在一行，in 3 columns)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_temp = px.line(filtered_df, x='Date', y='TemperatureC', title='Temperature Over Time')
        fig_temp.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_temp, use_container_width=True)
    with col2:
        fig_humidity = px.line(filtered_df, x='Date', y='Humidity', title='Humidity Over Time')
        fig_humidity.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_humidity, use_container_width=True)
    with col3:
        fig_pressure = px.line(filtered_df, x='Date', y='Pressure', title='Pressure Over Time')
        fig_pressure.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pressure, use_container_width=True)

with tab2:
    # 4. 两张scatter 图
    # Define color mapping for the scatter chart
    color_mapping = {'YES': '#FF9090', 'NO': '#A4FBAD'}

    # Create the scatter chart for Temperature
    fig_scatter_temp = px.scatter(filtered_df, x='TemperatureC', y='Percentage', color='Status', 
                                  color_discrete_map=color_mapping, title='Temperature vs Percentage',
                                  labels={'TemperatureC': 'Temperature (°C)', 'Percentage': 'Percentage of Yellow'},
                                  category_orders={'Status': ['YES', 'NO']})
    fig_scatter_temp.update_layout(legend_title_text='Status')
    st.plotly_chart(fig_scatter_temp)

    # Create the scatter chart for Humidity
    fig_scatter_humidity = px.scatter(filtered_df, x='Humidity', y='Percentage', color='Status', 
                                      color_discrete_map=color_mapping, title='Humidity vs Percentage',
                                      labels={'Humidity': 'Humidity (%)', 'Percentage': 'Percentage of Yellow'},
                                      category_orders={'Status': ['YES', 'NO']})
    fig_scatter_humidity.update_layout(legend_title_text='Status')
    st.plotly_chart(fig_scatter_humidity)
