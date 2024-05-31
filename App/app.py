import streamlit as st
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, UpdateMode

# Azure Storage and Table credentials
connection_string = "DefaultEndpointsProtocol=https;AccountName=515team2;AccountKey=+wc53G0GKd551uGI/gn+ow5YcrqralBanMwl+MqJoxReUPwSHwBE6wu4Eoh3awBwxR4za3qlC0hQ+AStlJ2PmA==;EndpointSuffix=core.windows.net"
table_name = "mile3"
container_name_raw = "mile3raw"
container_name_processed = "mile3processed"

# Initialize Blob and Table service clients
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_client_raw = blob_service_client.get_container_client(container_name_raw)
blob_client_processed = blob_service_client.get_container_client(container_name_processed)

table_service_client = TableServiceClient.from_connection_string(connection_string)
table_client = table_service_client.get_table_client(table_name)

# Fetch data from Azure Table
entities = table_client.list_entities()
data = []
for entity in entities:
    data.append(entity)
df = pd.DataFrame(data)

# Ensure PartitionKey and RowKey are part of the DataFrame
if 'PartitionKey' not in df.columns or 'RowKey' not in df.columns:
    st.error("The DataFrame must include 'PartitionKey' and 'RowKey' columns.")
    st.stop()

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Function to get image URL from Blob Storage
def get_image_url(blob_service_client, container_name, image_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=image_name)
    return blob_client.url

# Add raw image URLs to the dataframe
raw_image_urls = []
for index, row in df.iterrows():
    image_name = row['RowKey']  # Assuming image names are in the format of RowKey.jpg
    raw_image_url = get_image_url(blob_service_client, container_name_raw, image_name)
    raw_image_urls.append(raw_image_url)

df['RawImage'] = raw_image_urls

# Add processed image URLs to the dataframe
processed_image_urls = []
for index, row in df.iterrows():
    if row['Status'].upper() == "YES":
        raw_image_name = row['RowKey']  # Assuming raw image names are in the format of RowKey.jpg
        processed_image_name = raw_image_name.replace("-1.jpg", "-2.jpg")
        processed_image_url = get_image_url(blob_service_client, container_name_processed, processed_image_name)
        processed_image_urls.append(processed_image_url)
    else:
        processed_image_urls.append(None)

df['ProcessedImage'] = processed_image_urls

# Select and reorder columns
columns_order = ['Date', 'Status', 'Percentage', 'RawImage', 'ProcessedImage', 'TemperatureC', 'Humidity', 'Pressure']
df = df[columns_order]

# Rename columns for display purposes
df = df.rename(columns={'RawImage': 'Preview Image', 'ProcessedImage': 'Processed Image'})

# Overview section - Get the latest record
latest_record = df.iloc[-1]
previous_record = df.iloc[-2] if len(df) > 1 else latest_record

# Calculating changes for temperature and humidity
temp_change = latest_record['TemperatureC'] - previous_record['TemperatureC']
humidity_change = latest_record['Humidity'] - previous_record['Humidity']

# Streamlit app
st.title("🌾FarmBeats Monitor")

# Warning frame if status is "YES"
if latest_record['Status'].upper() == "YES":
    st.markdown(
        '<div style="background-color: #FFF1F1; padding: 8px; border-radius: 12px; border-style: solid; border-color: #FFF1F1; text-align: center; text-color: #FF9090">'
        '❗️The crops may get rusted''</div>',
        unsafe_allow_html=True
    )

# Sidebar for filtering data
with st.sidebar:
    st.caption("Filter data")
    
    # Date range filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # Status filter
    status_filter = st.selectbox('Status', ('ALL', 'YES', 'NO'))
    
    # Percentage filter
    percentage_min, percentage_max = st.slider(
        'Percentage', min_value=int(df['Percentage'].min()), 
        max_value=int(df['Percentage'].max()), 
        value=(int(df['Percentage'].min()), int(df['Percentage'].max())))
    
    # Temperature filter
    temperature_min, temperature_max = st.slider(
        'Temperature (°C)', min_value=int(df['TemperatureC'].min()), 
        max_value=int(df['TemperatureC'].max()), 
        value=(int(df['TemperatureC'].min()), int(df['TemperatureC'].max())))
    
    # Humidity filter
    humidity_min, humidity_max = st.slider(
        'Humidity', min_value=int(df['Humidity'].min()), 
        max_value=int(df['Humidity'].max()), 
        value=(int(df['Humidity'].min()), int(df['Humidity'].max())))

# Apply filters to dataframe
filtered_df = df.copy()
if len(pd.to_datetime(date_range)) == 2:
    start_date, end_date = pd.to_datetime(date_range)
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date + pd.Timedelta(days=1))]

if status_filter != 'ALL':
    filtered_df = filtered_df[filtered_df['Status'].str.upper() == status_filter]

filtered_df = filtered_df[(filtered_df['Percentage'] >= percentage_min) & (filtered_df['Percentage'] <= percentage_max)]
filtered_df = filtered_df[(filtered_df['TemperatureC'] >= temperature_min) & (filtered_df['TemperatureC'] <= temperature_max)]
filtered_df = filtered_df[(filtered_df['Humidity'] >= humidity_min) & (filtered_df['Humidity'] <= humidity_max)]

# Latest Data section
st.subheader("Latest Data")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image(latest_record['Preview Image'], caption='Latest Image', use_column_width=True)
with col2:
    st.write(f"**Date:** {latest_record['Date']}")
    st.write(f"**Status:** {latest_record['Status']}")
    st.write(f"**Percentage:** {latest_record['Percentage']}%")
with col3:
    st.caption("Temperature")
    st.write(f"{latest_record['TemperatureC']}°C")
    if temp_change > 0:
        st.write(f'<p style="color:red;">↑ {temp_change:.2f}°C Compared to previous</p>', unsafe_allow_html=True)
    else:
        st.write(f'<p style="color:red;">↓ {abs(temp_change):.2f}°C Compared to previous</p>', unsafe_allow_html=True)
    
    st.caption("Humidity")
    st.write(f"{latest_record['Humidity']}%")
    if humidity_change > 0:
        st.write(f'<p style="color:red;">↑ {humidity_change:.2f}% Compared to previous</p>', unsafe_allow_html=True)
    else:
        st.write(f'<p style="color:red;">↓ {abs(humidity_change):.2f}% Compared to previous</p>', unsafe_allow_html=True)

# Function to update status in Azure Table
def update_status(partition_key, row_key, new_status):
    entity = table_client.get_entity(partition_key=partition_key, row_key=row_key)
    entity['Status'] = new_status
    table_client.update_entity(mode=UpdateMode.REPLACE, entity=entity)

# Data History section with editable status
st.subheader("Past Data")
edited_df = st.data_editor(
    filtered_df,
    column_config={
        "Preview Image": st.column_config.ImageColumn(
            "Preview Image", help="Streamlit app preview screenshots"
        ),
        "Processed Image": st.column_config.ImageColumn(
            "Processed Image", help="Processed images for status 'YES'"
        ),
        "Status": st.column_config.SelectboxColumn(
            "Status", options=['YES', 'NO']
        )
    },
    hide_index=True,
)

# Detect changes in the "Status" column and update Azure Table
if st.button("Save Changes"):
    for index, row in edited_df.iterrows():
        original_status = filtered_df.loc[index, 'Status']
        new_status = row['Status']
        if original_status != new_status:
            update_status(row['PartitionKey'], row['RowKey'], new_status)
    st.success("Changes saved successfully!")
