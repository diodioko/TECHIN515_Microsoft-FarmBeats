import os
from dotenv import load_dotenv
import streamlit as st
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient
import pandas as pd
import google.generativeai as genai
import re

# Load API key from environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# Configure the API
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

# Azure Storage credentials
connection_string = 'DefaultEndpointsProtocol=https;AccountName=515team2;AccountKey=+wc53G0GKd551uGI/gn+ow5YcrqralBanMwl+MqJoxReUPwSHwBE6wu4Eoh3awBwxR4za3qlC0hQ+AStlJ2PmA==;EndpointSuffix=core.windows.net'
table_name = 'mile3'
container_name_raw = 'mile3raw'

# Initialize Blob and Table service clients
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
table_service_client = TableServiceClient.from_connection_string(connection_string)

# Get table client
table_client = table_service_client.get_table_client(table_name=table_name)

def fetch_data_from_azure():
    entities = table_client.list_entities()
    data = [entity for entity in entities]
    df = pd.DataFrame(data)
    return df

def extract_date_from_rowkey(rowkey):
    match = re.search(r'image_(\d{8})-(\d{6})-\d+\.jpg', rowkey)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        date = f"{date_str[4:6]}/{date_str[6:8]}/{date_str[:4]}"
        time = f"{time_str[:2]}:{time_str[2:4]}"
        return date, time
    return None, None

def generate_content(prompt):
    response = model.generate_content(prompt)
    return response.text

# Streamlit App
st.header("🤖Farmbeats ChatBot")

# Fetch data from Azure
data = fetch_data_from_azure()

# Ensure the necessary columns exist
required_columns = ['RowKey', 'Percentage', 'TemperatureC', 'TemperatureF', 'Pressure', 'Humidity', 'Date']
for column in required_columns:
    if column not in data.columns:
        st.error(f"The required column '{column}' is not present in the data.")
        st.stop()



# Explanation of Azure table data
azure_table_explanation = """
The Percentage shows how much rust appears on wheat leaves. 
Status 'YES' means the presence of disease, 'NO' means no disease. 
The Azure table contains the following fields: Percentage, Status, TemperatureC, TemperatureF, Pressure, Humidity, RowKey (which contains the date and time information).
"""

# Template for handling general questions
question_prompt_template = f"""
You are an AI assistant with access to plant health data. The data includes fields such as Percentage, Status, TemperatureC, TemperatureF, Pressure, Humidity, and RowKey (which contains the date and time information).
Here is the data:
{azure_table_explanation}
{data.to_string()}

Question: {{question}}

Based on the provided data, generate a detailed answer to the question.
"""

def handle_general_question(question, data):
    prompt = question_prompt_template.format(question=question)
    return generate_content(prompt)

# Streamlit Interface for Chat
st.subheader("Chat with Generative AI")
user_input = st.text_input("Ask me anything about the data:")
if user_input:
    ai_response = handle_general_question(user_input, data)
    st.write(ai_response)
