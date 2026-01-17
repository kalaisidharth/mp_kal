import streamlit as st
import requests 

API_URL="http://127.0.0.1:8000/process"

st.title("Streamlit to FastAPI")

with st.form("text_form"):
    text=st.text_input("Enter text")
    submitted=st.form_submit_button("Send to API")

if submitted:
    response=requests.post(
        API_URL,
        json={"text":text}
    )
    
    if response.status_code==200:
        data=response.json()
        st.success("Response from API")
        st.json(data)
    else:
        st.error("API call failed")
