import streamlit as st
from langchain_helper import generate_restaurant_name_and_items

st.title("Restaurant Name Generator")
cuisine = st.sidebar.selectbox("Select a cuisine", ("Indian", "Chinese", "Italian", "Mexican", "Japanese"))

if cuisine:
    response = generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].strip().split(',')
    st.write("**Menu Items**")
    for item in menu_items:
        st.write("-", item)