import streamlit as st    
def page_style():
    style = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://i.ibb.co/d2vsFP8/valli2.jpg");
    background-size: cover;
    }
    [data-testid="stVerticalBlock"]{
    background-color: rgb(14, 17, 23, 0.75);
    padding-block: 10px 20px;
    writing-mode: horizontal-tb;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.7);
    }
    [data-testid="element-container"]{
    padding: 5px 20px 15px 20px;
    text-align: center;
    }
    .stSlider [data-baseweb=slider]{
        width: 95%;
        padding: 5px 20px 15px 20px;
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)