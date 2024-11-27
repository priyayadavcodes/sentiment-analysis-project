import os
import torch
import logging
import json
import streamlit as st
from PIL import Image
from src.utils import preprocess_text, load_vectorizer, load_model, make_prediction


log_dir = os.path.join('log')
log_file = os.path.join(log_dir, 'app.log')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
# Set Streamlit page configuration
st.set_page_config(
    page_title="Sentiment Analysis Project",
    page_icon="😇",
    layout="centered",
)

# CSS to hide the Streamlit main menu, deploy button, and status indicator
hide_streamlit_style = """
    <style>
    /* Hide the Main Menu */
    #MainMenu {visibility: hidden;}

    /* Hide the Running status indicator */
    div[data-testid="stStatusWidget"] {visibility: hidden;}

    /* Hide the footer */
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def get_image_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
def display_flow_image():
    # Construct OS-independent path
    image_path = os.path.join("imgs", "dev-flow.png")
    
    if os.path.exists(image_path):
        # Set the image size as desired for viewing (100% width, 70% height)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{get_image_base64(image_path)}" 
                     alt="Application Development Flow" 
                     style="width: 100%; height: 70%;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Image 'dev-flow.png' not found.")
        logger.error("File 'dev-flow.png' not found. Cannot display application development flow image.")




def load_slang_dictionary():
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'slang_and_short_forms.json')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            slang_dict = json.load(file)
        logger.info("Slang dictionary loaded successfully")
        return slang_dict
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding slang dictionary JSON file: {e}")
        return {}

def perform_sentiment_analysis(text, device='cpu'):
    logger.info("Starting sentiment analysis")
    slang_dict = load_slang_dictionary()
    preprocessed_text = preprocess_text(text, slang_dict)
    vectorizer = load_vectorizer()
    
    if not vectorizer:
        logger.error("Failed to load vectorizer")
        return "Error loading vectorizer"
    
    try:
        review_vec = vectorizer.transform([preprocessed_text])
        review_tensor = torch.tensor(review_vec.toarray(), dtype=torch.float32).to(device)
        logger.info("Text converted to tensor successfully")
    except Exception as e:
        logger.error(f"Error transforming text to tensor: {e}")
        return f"Error transforming text to tensor: {e}"
    
    model = load_model(device)
    if not model:
        logger.error("Failed to load model")
        return "Error loading model"
    
    return make_prediction(review_tensor, model)

# Main UI

# Main UI
st.markdown(
    """
    <h1 style='position: fixed; top: 6.5%; left: 8.5%; transform: translateX(-50%); color: #148f77; padding: 5px;'>
    Project
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h5 style='position: fixed; top: 13%; left: 14%; transform: translateX(-50%); color: #148f77; padding: 5px;'>
    by Priya Yadav
    </h5>
    """,
    unsafe_allow_html=True
)

# Navigation button for GitHub Repository
st.markdown(
    """
    <div style="position: fixed; top: 13%; right: 14%;">
        <a href="https://github.com/priyayadavcodes/sentiment-analysis-project" 
           target="_blank" 
           style="background-color: #535353; color: white; padding: 5px 20px; border-radius: 5px; text-decoration: none; font-size: 16px;">
           GitHub Repository
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


# App Page
st.title("Sentiment Analysis App")
st.markdown("""
- **Enter Review**: Type or paste your review (minimum 10 words required).
- **Click "Analyze"**: To start the sentiment analysis.
- **Word Check**: A warning appears if fewer than 10 words or if text is empty.  
  If so, retry with more words.
""")

text_input = st.text_area("Enter your review for sentiment analysis (minimum 10 words required):", "")
word_count = len(text_input.split())

if st.button("Analyze"):
    if word_count >= 10:
        with st.spinner("Analyzing sentiment..."):
            logger.info(f"User input received for analysis: {text_input}")
            sentiment = perform_sentiment_analysis(text_input)
            st.success(f"Sentiment: {sentiment}")
    elif word_count > 0:
        st.warning("Please enter at least 10 words for analysis.")
    else:
        st.warning("Please enter some text to analyze.")