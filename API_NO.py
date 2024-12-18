import streamlit as st
from transformers import pipeline
from docx import Document

# Function to convert .docx to text
def docx_to_text(docx_file):
    doc = Document(docx_file)
    text_content = ""
    for para in doc.paragraphs:
        text_content += para.text + "\n\n"
    return text_content.strip()

# Streamlit app setup
st.title("LLM Interaction App Without OpenAI API Key")

# Input fields for user interaction
location = st.text_input("Location")
model_name = st.selectbox("Select Model", ["facebook/bart-large-cnn"])  # Summarization model
project_id = st.text_input("Project ID")

# File upload section
uploaded_file = st.file_uploader("Upload a .docx file", type=["docx"])
user_message = st.text_area("Enter your message")

# Load the Hugging Face model pipeline for summarization
@st.cache_resource
def load_model(model_name):
    return pipeline("summarization", model=model_name)

model_pipeline = load_model(model_name)

# Process the input when the button is clicked
if st.button("Submit"):
    try:
        if uploaded_file is not None:
            # Convert .docx to text and summarize
            text_content = docx_to_text(uploaded_file)

            # Check if the text is long enough for summarization
            if len(text_content) < 50:  # Adjust this threshold as needed
                st.warning("The document is too short for meaningful summarization.")
            else:
                # Summarize the text content using the model
                response = model_pipeline(text_content, max_length=150, min_length=30, do_sample=False)

                # Ensure response is a list and has at least one element
                if isinstance(response, list) and len(response) > 0:
                    # Display the original text and summary on the same page
                    st.subheader("Original Text from DOCX:")
                    st.write(text_content)

                    st.subheader("Summary from LLM:")
                    summary_text = response[0].get('summary_text', "No summary available.")
                    st.write(summary_text)
                else:
                    st.error("No valid response received from the summarization model.")

        elif user_message:
            # Send user message to selected LLM (for now, we won't summarize user messages)
            response = model_pipeline(user_message, max_length=150, min_length=30, do_sample=False)  # Adjust parameters as needed

            # Ensure response is a list and has at least one element
            if isinstance(response, list) and len(response) > 0:
                # Display user message and response on the same page
                st.subheader("User Message:")
                st.write(user_message)

                st.subheader("Response from LLM:")
                summary_text = response[0].get('summary_text', "No summary available.")
                st.write(summary_text)
            else:
                st.error("No valid response received from the summarization model.")

        else:
            st.error("Please provide either a message or upload a file.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
