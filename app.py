import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from yt_dlp import YoutubeDL
from langchain_community.document_loaders import UnstructuredURLLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.title("✨ YouTube & Web Content Summarizer ✨")
st.write("Enter the URL of the content (YouTube, webpages) you want to summarize:")

# Sidebar for API key
groq_api = os.getenv("GROQ_API_KEY")
with st.sidebar:
   groq_api_key = st.text_input("API Key loads from .env", value=groq_api, type="password")

# Input for URL
generic_url = st.text_input(
    "Enter the URL of the content you want to summarize",
    placeholder="Paste the URL here",
    type="default",
    label_visibility="collapsed",
)


# Initialize LLM (Groq API with chosen model)
llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it", max_tokens=2000)

# Prompt Template for Summarization
prompt_template = """
You are a helpful assistant that creates detailed and comprehensive summaries.
Please provide an in-depth summary of the content below:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Function to fetch YouTube metadata using yt-dlp
def fetch_youtube_metadata(url):
    try:
        with YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "No title available")
            description = info.get("description", "No description available")
            return f"Title: {title}\n\nDescription: {description}"
    except Exception as e:
        raise Exception(f"Error fetching YouTube metadata: {e}")

# Function to summarize any webpage using UnstructuredURLLoader
def fetch_webpage_metadata(url):
    try:
        loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={'User-Agent': 'Mozilla/5.0'})
        docs = loader.load()
        if docs:
            content = docs[0].page_content  # Take the first document's content
            return content
        else:
            raise Exception("No content found at the given URL.")
    except Exception as e:
        raise Exception(f"Error fetching webpage metadata: {e}")

# Summarization logic
if st.button("Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide all the required information to get started.")
    elif not validators.url(generic_url):
        st.error("Please provide a valid URL.")
    else:
        try:
            with st.spinner("Generating detailed summary..."):
                # Determine if the URL is YouTube or a webpage
                if "youtube" in generic_url:
                    content = fetch_youtube_metadata(generic_url)
                else:
                    content = fetch_webpage_metadata(generic_url)

                # Wrap the content in a Document object for LangChain
                docs = [Document(page_content=content)]

                # Create the summarization chain with the specified prompt
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.write(output_summary)

        except Exception as e:
            st.error(f"Error: {str(e)}")