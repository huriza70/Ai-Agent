import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import uuid
from datetime import datetime
import json
import os
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

# Load environment variables
load_dotenv()

# Initialize embeddings and vectorstore
embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma(
    persist_directory='content_vectors', 
    embedding_function=embedding_function, 
    collection_name="content_creation"
)

# Session management functions (similar to your medical app)
def load_sessions():
    try:
        with open("content_sessions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_sessions(sessions):
    with open("content_sessions.json", "w") as f:
        json.dump(sessions, f, indent=2)

def create_new_session():
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "id": session_id,
        "title": f"Session {timestamp}",
        "messages": [],
        "created_at": timestamp
    }

def get_session_title(messages):
    if messages:
        first_user_msg = next((msg[1] for msg in messages if msg[0] == "user"), "")
        if first_user_msg:
            title = first_user_msg[:30] + ("..." if len(first_user_msg) > 30 else "")
            return title
    return "New Session"

# Content-specific functions
def store_content_interaction(query, response):
    content = f'Query: {query}\nResponse: {response}'
    doc = Document(page_content=content)
    vectorstore.add_documents([doc])
    vectorstore.persist()

def get_content_context(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.get_relevant_documents(query)
    similar_content = []
    
    for content in results:
        text = content.page_content
        similar_content.append(text)
    
    return '\n---\n'.join(similar_content)

def post_to_blogger(title, content):
    """Post content to Blogger (requires OAuth setup)"""
    try:
        # This is a simplified version - you'd need proper OAuth credentials
        creds = Credentials.from_authorized_user_file('token.json')
        service = googleapiclient.discovery.build('blogger', 'v3', credentials=creds)
        
        blog_id = os.getenv("BLOGGER_BLOG_ID")
        
        body = {
            "kind": "blogger#post",
            "title": title,
            "content": content
        }
        
        post = service.posts().insert(blogId=blog_id, body=body).execute()
        return f"Posted successfully! URL: {post.get('url')}"
    except Exception as e:
        return f"Error posting to Blogger: {str(e)}"

# Initialize language model
llm_openai = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

# Content creation templates
content_templates = {
    "seo_article": """
    As an expert SEO content writer, create a comprehensive article about: {topic}
    
    Additional instructions: {instructions}
    
    Similar successful content examples:
    {context}
    
    Please provide:
    1. A compelling title with primary keyword
    2. Meta description with keywords
    3. Well-structured article with headings (H2, H3)
    4. Integration of relevant keywords naturally
    5. A call-to-action conclusion
    6. 5 suggested tags
    
    Write in a {tone} tone and aim for approximately {word_count} words.
    """,
    
    "script": """
    As a creative scriptwriter, create a {script_type} about: {topic}
    
    Additional instructions: {instructions}
    
    Similar successful content examples:
    {context}
    
    Please provide:
    1. Engaging opening
    2. Character descriptions (if applicable)
    3. Scene settings
    4. Dialogue with natural flow
    5. Plot development with conflict/resolution
    6. Memorable ending
    
    Write in a {tone} tone.
    """,
    
    "social_media": """
    As a social media manager, create {platform} content about: {topic}
    
    Additional instructions: {instructions}
    
    Similar successful content examples:
    {context}
    
    Please provide:
    1. Attention-grabbing caption
    2. Relevant hashtags (10-15)
    3. Emoji suggestions
    4. Call-to-action
    5. Suggested posting time
    
    Write in a {tone} tone.
    """
}

# Streamlit UI setup
st.set_page_config(page_title="AI Content Creator", layout="centered")
st.title("🤖 AI-Powered Content Creation Assistant")
st.markdown("Create SEO articles, scripts, and social media content with AI")

# Initialize session state
if "current_session_id" not in st.session_state:
    new_session = create_new_session()
    st.session_state.current_session_id = new_session["id"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "content_type" not in st.session_state:
    st.session_state.content_type = "seo_article"

# Sidebar for chat history and content type selection
with st.sidebar:
    st.title("💬 Chat History")
    
    # Load all sessions
    all_sessions = load_sessions()
    
    # Button to create new session
    if st.button("➕ New Session", use_container_width=True):
        if "current_session_id" in st.session_state and "chat_history" in st.session_state:
            if st.session_state.chat_history:
                current_session = {
                    "id": st.session_state.current_session_id,
                    "title": get_session_title(st.session_state.chat_history),
                    "messages": st.session_state.chat_history,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                all_sessions[st.session_state.current_session_id] = current_session
                save_sessions(all_sessions)
        
        # Create new session
        new_session = create_new_session()
        st.session_state.current_session_id = new_session["id"]
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.subheader("Previous Sessions")
    
    sorted_sessions = sorted(all_sessions.items(), key=lambda x: x[1]["created_at"], reverse=True)
    for session_id, session in sorted_sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(session["title"], key=session_id, use_container_width=True):
                st.session_state.current_session_id = session_id
                st.session_state.chat_history = session["messages"]
                st.rerun()
        with col2:
            if st.button("🗑", key=f"delete_{session_id}"):
                del all_sessions[session_id]
                save_sessions(all_sessions)
                if st.session_state.current_session_id == session_id:
                    new_session = create_new_session()
                    st.session_state.current_session_id = new_session["id"]
                    st.session_state.chat_history = []
                    st.rerun()
    
    st.divider()
    st.subheader("Content Type")
    st.session_state.content_type = st.selectbox(
        "Select content type:",
        options=["seo_article", "script", "social_media"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    # Content type specific options
    if st.session_state.content_type == "seo_article":
        st.session_state.word_count = st.slider("Word count", 500, 2000, 1000)
    elif st.session_state.content_type == "script":
        st.session_state.script_type = st.selectbox(
            "Script type:",
            options=["video", "podcast", "short_film", "commercial", "presentation"]
        )
    elif st.session_state.content_type == "social_media":
        st.session_state.platform = st.selectbox(
            "Platform:",
            options=["Instagram", "Twitter", "Facebook", "LinkedIn", "TikTok"]
        )
    
    st.session_state.tone = st.selectbox(
        "Tone:",
        options=["Professional", "Casual", "Informative", "Persuasive", "Humorous", "Inspirational"]
    )

# Main chat interface
if st.session_state.chat_history:
    current_title = get_session_title(st.session_state.chat_history)
    st.info(f"Current Session: {current_title}")

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == 'user':
        with st.chat_message('user'):
            st.write(msg)
    else:
        with st.chat_message('assistant'):
            st.write(msg)
            # Add post to blogger button for AI responses
            if role == 'bot' and "article" in st.session_state.content_type:
                if st.button("Post to Blogger", key=f"blog_{len(st.session_state.chat_history)}"):
                    # Extract title and content (this would need more sophisticated parsing)
                    lines = msg.split('\n')
                    title = lines[0] if lines else "AI Generated Content"
                    post_result = post_to_blogger(title, msg)
                    st.session_state.chat_history.append(("system", post_result))
                    st.rerun()

# User input
user_input = st.chat_input("What content would you like to create today?", key="user_input")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    
    # Get context from similar past content
    context = get_content_context(user_input)
    
    # Prepare prompt based on content type
    if st.session_state.content_type == "seo_article":
        template = content_templates["seo_article"]
        prompt = PromptTemplate(
            template=template,
            input_variables=["topic", "instructions", "context", "tone", "word_count"]
        )
        chain = LLMChain(llm=llm_openai, prompt=prompt)
        response = chain.run({
            "topic": user_input,
            "instructions": "",
            "context": context,
            "tone": st.session_state.tone,
            "word_count": st.session_state.word_count
        })
    
    elif st.session_state.content_type == "script":
        template = content_templates["script"]
        prompt = PromptTemplate(
            template=template,
            input_variables=["script_type", "topic", "instructions", "context", "tone"]
        )
        chain = LLMChain(llm=llm_openai, prompt=prompt)
        response = chain.run({
            "script_type": st.session_state.script_type,
            "topic": user_input,
            "instructions": "",
            "context": context,
            "tone": st.session_state.tone
        })
    
    elif st.session_state.content_type == "social_media":
        template = content_templates["social_media"]
        prompt = PromptTemplate(
            template=template,
            input_variables=["platform", "topic", "instructions", "context", "tone"]
        )
        chain = LLMChain(llm=llm_openai, prompt=prompt)
        response = chain.run({
            "platform": st.session_state.platform,
            "topic": user_input,
            "instructions": "",
            "context": context,
            "tone": st.session_state.tone
        })
    
    st.session_state.chat_history.append(("bot", response))
    store_content_interaction(user_input, response)
    
    # Update sessions
    all_sessions = load_sessions()
    current_session = {
        "id": st.session_state.current_session_id,
        "title": get_session_title(st.session_state.chat_history),
        "messages": st.session_state.chat_history,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    all_sessions[st.session_state.current_session_id] = current_session
    save_sessions(all_sessions)
    
    st.rerun()
