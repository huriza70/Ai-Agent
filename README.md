# AI-Powered Content Creation Assistant

A Streamlit-based app that helps you generate SEO articles, scripts, and social media content using OpenAI and LangChain. It also allows optional publishing to Blogger via Google API.

## Features
- Generate SEO-optimized articles with headings, meta descriptions, and tags.
- Create scripts for video, podcast, short films, commercials, and presentations.
- Generate social media posts with hashtags, emojis, and posting suggestions.
- Save and manage chat sessions.
- Post AI-generated articles directly to Blogger (requires Google OAuth setup).
- Stores past content in a Chroma vector database for context-aware responses.

## Project Structure
.
├── code_1.py # Main Streamlit app
├── content_sessions.json # Stores chat session history
├── content_vectors/ # Chroma vector database (auto-created)
├── requirements.txt # Dependencies
└── README.md # Documentation


## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-content-creator.git
   cd ai-content-creator
   
2. Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3. Install dependencies
   
pip install -r requirements.txt

5. Set up environment variables

Create a .env file in the project root with the following:

OPENAI_API_KEY=your_openai_api_key
BLOGGER_BLOG_ID=your_blogger_blog_id

6. Run the app

streamlit run code_1.py





