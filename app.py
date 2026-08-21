import streamlit as st
from src.search_engine import search_movies
from src.query_parser import parse_query
from src.generator import generate_rag_response

st.set_page_config(page_title="MovieSense RAG", layout="centered")

with st.sidebar:
    st.title("Settings")
    user_api_key = st.text_input("Groq API Key (Optional)", type="password", help="Leave blank to use the backend default key.")

# Determine which API key to use
backend_api_key = st.secrets.get("GROQ_API_KEY", "")
active_api_key = user_api_key.strip() if user_api_key.strip() else backend_api_key

st.title("MovieSense")
st.caption("Semantic Movie Search with Intelligent Filtering")

# Initialize session state for preferences
if "preferences" not in st.session_state:
    st.session_state.preferences = {
        "action_weight": 5,
        "comedy_weight": 5,
        "drama_weight": 5,
        "scifi_weight": 5,
        "dealbreakers": [],
        "mood": "Balanced"
    }

tab_search, tab_profile = st.tabs(["Search", "Profile Dashboard"])

with tab_profile:
    st.header("Your Movie Profile")
    st.write("Customize your preferences to get deterministic, tailored recommendations.")
    
    st.subheader("Why use MovieSense instead of ChatGPT?")
    st.info(
        "Generic LLMs (like ChatGPT) struggle with strict negative constraints (e.g., 'Do not show me Horror') "
        "and precise weighting (e.g., 'I want 80% Action and 20% Comedy'). MovieSense uses your structured profile "
        "to enforce hard dealbreakers and exact genre weighting, while still using semantic AI to understand the 'vibe' "
        "of your natural language search."
    )

    st.subheader("Genre Preferences")
    st.write("Set your exact preferences (0 = Hate it, 10 = Love it)")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.preferences["action_weight"] = st.slider("Action", 0, 10, st.session_state.preferences["action_weight"])
        st.session_state.preferences["comedy_weight"] = st.slider("Comedy", 0, 10, st.session_state.preferences["comedy_weight"])
    with col2:
        st.session_state.preferences["drama_weight"] = st.slider("Drama", 0, 10, st.session_state.preferences["drama_weight"])
        st.session_state.preferences["scifi_weight"] = st.slider("Sci-Fi", 0, 10, st.session_state.preferences["scifi_weight"])

    st.subheader("Dealbreakers")
    st.write("Select themes or elements you absolutely do not want to see.")
    st.session_state.preferences["dealbreakers"] = st.multiselect(
        "Exclude these categories",
        ["Horror", "Gore", "Slow-paced", "Black and White", "Musicals"],
        default=st.session_state.preferences["dealbreakers"]
    )

    st.subheader("Default Mood")
    st.session_state.preferences["mood"] = st.selectbox(
        "What vibe do you usually lean towards?",
        ["Balanced", "Fast-paced and Exciting", "Slow-burn and Thoughtful", "Dark and Gritty", "Lighthearted and Fun"],
        index=["Balanced", "Fast-paced and Exciting", "Slow-burn and Thoughtful", "Dark and Gritty", "Lighthearted and Fun"].index(st.session_state.preferences["mood"])
    )
    
    st.success("Profile saved automatically. Your searches will now prioritize these precise constraints!")

with tab_search:
    st.write("Chat with the MovieSense RAG engine (Powered by local semantic search + Groq LLM)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if query := st.chat_input("Describe the movies you're looking for..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Process the query
        with st.chat_message("assistant"):
            filters = parse_query(query)
            with st.expander("Detected natural language filters"):
                st.json(filters)
            
            with st.expander("Applied Profile Constraints"):
                st.json(st.session_state.preferences)

            # Retrieve movies via semantic search!
            results = search_movies(query, top_k=5, preferences=st.session_state.preferences)
            
            if not results:
                response = "No movies matched your filters. Try adjusting your query or your profile dealbreakers."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                if not active_api_key:
                    st.warning("Please enter your Groq API Key in the sidebar or set it in the backend to get conversational RAG responses.")
                    
                    # Fallback to standard listing
                    response_str = "Here are the top matches based on your query and profile constraints:\n\n"
                    st.markdown(response_str)
                    
                    for i, res in enumerate(results, 1):
                        st.subheader(f"{i}. {res['title']} ({res['release_year']})")
                        st.write(f"**Genre:** {res['genre']}  |  **Rating:** {res['rating']}  |  **Score:** {res['score']:.4f}")
                        st.write(res['description'])
                        st.markdown("---")
                        
                        response_str += f"**{i}. {res['title']} ({res['release_year']})**\n"
                        response_str += f"Genre: {res['genre']} | Rating: {res['rating']} | Score: {res['score']:.4f}\n"
                        response_str += f"{res['description']}\n\n"
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_str})
                else:
                    # Generate RAG response
                    response_stream = generate_rag_response(
                        query=query,
                        retrieved_movies=results,
                        api_key=active_api_key,
                        preferences=st.session_state.preferences
                    )
                    
                    full_response = st.write_stream(response_stream)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
