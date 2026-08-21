import os
from typing import List, Dict, Generator
from groq import Groq

def generate_rag_response(
    query: str,
    retrieved_movies: List[Dict],
    api_key: str,
    preferences: Dict
) -> Generator[str, None, None]:
    """
    Generates a conversational RAG response using the Groq API.
    Streams the response back as a generator.
    """
    if not api_key:
        yield "Error: Groq API Key is missing. Please provide it in the sidebar."
        return

    client = Groq(api_key=api_key)
    
    # Format the retrieved movies into a context string
    context_str = ""
    for i, movie in enumerate(retrieved_movies, 1):
        context_str += f"Movie {i}: {movie['title']} ({movie['release_year']})\n"
        context_str += f"Genre: {movie['genre']} | Rating: {movie['rating']}\n"
        context_str += f"Description: {movie['description']}\n\n"
        
    # Format the user's dashboard preferences to inform the LLM
    pref_str = f"- Genre Weights (0-10): Action({preferences.get('action_weight', 5)}), Comedy({preferences.get('comedy_weight', 5)}), Drama({preferences.get('drama_weight', 5)}), Sci-Fi({preferences.get('scifi_weight', 5)})\n"
    pref_str += f"- Dealbreakers: {', '.join(preferences.get('dealbreakers', [])) or 'None'}\n"
    pref_str += f"- Preferred Mood: {preferences.get('mood', 'Balanced')}\n"

    system_prompt = (
        "You are 'MovieSense', an expert, enthusiastic movie recommendation AI. "
        "Your job is to recommend movies to the user based on the Semantic Search Results provided to you. "
        "You must ONLY recommend movies that are in the Search Results.\n\n"
        f"Here are the user's explicit profile preferences from their dashboard:\n{pref_str}\n"
        "Explain WHY these movies fit their query and their profile preferences. Keep it conversational and concise."
    )

    user_prompt = f"User Query: {query}\n\nSemantic Search Results:\n{context_str}"

    try:
        stream = client.chat.completions.create(
            model="qwen/qwen3.6-27b", # Updated to a model available for your API key
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        
        in_think = False
        buffer = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
                
            buffer += content
            
            while True:
                if not in_think:
                    start_idx = buffer.find("<think>")
                    if start_idx != -1:
                        if start_idx > 0:
                            yield buffer[:start_idx]
                        buffer = buffer[start_idx + len("<think>"):]
                        in_think = True
                    else:
                        last_lt = buffer.rfind("<")
                        # If a '<' is near the end, hold it to see if it becomes '<think>'
                        if last_lt != -1 and len(buffer) - last_lt < 7:
                            if last_lt > 0:
                                yield buffer[:last_lt]
                            buffer = buffer[last_lt:]
                            break
                        else:
                            yield buffer
                            buffer = ""
                            break
                else:
                    end_idx = buffer.find("</think>")
                    if end_idx != -1:
                        buffer = buffer[end_idx + len("</think>"):]
                        in_think = False
                    else:
                        last_lt = buffer.rfind("<")
                        # If a '<' is near the end, hold it to see if it becomes '</think>'
                        if last_lt != -1 and len(buffer) - last_lt < 8:
                            buffer = buffer[last_lt:]
                        else:
                            buffer = ""
                        break
                        
        if not in_think and buffer:
            yield buffer
                
    except Exception as e:
        yield f"\n\n*Error generating response from Groq API: {str(e)}*"
