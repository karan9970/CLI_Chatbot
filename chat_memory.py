# Handles the sliding window for context memory
%%writefile chat_memory.py
MAX_MEMORY_TURNS = 4  # number of (user,bot) exchange pairs to keep
chat_history = []     # list of dicts: [{'role': 'user'/'bot', 'text': '...'}]

def build_context(history):
    context = ""
    for turn in history:
        role = "User" if turn['role'] == "user" else "Assistant"
        context += f"{role}: {turn['text']}\n"
    return context

def add_to_history(role, text):
    """
    Append a user/bot message and enforce a sliding window.
    """
    chat_history.append({"role": role, "text": text})

    # Maintain window for last N pairwise exchanges (user+bot = 2 entries per turn)
    max_len = MAX_MEMORY_TURNS * 2
    if len(chat_history) > max_len:
        # Remove oldest entries exceeding the memory window
        del chat_history[0:len(chat_history) - max_len]
