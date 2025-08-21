%%writefile interface.py

from model_loader import get_textgen_pipeline, generate_response
from chat_memory import add_to_history, build_context, chat_history

def start_chat():
    print("Chat started! Type '/exit' or '/q' to end.\n")
    # Load the model pipeline ONCE and reuse it
    pipe = get_textgen_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["/exit", "/q"]:
            print("Exiting chatbot. Goodbye!")
            break

        add_to_history("user", user_input)
        context = build_context(chat_history)
        bot_response = generate_response(pipe, context)
        add_to_history("bot", bot_response)
        print(f"Bot: {bot_response}\n")

if __name__ == "__main__":
    start_chat()
