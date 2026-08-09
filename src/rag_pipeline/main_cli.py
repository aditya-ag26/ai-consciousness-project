"""
Command-Line Interface (CLI) for the RAG chatbot.

This script provides the user-facing entry point to interact with the QueryBot.
"""
import logging
import time

from src.config import config
from src.rag_pipeline.bot import QueryBot, format_response


def setup_logging():
    """Configures the root logger for the application."""
    log_path = config.rag_application.log_path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def main():
    """Initializes and launches the RAG chatbot CLI."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Initializing QueryBot...")
    bot = QueryBot(config.rag_application)
    logger.info("QueryBot initialized successfully.")
    
    print("\n--- AI Consciousness Chatbot ---")
    # Get user preferences
    length_map = config.rag_application.answer_length_map
    while True:
        lengths = ", ".join(length_map.keys())
        choice = input(f"Choose answer length ({lengths}): ").lower().strip()
        if choice in length_map:
            break
        print(f"Invalid choice. Please select from: {lengths}")

    num_tokens = length_map[choice]
    logger.info(f"User set answer length to '{choice}' ({num_tokens} tokens).")
    
    print("\nAsk your questions (Type 'exit' to quit):\n")
    history: list[tuple[str, str]] = []
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        start_time = time.time()
        result = bot.ask(query, num_predict_tokens=num_tokens, chat_history=history)
        end_time = time.time()

        logger.info(f"Query processed in {end_time - start_time:.2f} seconds.")

        history.append(("user", query))
        history.append(("assistant", result["result"]))

        formatted_output = format_response(result)
        print(formatted_output)
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()