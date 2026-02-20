from rag_chatbot import RAGChatbot


def main() -> None:
    bot = RAGChatbot("data/knowledge_base.txt")
    bot.load_and_index()

    print("RAG Chatbot is ready. Type your question, or type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        response = bot.answer(user_input)
        print(f"\nBot: {response}")


if __name__ == "__main__":
    main()
