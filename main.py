from groq_client import GroqClient


def main():

    print("=" * 45)
    print("          LLM CLIENT - GROQ")
    print("=" * 45)

    try:

        llm = GroqClient()

        print("Groq connection initialized.")
        print("Type 'exit' to quit.\n")

        while True:

            prompt = input("You: ").strip()

            if prompt.lower() == "exit":
                print("Goodbye!")
                break

            if not prompt:
                print("Please enter a question.")
                continue

            try:

                print("\nGenerating response...\n")

                answer = llm.generate(prompt)

                print("Groq:")
                print(answer)
                print("\n" + "-" * 45 + "\n")

            except Exception as error:

                print(f"Request failed: {error}\n")

    except Exception as error:

        print(f"Initialization failed: {error}")


if __name__ == "__main__":
    main()