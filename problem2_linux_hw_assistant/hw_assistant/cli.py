"""Interactive CLI for the Local Linux Hardware Q&A Assistant."""

import sys
from hw_assistant.llm_router import HardwareAssistant


def main() -> None:
    assistant = HardwareAssistant()
    print("Asama.ai Local Linux Hardware Q&A Assistant (type 'exit' to quit)")
    avail = [k for k, v in assistant.available_tools.items() if v["available"]]
    print(f"Discovered system tools: {', '.join(avail) if avail else 'none'}")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        res = assistant.ask(query)
        print(f"\nAssistant: {res['answer']}")
        return

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query or query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        res = assistant.ask(query)
        print(f"Assistant: {res['answer']}")


if __name__ == "__main__":
    main()
