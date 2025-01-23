import ollama
from crewai_tools import PDFSearchTool
import os

def get_pdf_summary_or_answer(pdf_path, prompt:str):
    try:
        pdf_search_tool = PDFSearchTool(
            pdf=pdf_path,
            config=dict(
                llm=dict(
                    provider="ollama",
                    config=dict(
                        model="llama3.2:3b",
                        ),
                    ),
                embedder=dict(
                    provider="ollama",
                    config=dict(
                        model="llama3.2:3b",
                    ),
                )
            )
        )

        result = pdf_search_tool.run(query=prompt)
        return result

    except Exception as e:
        print(f"Error during PDF read: {e}")
        return None


while True:
    # Call the Ollama model to generate an answer
    try:
        pdf_path = input("Enter the path to the PDF or Ctrl+C: ")
        if not os.path.isfile(pdf_path):
            print("Invalid PDF file path")
            continue

        prompt = input("Enter your question: ")

        result = get_pdf_summary_or_answer(pdf_path, prompt)

        if result:
            print("Summary: ", result)
        else:
            print("No result or error")

        # Use the correct model name from the list (e.g., llama3.2:3b or llama3.2:latest)

        prompt = input("Enter your question: ")
        output = ollama.generate(
            model="llama3.2:3b",  # Replace this with the correct model name
            prompt=f"answer the question: {prompt}"
        )

        # Check the response structure and print the output
        if "response" in output:
            print("Response: ",output["response"])
        else:
            print("Error: No 'response' key in the output:", output)

    except KeyboardInterrupt:
        print("\n Exiting the program")
        break

    except Exception as e:
        print("An error occurred:", e)
