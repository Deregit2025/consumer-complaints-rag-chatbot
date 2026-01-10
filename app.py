import gradio as gr
from src.rag_pipeline import RAGPipeline

# Initialize RAG pipeline (load once)
rag = RAGPipeline(device=-1)  # CPU

def ask_question(question, top_k):
    if not question.strip():
        return "", "Please enter a question."

    answer, chunks = rag.ask(question, top_k=top_k)

    # Format retrieved sources for display
    if not chunks:
        sources_text = "No sources retrieved."
    else:
        formatted_sources = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            text = chunk.get("document", "")

            company = meta.get("company", "Unknown")
            issue = meta.get("issue", "Unknown")
            date = meta.get("date_received", "Unknown")

            formatted_sources.append(
                f"Source {i}\n"
                f"Company: {company}\n"
                f"Issue: {issue}\n"
                f"Date: {date}\n"
                f"Excerpt: {text}\n"
            )

        sources_text = "\n" + "\n".join(formatted_sources)

    return answer, sources_text


def clear_fields():
    return "", "", ""


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("## 💬 CrediTrust RAG-Based Complaint Assistant")
    gr.Markdown(
        "Ask questions about customer complaints. "
        "The system retrieves relevant complaint excerpts and generates answers with sources."
    )

    question_input = gr.Textbox(
        label="Enter your question",
        placeholder="e.g. Which companies have billing disputes?",
        lines=2
    )

    top_k_slider = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        label="Number of retrieved sources (Top-K)"
    )

    ask_button = gr.Button("Ask")
    clear_button = gr.Button("Clear")

    answer_output = gr.Textbox(
        label="AI Generated Answer",
        lines=5
    )

    sources_output = gr.Textbox(
        label="Retrieved Sources",
        lines=15
    )

    ask_button.click(
        fn=ask_question,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, sources_output]
    )

    clear_button.click(
        fn=clear_fields,
        outputs=[question_input, answer_output, sources_output]
    )

demo.launch()
