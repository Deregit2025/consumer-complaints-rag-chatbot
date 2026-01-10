import gradio as gr
from src.rag_pipeline import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline(device=-1)  # CPU, change to 0 if GPU is available

def ask_question(user_question):
    """
    Generator function for Gradio to stream answers token by token.
    Returns answer as it is generated, along with retrieved sources.
    """
    for answer, chunks in rag.ask_stream(user_question, top_k=5):
        # Build a simple sources string
        sources = "\n\n".join([f"Chunk {i+1} | Company: {c.get('company','Unknown')} | Issue: {c.get('issue','Unknown')} | Date: {c.get('date_received','Unknown')}\n{c.get('text','')}" 
                               for i, c in enumerate(chunks)])
        yield answer, sources

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## CrediTrust Complaint Analyzer (RAG Chatbot)")

    with gr.Row():
        txt_input = gr.Textbox(label="Ask a question about complaints:", placeholder="Type your question here...")
        btn = gr.Button("Ask")

    answer_output = gr.Textbox(label="AI Answer", interactive=False)
    sources_output = gr.Textbox(label="Retrieved Sources", interactive=False)

    btn.click(ask_question, inputs=txt_input, outputs=[answer_output, sources_output])

    gr.Button("Clear").click(lambda: ("", ""), inputs=[], outputs=[txt_input, answer_output, sources_output])

# Launch app
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
