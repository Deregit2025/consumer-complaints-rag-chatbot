from src.rag_pipeline import RAGPipeline

# ----------------------------
# Evaluation Questions
# ----------------------------
evaluation_questions = [
    "Which companies have billing disputes?",
    "What complaints were reported against American Express?",
    "Which companies had issues with closing accounts?",
    "How many complaints involved Bank of America?",
    "What type of product is most commonly disputed?",
]

# ----------------------------
# Initialize RAG pipeline
# ----------------------------
rag = RAGPipeline(device=-1)  # CPU

results = []

# ----------------------------
# Helper function: safely extract text
# ----------------------------
def extract_generated_text(gen_output):
    """
    Extract text from the RAG pipeline output safely.
    Handles:
    - dict with 'generated_text'
    - list of dicts
    - tuple
    """
    # If list of dicts
    if isinstance(gen_output, list) and len(gen_output) > 0 and "generated_text" in gen_output[0]:
        return gen_output[0]["generated_text"].split("Answer:")[-1].strip()
    # If dict
    if isinstance(gen_output, dict) and "generated_text" in gen_output:
        return gen_output["generated_text"].split("Answer:")[-1].strip()
    # If tuple
    if isinstance(gen_output, tuple) and len(gen_output) > 0:
        first = gen_output[0]
        if isinstance(first, str):
            return first.split("Answer:")[-1].strip()
        elif isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"].split("Answer:")[-1].strip()
    # fallback
    return str(gen_output)

# ----------------------------
# Run Evaluation
# ----------------------------
for question in evaluation_questions:
    print("\n--- Evaluating Question ---")
    print(question)

    # Generate answer
    answer_raw = rag.ask(question, top_k=5)
    answer = extract_generated_text(answer_raw)

    # Retrieve chunks separately for inspection
    retrieved_chunks = rag.retriever.retrieve(question, top_k=2)

    # Prepare retrieved sources (1–2 chunks)
    sources = []
    for chunk in retrieved_chunks:
        company = chunk.get("company", "Unknown")
        issue = chunk.get("issue", "Unknown")
        text_preview = chunk.get("text", "")[:120].replace("\n", " ")
        sources.append(f"{company} | {issue} | {text_preview}...")

    results.append({
        "Question": question,
        "Generated Answer": answer,
        "Retrieved Sources": " || ".join(sources),
        "Quality Score": "TBD",
        "Comments": "To be analyzed"
    })

# ----------------------------
# Print Markdown Evaluation Table
# ----------------------------
print("\n# RAG Pipeline Evaluation Table\n")
print("| Question | Generated Answer | Retrieved Sources | Quality Score | Comments |")
print("|----------|------------------|------------------|---------------|----------|")

for r in results:
    q = r["Question"].replace("\n", " ")
    a = r["Generated Answer"].replace("\n", " ")
    s = r["Retrieved Sources"].replace("\n", " ")
    print(f"| {q} | {a} | {s} | {r['Quality Score']} | {r['Comments']} |")
