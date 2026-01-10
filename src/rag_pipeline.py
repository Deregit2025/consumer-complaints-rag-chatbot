import os
from transformers import pipeline
from src.retriever import ComplaintRetriever

# Load HF token from environment (optional, if model is private)
HF_TOKEN = os.getenv("HF_TOKEN")

class RAGPipeline:
    def __init__(self, retriever_model_name="all-MiniLM-L6-v2", llm_model_name="mistralai/Mistral-7B-Instruct-v0.2", device=-1):
        # Initialize retriever with pre-built embeddings
        self.retriever = ComplaintRetriever(model_name=retriever_model_name)

        # Initialize generator pipeline
        self.generator = pipeline(
            "text-generation",
            model=llm_model_name,
            device=device  # -1 for CPU, 0 for GPU
            # use_auth_token removed because model should already be cached locally
        )

    def ask(self, question, top_k=5):
        """
        Given a user question:
        1. Retrieve top_k relevant complaint chunks.
        2. Construct a prompt using the assignment template.
        3. Generate the answer using the LLM.
        Returns both the answer and retrieved chunks.
        """
        # Step 1: Retrieve
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved_chunks:
            return "No relevant context found.", []

        # Step 2: Build context string including metadata
        context_list = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            # Use available metadata if exists
            text = chunk.get("text", "")
            company = chunk.get("company", "Unknown")
            issue = chunk.get("issue", "Unknown")
            date = chunk.get("date_received", "Unknown")
            context_list.append(f"Chunk {idx} | Company: {company} | Issue: {issue} | Date: {date}\n{text}")

        context = "\n\n".join(context_list)

        # Step 3: Prompt engineering
        prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. 
If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:
"""

        # Step 4: Generate response
        generated = self.generator(prompt, max_new_tokens=256)
        answer = generated[0]["generated_text"].split("Answer:")[-1].strip()

        # Step 5: Return both answer and retrieved chunks
        return answer, retrieved_chunks


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    rag = RAGPipeline(device=-1)  # CPU
    question = "Which companies have billing disputes?"
    response, chunks = rag.ask(question, top_k=5)

    print("Generated Response:")
    print(response)
    print("\nRetrieved Chunks:")
    for c in chunks:
        print(c)
