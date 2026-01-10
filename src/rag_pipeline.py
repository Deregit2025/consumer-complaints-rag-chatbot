import os
import threading
from transformers import pipeline, TextIteratorStreamer
from src.retriever import ComplaintRetriever

HF_TOKEN = os.getenv("HF_TOKEN")


class RAGPipeline:
    def __init__(
        self,
        retriever_model_name="all-MiniLM-L6-v2",
        llm_model_name="mistralai/Mistral-7B-Instruct-v0.2",
        device=-1,
    ):
        self.retriever = ComplaintRetriever(model_name=retriever_model_name)

        # Generator pipeline
        self.generator = pipeline(
            "text-generation",
            model=llm_model_name,
            device=device,
        )

    def ask(self, question, top_k=5):
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved_chunks:
            return "No relevant context found.", []

        context_list = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")
            company = chunk.get("company", "Unknown")
            issue = chunk.get("issue", "Unknown")
            date = chunk.get("date_received", "Unknown")
            context_list.append(
                f"Chunk {idx} | Company: {company} | Issue: {issue} | Date: {date}\n{text}"
            )

        context = "\n\n".join(context_list)

        prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. 
If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:
"""

        generated = self.generator(prompt, max_new_tokens=128)
        answer = generated[0]["generated_text"].split("Answer:")[-1].strip()

        return answer, retrieved_chunks

    def ask_stream(self, question, top_k=5, max_new_tokens=128):
        """
        Streaming version for Gradio:
        Yields the answer token-by-token.
        """
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved_chunks:
            yield "No relevant context found.", []
            return

        context_list = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")
            company = chunk.get("company", "Unknown")
            issue = chunk.get("issue", "Unknown")
            date = chunk.get("date_received", "Unknown")
            context_list.append(
                f"Chunk {idx} | Company: {company} | Issue: {issue} | Date: {date}\n{text}"
            )

        context = "\n\n".join(context_list)

        prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. 
If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:
"""

        # Set up streamer with tokenizer
        streamer = TextIteratorStreamer(
            self.generator.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        # Encode prompt
        inputs = self.generator.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.generator.device)

        # Generate in a thread
        thread = threading.Thread(
            target=self.generator.model.generate,
            kwargs={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "streamer": streamer,
            },
        )
        thread.start()

        # Stream token-by-token
        answer_text = ""
        for new_text in streamer:
            answer_text += new_text
            yield answer_text, retrieved_chunks
