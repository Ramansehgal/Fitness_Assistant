import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class ExerciseKnowledgeIngestor:
    def __init__(
        self,
        embedding_model="text-embedding-3-small",
        persist_dir="data/vectorstore",
        rebuild = True
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.persist_dir = persist_dir

        if rebuild and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        if os.path.exists(persist_dir):
            self.vectorstore = FAISS.load_local(
                persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = None

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150
        )

    def ingest_pdf(
        self,
        pdf_path: str,
        exercise: str,
        joint: str | None = None,
        source: str = "expert_notes"
    ):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Attach metadata to each page
        for d in docs:
            d.metadata.update({
                "exercise": exercise,
                "joint": joint if joint else "general",
                "source": source,
                "file": os.path.basename(pdf_path)
            })

        chunks = self.splitter.split_documents(docs)
        
        for c in chunks:
            c.metadata["exercise"] = exercise
            c.metadata["joint"] = joint
            c.metadata["source_file"] = os.path.basename(pdf_path)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        os.makedirs(self.persist_dir, exist_ok=True)
        self.vectorstore.save_local(self.persist_dir)

        print(
            f"[RAG] Added {len(chunks)} chunks "
            f"(exercise={exercise}, joint={joint})"
        )

        print(f"[RAG INGEST] {len(chunks)} chunks added from {pdf_path}")
        return self.vectorstore
    
    def ingest_pdfs(self, pdf_paths: list[str]):
        all_docs = []

        for pdf in pdf_paths:
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            all_docs.extend(docs)

        chunks = self.splitter.split_documents(all_docs)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        os.makedirs(self.persist_dir, exist_ok=True)
        self.vectorstore.save_local(self.persist_dir)

        print(f"[KnowledgeIngestor] Added {len(chunks)} chunks from {len(pdf_paths)} PDFs")

        return self.vectorstore