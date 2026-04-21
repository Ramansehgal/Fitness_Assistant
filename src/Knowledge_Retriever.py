from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ExerciseKnowledgeRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer ONLY from the provided context. "
             "If information is missing, say you don't know."),
            ("human",
             "Exercise: {exercise}\n"
             "Question: {question}\n\n"
             "Context:\n{context}")
        ])

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = StrOutputParser()

    def retrieve(self, exercise: str, question: str):
        docs = self.retriever.invoke(f"{exercise}: {question}")
        context = "\n\n".join(d.page_content for d in docs)

        chain = self.prompt | self.llm | self.parser
        return chain.invoke({
            "exercise": exercise,
            "question": question,
            "context": context
        })

    def retrieve2(
        self,
        exercise: str,
        question: str,
        joint: str | None = None,
        k: int = 4
    ):
        filter_meta = {"exercise": exercise}
        if joint:
            filter_meta["joint"] = joint

        docs = self.vectorstore.similarity_search(
            question,
            k=k,
            filter=filter_meta
        )
        print(f"DOCS\n---------------\n{docs}")
        print("------------------------------")
        return "\n\n".join(d.page_content for d in docs)
    
    def retrieve_with_debug(
        self,
        exercise,
        question,
        joint=None,
        k=4
    ):
        print("\n[RAG QUERY]")
        print("Exercise :", exercise)
        print("Joint    :", joint)
        print("Question :", question)

        filter_meta = {"exercise": exercise}
        if joint:
            filter_meta["joint"] = joint

        results = self.vectorstore.similarity_search_with_score(
            question,
            k=k,
            filter=filter_meta
        )

        if len(results) == 0:
            print("[WARN] No chunks with metadata filter, retrying without joint filter")
            filter_meta.pop("joint", None)

            results = self.vectorstore.similarity_search_with_score(
                question,
                k=k,
                filter=filter_meta
            )

        if not results:
            print("[ERROR] No documents retrieved. LLM will hallucinate.")

        print("\n[TOP-K RETRIEVED CHUNKS]")
        contexts = []

        for i, (doc, score) in enumerate(results):
            print(f"\n--- Chunk {i+1} ---")
            print("Score    :", round(score, 4))
            print("Metadata :", doc.metadata)
            print("Content  :")
            print(doc.page_content[:500], "...\n")

            contexts.append(doc.page_content)
        if len(results) > 0:
            avg_score = sum(score for _, score in results) / len(results)
            print(f"Average Retrived Score:{avg_score}")

        return "\n\n".join(contexts)
    
    def retrieve_with_debug2(self, exercise, question, joint=None, k=6):
        print("\n[RAG QUERY]")
        print("Exercise :", exercise)
        print("Joint    :", joint)
        print("Question :", question)

        # Step 1: retrieve WITHOUT filters
        results = self.vectorstore.similarity_search_with_score(
            question,
            k=k
        )

        # print("\n[RAW RETRIEVAL RESULTS]")
        filtered = []

        for doc, score in results:
            meta = doc.metadata
            # print(f"Score={round(score,4)} | Meta={meta}")

            if meta.get("exercise") == exercise:
                if joint is None or meta.get("joint") in [joint, "general"]:
                    filtered.append((doc, score))

        if not filtered:
            print("[ERROR] No docs match exercise after manual filtering")
            return ""

        print("\n[FINAL SELECTED CHUNKS]")
        contexts = []
        for i, (doc, score) in enumerate(filtered):
            print(f"\n--- Chunk {i+1} ---")
            print("Score:", round(score,4))
            print("Content:", doc.page_content[:400], "...\n")
            contexts.append(doc.page_content)

        return "\n\n".join(contexts)
