from backend.rag.retriever import retrieve

def search(query: str):

    results = retrieve(
        query=query,
        top_k=5,
        use_reranker=True
    )

    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for idx, hit in enumerate(results, start=1):

        print(f"\nRESULT #{idx}")
        print("-" * 100)

        if "rerank_score" in hit:
            print("Rerank Score:", round(hit["rerank_score"], 4))

        print("Metadata:", hit["metadata"])

        print("\nContent:")
        print(hit["text"][:1200])


if __name__ == "__main__":

    print("\nKaif Knowledge Base Retrieval Tester")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        search(query)