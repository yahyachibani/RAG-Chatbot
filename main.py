from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os

from create_db_rag import OpenRouterEmbeddings, embed_text

load_dotenv(override=True)

client = OpenAI( base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"))




#### Partie réponse avec contexte RAG et raisonnement ####

def retrieve_similar_documents(vectordb, user_query, k=4):
    """
    Récupère les documents les plus similaires dans le vectordb pour la question de l'utilisateur.
    
    Paramètres:
    - vectordb : instance de la base vectorielle
    - user_query (str) : question de l'utilisateur
    - k (int) : nombre de documents à récupérer
    
    Retour:
    - list of tuples : chaque tuple contient (Document, score)
    """
    results = vectordb.similarity_search_with_score(
        query=user_query,
        k=k
    )
    
    # Affiche les scores pour debug
    for doc, score in results:
        print(f"Document ID: {doc.metadata.get('id', 'N/A')}, Score: {round(score,4)}")

    return results


def format_retrieved_context(docs, top_k=4):
    """
    Transforme les top_k documents récupérés en un texte unique pour l'LLM.
    
    Paramètres:
    - docs : liste de tuples (Document, score)
    - top_k : nombre de documents à inclure
    
    Retour:
    - str : texte formaté
    """
    # Trie par score descendant
    sorted_docs = sorted(docs, key=lambda d: d[1], reverse=True)[:top_k]

    blocks = []
    for doc, score in sorted_docs:
        blocks.append(
            f"---\nID: {doc.metadata.get('id','N/A')}\n"
            f"Title: {doc.metadata.get('title','N/A')}\n"
            f"Source: {doc.metadata.get('source','N/A')}\n"
            f"Score: {round(score,4)}\n"
            f"Text: {getattr(doc, 'page_content', '')[:]}...\n"
        )

    return "Retrieved documents (top {}):\n\n{}".format(top_k, "\n".join(blocks))


def load_vector_db():
    """Charge la base de données vectorielle Chroma existante."""
    if not os.path.exists("chroma_db"):
        raise FileNotFoundError("Le répertoire 'chroma_db' n'existe pas. Veuillez créer la base de données vectorielle d'abord.")

    embed_text = OpenRouterEmbeddings()
    # recharge le vectordb existant
    vectordb = Chroma(
        persist_directory="chroma_db",  
        embedding_function=embed_text    
    )
    return vectordb

def reponse_func(user_question, vectordb=load_vector_db(), client=client, k=4):
    """
    Génère une réponse à la question de l'utilisateur en utilisant le contexte RAG et le raisonnement.
    
    Paramètres:
    - user_question (str) : question de l'utilisateur
    - vectordb : base vectorielle pour récupérer les documents similaires
    - client : client OpenAI déjà instancié
    - k : nombre de documents à récupérer
    
    Retour:
    - str : réponse générée par le modèle
    """
    if vectordb is None or client is None:
        raise ValueError("vectordb et client doivent être fournis")

    # Récupère les documents similaires
    retrieved_docs = retrieve_similar_documents(vectordb, user_question, k=k)

    # Formate le contexte pour le LLM
    context_text = format_retrieved_context(retrieved_docs, top_k=k)
    print(context_text)  # debug

    # Prépare les messages pour le modèle
    system_prompt = (
        "Tu es un assistant professionnel qui répond aux questions en utilisant uniquement les informations du contexte fourni.\n"
        "Important :\n"
        "- Ne jamais dire des phrases génériques comme 'D’après les extraits de la candidature que vous avez partagée'.\n"
        "- Si la réponse n'est pas dans le contexte, réponds poliment et professionnellement : "
        "'D’après les informations disponibles, je ne peux pas répondre à cette question.' ou 'Je ne sais pas.'\n"
        "- Donne la réponse directement et clairement.\n"
        "Contexte RAG :\n\n"
        + context_text
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # Appel unique à l'API avec raisonnement activé
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=messages,
        extra_body={"reasoning": {"enabled": True}}
    )

    # Récupère la réponse de l'assistant
    assistant_message = response.choices[0].message
    return assistant_message.content


if __name__ == "__main__":
    # example usage
    question = "Quand est ce que Yahya est disponible pour commencer son stage ?"
    answer = reponse_func(question)
    print("Q:", question)
    print("A:", answer)
