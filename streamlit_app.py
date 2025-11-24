# app.py (Code Propre Révisé)
import streamlit as st
from main import reponse_func
from openai import OpenAI
import time

api_key = st.secrets["OPENAI_API_KEY"]
embeddings_key=st.secrets["EMBEDDINGS_API_KEY"]

st.set_page_config(page_title="Yahya CV-GPT Assistant", layout="wide")

st.title("👨‍💼 Yahya CV-GPT Assistant ")
st.markdown("Pose une question sur le CV et la lettre de motivation de **Yahya**, et l'assistant va répondre.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# --- Section pour les Suggestions de Questions ---
# Utilisation de st.expander pour un affichage discret
with st.expander("Suggestions de Questions à Poser :"):
    st.markdown("""
    * **Quelle est ton adresse e-mail professionnelle ?**
    * **Dans quelle école ou université as-tu étudié ?**
    * **A partir de quand Yahya peut commencer son stage ?**
    * *Quelles sont tes compétences techniques principales ?*
    """)

# --- Affichage de l'historique ---
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

with st.sidebar:
    st.header("Outils et Paramètres 🛠️")
    if st.button("Supprimer la conversation", type="primary"):
        st.session_state.conversation.clear()
        st.rerun()

user_input = st.chat_input("Pose ta question ici...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.conversation.append({"role": "user", "content": user_input})

    with st.spinner("Je réfléchis... "):
        time.sleep(1) 
        answer = reponse_func(user_input,api_key)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.conversation.append({"role": "assistant", "content": answer})