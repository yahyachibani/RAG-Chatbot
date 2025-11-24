# Projet RAG Chatbot

## Auteur

**Yahya Chibani**

---

## Description du projet

Ce projet est un **chatbot** basé sur l'architecture **Retrieval-Augmented Generation (RAG)**.

Le chatbot est conçu pour fournir des réponses précises en se basant sur une **base de connaissances locale** spécifique.

### Pile Technologique RAG

* **Vector Store :** **ChromaDB** est utilisé pour créer et gérer la base de données de vecteurs locale et persistante.
* **Pipeline d'Orchestration :** La librairie **Langchain** est employée pour le *chunking* (découpage) et le *parsing* des documents sources.
* **Modèles de Langage et d'Embeddings :**
    * **LLM (Large Language Model) :** Utilisation de l'API pour le modèle **`openai/gpt-oss-20b`** pour la génération de texte.
    * **Embeddings Model :** Le modèle **`sentence-transformers/all-minilm-l12-v2`** est utilisé pour transformer le texte en représentations vectorielles.

### Interface Utilisateur

L'interface utilisateur est réalisée avec **Streamlit**, offrant un moyen simple et visuel d'interagir avec le système RAG.
![Alt Text](Screenshots/Fonctionnement_du_Chatbot.png) 

Démonstration en Ligne : Vous pouvez tester l'application déployée ![ici](<**https://yahya-chatbot-cv.streamlit.app/**>).

---

## Fonctionnalités Principales

* **Gestion de Base de Connaissances :** Création et maintenance d'une base de connaissances locale (ChromaDB).
* **Réponses Contextuelles :** Génération de réponses précises et factuelles, extraites de la base de connaissances.
* **Interface Web :** Interface interactive et conviviale développée avec Streamlit.

---

## Installation

Suivez ces étapes pour configurer le projet localement.

### 1. Clonage du Dépôt

```bash
git clone https://github.com/yahyachibani/RAG-Chatbot.git
