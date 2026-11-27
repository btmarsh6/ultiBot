# Ulti Rules Chatbot

## Purpose
This tool is designed to be a quick resource to help answer questions about the rules of Ultimate. In my years playing Ultimate,
there have been many situations where there was a disagreement about a call or how to resolve a situation. Not all players are 
knowledgeable about the rules or most recent changes. It can be slow and inconvenient to pull up the full ruleset on a phone while
on the sideline and try to find the relevant section. This chatbot can take questions in natural language and quickly return a concise 
answer along with a citation to the relevant section.

## Design
### Data
The data behind this RAG chatbot comes from 3 sources: The most recent version of the USAU rules, the WFDF rules, as well as a custom 
glossary of some common Ultimate jargon (Slang-Glossary.csv).

### Ingestor
The rulesets are both chunked by section to maintain relevant context. Subsections are combined up to 1500 characters. Each chunk also has metadata 
containing the ruleset, sections included, section heading (topic) and page number. This metadata aids with proper retrieval and citation.<br>
Chunks are embedded using the BAAI/bge-base-en-v1.5 embedding model, then stored in a Pinecone vector store.

### Chatbot
The chatbot is deployed using Streamlit. The query is first scanned to see if it includes any of the slang terms included in the glossary. 
If any slang is detected, it appends the definition of those terms to the query. Next it is embedded using the same embedding model from the ingestor. 
The LLM for the chatbot responses uses Claude Haiku 4.5.
