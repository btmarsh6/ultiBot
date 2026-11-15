Ulti Chatbot Overview

Goal:
Create a RAG chatbot that can answer questions about the rules of the sport Ultimate. This is designed to be used on the sideline to resolve disputes or clarify what should happen in specific in-game scenarios.

RAG Data:
The chatbot will be provided the official rulesets for both USA Ultimate (USAU) and the World Flying Disc Federation (WFDF). It will also have a slang glossary of common jargon to help it properly understand user input.

Design Requirements:
- Provide a concise answer to the specific question along with a citation for the section(s) of the ruleset that relates to the scenario.
- If the scenario or question from the user is ambiguous, ask follow up questions to clarify before providing an answer.
- Use Streamlit for the UI and hosting.
- Optimize layout for use on mobile devices. This would most commonly be used on cellphones.
- Toggle to switch between USAU ruleset and WFDF ruleset.
- Use open source or free models and databases to keep the cost of hosting and operating the chatbot to a minimum

Example queries/responses (using USAU rules): 

Q: What happens if the offense and defense catch the disc at the same time?
R: The offense retains posession. (3.I.6)


Q: What happens when the pull lands inbounds but rolls out of bounds?
R: Did anyone touch the disc before it rolled out of bounds?
Q: No
R: The disc is put in play at the nearest part of the central zone to where it rolled out of bounds, either on the sideline or goal line. (9.F.2)

