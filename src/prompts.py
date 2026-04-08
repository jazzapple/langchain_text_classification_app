from langchain_core.prompts import ChatPromptTemplate

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an experienced newspaper editor who has experience working with content across news categories. "
        "Classify the article into exactly one of: World, Sports, Business, Sci/Tech. "
        "Respond with the category and a brief one-sentence reason."
    )),
    ("human", "{text}"),
])
