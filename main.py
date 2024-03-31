import os
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


def setOpenAPIKey():
    # Set OpenAI API Key
    os.environ["OPENAI_API_KEY"] = ""


def connectToDB():
    # Create SQLDatabase instance

    db = SQLDatabase.from_uri("sqlite:///Chinook.db")

    print(db.dialect)
    print(db.get_usable_table_names())
    #db.run("SELECT * FROM Artist LIMIT 10;")
    return db


# Streamlit UI
def run_streamlit_interface(agent_executor):
    st.title("SQL Query Interface")
    user_query = st.text_input("Enter your SQL query")

    if st.button("Submit"):
        response = agent_executor.invoke({"input": user_query})
        st.write("Response:")
        st.write(response)


# Main function
def main():
    # SQLDatabaseToolkit initialization
    setOpenAPIKey()

    db = connectToDB()

    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
    context = toolkit.get_context()
    tools = toolkit.get_tools()

    # Chat prompt creation
    messages = [
        HumanMessagePromptTemplate.from_template("{input}"),
        AIMessage(content="{SQL_FUNCTIONS_SUFFIX}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    prompt = prompt.partial(**context)

    # Agent initialization
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_openai_tools_agent(llm, tools, prompt)

    # AgentExecutor initialization
    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=True,
    )

    # Run Streamlit UI
    run_streamlit_interface(agent_executor)


if __name__ == "__main__":
    main()
