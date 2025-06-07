import streamlit as st 
from bot import Chatbot

mybot = Chatbot()
workflow = mybot()

st.title("Chatbot with LangGraph")
st.write("Ask any question, and I'll try to answer it!")

question = st.text_input("Enter question here: ")
input = {"messages": [question]} 

if st.button("Get Answer"):
    if input:
        response = workflow.invoke(input)
        st.write("**Answer**", response['messages'][-1].content)
    else:
        st.write("Please enter a question to get answer")
        
st.markdown("------")
st.caption("Powered by Streamlit and Transformers")