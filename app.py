import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

st.set_page_config(page_title="ZebPay Chatbot v2", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("ZiVa💬🦙")
st.info("Your Guide to cryto-trading", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about cryptocurrencyt!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_files=["1.docx", "2.docx", "3.docx", "chat_history.docx"], recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", temperature=0.1))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
@st.cache_resource(show_spinner=False)
def build_agent():
    agent = ReActAgent.from_tools([search_tool, escalate_tool, resource_tool], llm=OpenAI(model="gpt-4"), verbose=True)
    return agent

def escalate() -> None:
    """Recognizes a high-risk statement from the mental health chatbot and escalates to the next level of management. High-risk is defined as a statement that suggests that the client has a plan, means, and intent to harm oneself or others (specific details on when, where, and how)."""
    st.error("High risk detected. Please consider escalating immediately.", icon="🚨")

def get_resource_for_response(user_input) -> str:
    """Recognizes a no, low- or medium-risk statement from the mental health chatbot, seeks resources to inform potential chat responses"""
    response = st.session_state.query_engine.retrieve(user_input)
    resources = [t.node.metadata["file_name"] for t in response]
    content = [t.node.text for t in response]
    result = dict(zip(resources, content))
    return result

def search_for_therapists(locality: str = "Houston, Texas") -> str:
    """Use the Google Search Tool but only specifically to find therapists in the client's area, then send email to update the client with the results."""
    google_spec = GoogleSearchToolSpec(key=st.secrets.google_search_api_key, engine=st.secrets.google_search_engine)
    tools = LoadAndSearchToolSpec.from_defaults(google_spec.to_tool_list()[0],).to_tool_list()
    agent = OpenAIAgent.from_tools(tools, verbose=True)
    response = agent.chat(f"what are the names of three individual therapists in {locality}?")
    message = emails.html(
        html=f"<p>Hi Riley.<br>{response}</p>",
        subject="Helpful resources from TrevorChat",
        mail_from=('TrevorChat Counselor', 'contact@mychesscamp.com')
    )
    smtp_options = {
        "host": "smtp.gmail.com", 
        "port": 587,
        "user": "example@example.com", # To replace
        "password": "mypassword", # To replace   
        "tls": True
    }
    response = message.send(to='contact.email@gmail.com', smtp=smtp_options) # To replace with client's email
    return f"Message sent: {response.success}"

search_tool = FunctionTool.from_defaults(fn=search_for_therapists)
escalate_tool = FunctionTool.from_defaults(fn=escalate)
resource_tool = FunctionTool.from_defaults(fn=get_resource_for_response)
index = load_data()
agent = build_agent()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
         memory = ChatMemoryBuffer.from_defaults(token_limit=15000)
         st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", memory=memory, system_prompt=(
                  "You are an assistant who is an expert in cryptocurrency and ZebPay services. "
                  "Keep your answers technical and based on facts. Do not hallucinate features. "
                  "If the user's query is not related to cryptocurrency or ZebPay, simply say 'Please ask something related to crypto or ZebPay!'."
                  "Else If the user's query is related but you cannot find an answer, simply say 'Connecting you with customer support'."
                  "Else If the user seems annoyed or explicitly asks for customer support, simply say 'Connecting you with customer support'."
                  "Else If the user does not seem to understand your responses after 2-3 attempts, simply say 'Connecting you with customer support'."
        ), verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
