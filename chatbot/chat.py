# Importing necessary modules and libraries
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

# Loading environment variables from a .env file
load_dotenv()

# Initializing ChatOpenAI for conversational chat
chat = ChatOpenAI()   #verbose=True
    

# Initializing memory for conversation summary
memory = ConversationSummaryMemory(llm=chat,
                                #chat_memory=FileChatMessageHistory('messages.json'), 
                                memory_key = "messages",
                                return_messages = True)

# Defining the prompt template for the conversation
prompt = ChatPromptTemplate(
    input_variables = ["messages", 'content'],
    messages = [
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")  #tell me about why i need to {query}
    ]
)

# Initializing LLMChain for generating responses based on prompts
chain = LLMChain(
    llm=chat,
    prompt= prompt,
    memory=memory,
    # verbose=True  # To check all intermediate outputs
)

# Continuously prompt user for input and generate responses
while True:
    content = input(">> ")  # Taking user input
    result = chain({"content": content})  # Generating response based on input

    print(result["text"])  # Printing the generated response
