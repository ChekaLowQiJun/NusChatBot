import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_huggingface.llms import HuggingFaceEndpoint
from getpass import getpass
from langchain.memory import ChatMessageHistory
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

'''
HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
'''

loader = PyPDFLoader('eviewbook.pdf')
data = loader.load()

chunk_size = 500
chunk_overlap = 100

ct_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

docs = ct_splitter.split_documents(data)

embedding_function = OpenAIEmbeddings(openai_api_key='sk-proj-fbLzrqsNlz5R2I7o78SAT3BlbkFJyUNTD38LCyWcZNsv1LVI')

vectordb = Chroma(
    persist_directory='data',
    embedding_function=embedding_function
)
vectordb.persist()

docstorage= Chroma.from_documents(docs, embedding_function)

retriever=docstorage.as_retriever()
llm = OpenAI(model_name='gpt-3.5-turbo-instruct', openai_api_key='sk-proj-fbLzrqsNlz5R2I7o78SAT3BlbkFJyUNTD38LCyWcZNsv1LVI')
#llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.3', huggingfacehub_api_token='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')

chat_history = ChatMessageHistory()

system_prompt = (
    'Imagine you are an officer at the office of admissions at NUS answering questions from a potential or current student of NUS.'
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know and mention that additional support can be provided by emailing the Admissions Office. "
    "Keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

chat_history = []

while True:

    query = input("You: ")
    chat_history.append(HumanMessage(content=query))

    if query.lower() == 'exit':
        break

    else:
        output = chain.invoke({"input": query, "chat_history": chat_history})['answer']
        chat_history.append(AIMessage(content=output))
        print(output)






