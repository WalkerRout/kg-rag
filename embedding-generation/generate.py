import os
import sys
import pathlib

from langchain.text_splitter import TokenTextSplitter

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WikipediaLoader, PyPDFDirectoryLoader, PyMuPDFLoader
from langchain_community.graphs.neo4j_graph import Neo4jGraph

from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_openai import ChatOpenAI

model_name = os.environ["MODEL_NAME"]
pdf_path = os.environ["PDF_PATH"]

def is_empty(directory):
  return not any((True for _ in os.scandir(directory)))

if is_empty(pdf_path):
  sys.exit(f"PDF path at {pdf_path} is empty")

# load the documents
print(f"Loading documents from {pdf_path}") # todo make this logger
files = list(map(lambda p: str(p), list(pathlib.Path(pdf_path).rglob("*.pdf"))))
print(f"Found documents: {files}")

# how nice it would be to just use:
# raw_documents = PyPDFDirectoryLoader(pdf_path, recursive=True).load()
# but for some reason 'Ignoring wrong pointing object' errors abound

raw_documents = []
for file in files:
  raw_documents.extend(PyMuPDFLoader(file).load())

# chunk documents
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents)

print("Documents split")

llm = ChatOpenAI(temperature=0, model_name=model_name)
prompt = ChatPromptTemplate.from_messages([
  ("system", """Your purpose is to construct a knowledge base centred around the AI usage and limitation 
                policies of other companies and organizations. Use the documents provided to form 
                relations such as 'Complies with', 'Utilizes', 'Guarantees/Ensures', 'Privatizes', 'Protects',
                'Enforces', 'Restricts', and other relations between the guarantees the documents mention. 
                You will focus specifically on the 'Code of Conduct' aspect of these policies. Make sure to 
                properly identify relations; if you are given a phrase such as 'CIBC intellectual property is 
                protected by law', you should have relations such as;
                (CIBC intellectual property) --- PROTECTED_BY ---> (law). Try to limit the number of 'Mentions'
                relations; consider if a more descriptive relation could be used to enhance later 
                retrieval augmented generation.
                """
  ),
])
llm_transformer = LLMGraphTransformer(llm=llm, prompt=prompt)

# add "Document" label to metadata
for doc in documents:
  if "metadata" not in doc:
    doc.metadata = {}
  doc.metadata["label"] = "Document"

print("Beginning graph construction")

# convert documents to graph documents
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# initialize the Neo4j connection
graph = Neo4jGraph()

print("Adding graph documents to database")

# add graph documents to Neo4j
graph.add_graph_documents(
  graph_documents,
  baseEntityLabel=True,
  include_source=True
)