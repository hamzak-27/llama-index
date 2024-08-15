import os 
os.environ['OPENAI_KEY'] = ''
os.environ['ACTIVELOOP_TOKEN'] = ''

from llama_index.readers.wikipedia import WikipediaReader

reader = WikipediaReader()

documents = reader.load_data(pages=['Natural Language Processing', 'Artificial Intelligence'])
print(len(documents))

#pip install llama-index-readers-wikipedia

from llama_index.core.node_parser import SimpleNodeParser 

parser = SimpleNodeParser.from_defaults(chunk_size = 512, chunk_overlap=20)

nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

from llama_index.vector_stores.deeplake import DeepLakeVectorStore 
dataset_path = 'hub://ihamzakhan89/LlamaIndex_intro'
vector_store = DeepLakeVectorStore(dataset_path = dataset_path, overwrite=False)

from llama_index.core import StorageContext 
from llama_index.core import VectorStoreIndex 

storage_context = StorageContext.from_defaults(vector_store = vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context)

from llama_index import GPTVectoreStoreIndex 
index = GPTVectoreStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query('What does NLP stand for?')
print(response.response)