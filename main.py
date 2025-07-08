from langchain_community.document_loaders import TextLoader         #Carrega fitxers de text
from langchain.text_splitter import RecursiveCharacterTextSplitter  #Divideix el text en fragments més petits "Chunks"

from langchain_huggingface import HuggingFaceEmbeddings             #Utilitza models "SBERT" per convertir cada chunk en vector
from langchain_community.vectorstores import Chroma                 #S'encarrega de guardar els vectors

from langchain_ollama import OllamaLLM                             #Envia prompts al model local

from langchain.chains import RetrievalQA                            #Uneix el sistema de recuperació "Chroma" amb "Ollama" per poder fer preguntes

# Carregar l'arxiu de text:
def carregar_document(ruta_txt: str):                           #La part de str indica a VisualStudio que sera un "string" és a dir, text, per a que així pugui detectar els errors més facilment
    loader = TextLoader(ruta_txt, encoding='utf-8')             #Crea una variable que sap llegir l'arxiu, la part d'encodig fa que entengui més tipus de caràcters
    documents = loader.load()                                   #Crea una llista per cada document que detecta
    return documents

# Dividir el text en fragments més petits "chunks":
def dividir_chunks(documents, chunk_size: int = 1500, chunk_overlap: int = 500):         # La superposicio indica que el seguent fragment agafi els últims 200 caràcters per tal de no tallar frases per la meitat
    splitter = RecursiveCharacterTextSplitter(                                           # Separa el text segons el que haguem definit en els paràmetres
        chunk_size = chunk_size,                                                         # Defineix el paràmetre de la funció com un objecte que té la llibreria
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_documents(documents)                                         # Indiquem quin text ha de separar i utilitza una funció per mantenir millor el context
    return chunks


# Convertir els chunks en vectors 
def crear_vectors(chunks, persist_directory: str = "vectordb"):                                     # Persist_directory és la carpeta on Chroma guarda els fitxers en el disc
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")   # Crea un model que transforma el text en vectors numèrics 
    vectordb = Chroma.from_documents(                                                               # Crea una base de dades de vectors a partir dels chunks utilitzant el model especificat abans, i especifica la carpeta on guarda els fitxers
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vectordb                                                                                 # Això el que retorna és la ubicació on Chroma ha guardat aquests vectors


# Connectar Ollama amb el model que hem triat
def connectar_ollama(model: str = "llama3:8b"):
    llm = OllamaLLM(model=model)                       # Indica a Ollama quin model ha d'utlitzar, aquest ja hauria d'estar instal·lat localment a l'ordinador
    return llm

# Crear la cadena RAG, unint la base vectorial de Chroma amb el model de Llama
def crear_RAG(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})             # Busca els troços de text més semblants a la pregunta
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Funció que fa la pregunta
def fer_pregunta(qa_chain, pregunta: str):
    resultat = qa_chain.invoke({"query": pregunta})
    return resultat

document = carregar_document("Documents/TDR Informació (1).txt")

chunks = dividir_chunks(document)

vectordb = crear_vectors(chunks)

llm = connectar_ollama()

qa_chain = crear_RAG(llm, vectordb)

pregunta = "Quin és el teu propòsit com a intel·ligència artificial? Sobre quines drogues tens informació i sobre quines no?"

print("🟡 Pregunta enviada al model...")
resultat = fer_pregunta(qa_chain, pregunta)
print("✅ Resposta rebuda!")
print(resultat["result"])

# Mostrem els fragments de text utilitzats pel model per respondre
print("\n Fragments utilitzats pel model:")
for i, doc in enumerate(resultat["source_documents"]):
    print(f"\n--- Fragment {i+1} ---")
    print(doc.page_content[:1000])  # Mostra els primers 1000 caràcters del chunk
