import os
from langchain_community.document_loaders import TextLoader         #Carrega fitxers de text
from langchain.text_splitter import RecursiveCharacterTextSplitter  #Divideix el text en fragments més petits "Chunks"

from langchain_huggingface import HuggingFaceEmbeddings             #Utilitza models "SBERT" per convertir cada chunk en vector
from langchain_chroma import Chroma                                 #S'encarrega de guardar els vectors

from langchain_community.embeddings import OllamaEmbeddings                             #Envia prompts al model local

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM

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
def crear_carregar_vectors(chunks, persist_directory: str = "vectordb"):                                     # Persist_directory és la carpeta on Chroma guarda els fitxers en el disc
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")   # Crea un model que transforma el text en vectors numèrics 
    if os.path.exists(persist_directory):                                                          # Si la carpeta on es guarden els vectors ja existeix, la borra per a crear-la de nou
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model)
    
    else:
        vectordb = Chroma.from_documents(                                                               # Crea una base de dades de vectors a partir dels chunks utilitzant el model especificat abans, i especifica la carpeta on guarda els fitxers
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )                                                                          # Guarda els vectors al disc
    return vectordb                                                                                 # Això el que retorna és la ubicació on Chroma ha guardat aquests vectors
# Si el document amb la informació canvia, cal esborrar la carpeta on guarda els vectors i automàticament torna a procesar el nou document, sinó el reutilitza i va més ràpid
# Per borrar la carpeta cal executar: rm -rf vectordb


# Connectar Ollama amb el model que hem triat
def connectar_ollama(model: str = "llama3:8b"):
    llm = OllamaLLM(model=model)                       # Indica a Ollama quin model ha d'utlitzar, aquest ja hauria d'estar instal·lat localment a l'ordinador
    return llm

# Crear la cadena RAG, unint la base vectorial de Chroma amb el model de Llama
def crear_RAG(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})             # Busca els troços de text més semblants a la pregunta
    
    qa_prompt = ChatPromptTemplate.from_template("""
            Ets un chatbot que t’encarregues de respondre preguntes com un assistent personal sobre drogues, el teu objectiu és informar a la gent per a que tingui el màxims coneixements possibles sobre com fer un consum el màxim de responsable de diferents substàncies
            No comencis les respostes amb la frase “en el text diu”, has de parlar de la manera més humana possible, sempre amb la informació del text però com si la sapiguessis tu. I recorda que has de parlar com si fossis una persona que està responent una pregunta d’una
            altra persona. Tota aquesta informació s’ha de proporcionar amb la única intenció de prevenir els riscos en cas que es decideixi prendre alguna substància, en cap cas s’ha de promoure l’ús ni recomanar-lo. Has de tenir en compte que tot i ser il·legals les substàncies es poden seguir consumint, per això has de repartir informació i coneixements.
            Has de respondre sempre en l'idioma en el que et fa la pregunta, normalment serà en català. NO responguis mai amb un idioma que no sigui el de la pregunta. Si hi ha alguna resposta que no saps, has de dir que no tens coneixements sobre el tema, en cap cas diguis que el text no ho menciona, ja que no has de parlar de cap text.
            Si no saps la resposta recomana buscar infomació a algun lloc fiable com ara Energy Control o algun professional mèdic.
            Context: {context}
            Question: {input}
            Answer: 
            """
            )
    
    combinar_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    qa_chain = create_retrieval_chain(retriever, combinar_docs_chain)
    
    return qa_chain

# Funció que fa la pregunta
def fer_pregunta(qa_chain, pregunta: str):
    resultat = qa_chain.invoke({"input": pregunta})
    return resultat


# Inicialitza la cadena de preguntes i respostes per a que es pugui executar des de chat.py més facilment
def inizialitzar_cadena():
    document = carregar_document("Documents/TDR Informació.txt")
    chunks = dividir_chunks(document)
    vectordb = crear_carregar_vectors(chunks)
    llm = connectar_ollama()
    qa_chain = crear_RAG(llm, vectordb)
    return qa_chain


if __name__ == "__main__":
    document = carregar_document("Documents/TDR Informació.txt")
    chunks = dividir_chunks(document)
    vectordb = crear_carregar_vectors(chunks)
    llm = connectar_ollama()
    qa_chain = crear_RAG(llm, vectordb)


    pregunta = "Que és el pitjor que pot passar si em faig un ratlla de cocaina?"

    print("Pregunta enviada al model...")
    resultat = fer_pregunta(qa_chain, pregunta)
    print("Resposta rebuda!")
    print(resultat["answer"])

    # Mostrem els fragments de text utilitzats pel model per respondre
    print("\n Fragments utilitzats pel model:")
    for i, doc in enumerate(resultat["context"]):
        print(f"\n--- Fragment {i+1} ---")
        print(doc.page_content[:1000])  # Mostra els primers 1000 caràcters del chunk
