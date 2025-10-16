import os
import json
from langchain_community.document_loaders import TextLoader         #Carrega fitxers de text
from langchain.text_splitter import RecursiveCharacterTextSplitter  #Divideix el text en fragments més petits "Chunks"

from langchain_huggingface import HuggingFaceEmbeddings             #Utilitza models "SBERT" per convertir cada chunk en vector
from langchain_chroma import Chroma                                 #S'encarrega de guardar els vectors

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaLLM

# Carregar l'arxiu de text:
def carregar_document(ruta_txt: str):                           #La part de str indica a VisualStudio que sera un "string" és a dir, text, per a que així pugui detectar els errors més facilment
    if ruta_txt.endswith('.json'):                             #Si el fitxer és un json, utilitza aquesta part del codi
        with open(ruta_txt, 'r', encoding='utf-8') as f:                                     #Retorna la llista de documents
            dades = json.load(f)                                                                 #Carrega el fitxer json
        substancies = dades.get("substancies", {})
        metadades = dades.get("metadades", {})
        documents = []
        for id_s, info in substancies.items():
            contingut = format_mes_compacte(id_s, info, metadades)
            documents.append(Document(page_content=contingut))
        return documents
    
    loader = TextLoader(ruta_txt, encoding='utf-8')             #Crea una variable que sap llegir l'arxiu, la part d'encodig fa que entengui més tipus de caràcters
    return loader.load()                                   #Carrega el fitxer de text

def format_mes_compacte(id_s, info, metadades):
    try:
        nom = info.get("nom_cientific", "No especificat")
        
        # Efectos positivos
        efectos_pos = info.get("efectes_desitjats_curt", [])
        if isinstance(efectos_pos, list) and efectos_pos:
            efectos_pos_str = ", ".join(efectos_pos[:3])
        else:
            efectos_pos_str = "efectes variables"
        
        # Efectos negativos  
        efectos_neg = info.get("efectes_no_desitjats_curt", [])
        if isinstance(efectos_neg, list) and efectos_neg:
            efectos_neg_str = ", ".join(efectos_neg[:3])
        else:
            efectos_neg_str = "efectes variables"
        
        dosi = info.get("dosi", "No especificada")
        durada = info.get("durada_efectes", "No especificada")
        
        # Texto natural
        texto = f"{id_s.capitalize()} ({nom}): Produeix {efectos_pos_str}. "
        texto += f"Efectes negatius: {efectos_neg_str}. "
        texto += f"Dosi: {dosi}. Durada: {durada}."
        
        return texto
        
    except Exception as e:
        # Fallback seguro
        return f"{id_s}: Informació disponible sobre aquesta substància."



# Dividir el text en fragments més petits "chunks":
def dividir_chunks(documents, chunk_size: int = 1500, chunk_overlap: int = 200):         # La superposicio indica que el seguent fragment agafi els últims 200 caràcters per tal de no tallar frases per la meitat
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

def carregar_vectordb():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(persist_directory="vectordb", embedding_function=embedding_model)
    return vectordb

def inicialitzar_chat():
    vectordb = carregar_vectordb()
    llm = connectar_llamacpp()
    qa_chain = crear_RAG(llm, vectordb)
    #Preescalfament del model
    print("Preescalfant el model...")
    try:
        fer_pregunta(qa_chain, "Hola")
        print("Model preescalfat correctament!")
    except Exception as e:
        print(f"S'ha produït un error durant el preescalfament: {e}")
    
    return qa_chain

# Connectar Ollama amb el model que hem triat
def connectar_llamacpp(model_name: str = "mistral:7b"):
    llm = OllamaLLM(
        model=model_name, 
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.1)                                               # Indica a Ollama quin model ha d'utlitzar, aquest ja hauria d'estar instal·lat localment a l'ordinador
    return llm

# Crear la cadena RAG, unint la base vectorial de Chroma amb el model de Llama
def crear_RAG(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})             # Busca els troços de text més semblants a la pregunta, en aquest cas n'hem definit 2
    
    qa_prompt = ChatPromptTemplate.from_template("""
            Ets un expert en drogues. Respon en català amb la informació del context.

            Context: {context}
            Pregunta: {input}

            Instruccions:
            - Respon de manera clara i concisa.
            - Al final de la resposta, afegeix: "Recorda que la manera més segura d'evitar riscos és no consumir. Si decideixes consumir, informa't sempre i analitza la substància amb Energy Control per coneixer la seva composició real."

            Resposta:
            """
            )
    
    combinar_docs_chain = create_stuff_documents_chain(llm, qa_prompt)      # Crea una cadena ajuntant tots els troços de text rellevants i el prompt inicial per a enviar-ho al model
    
    qa_chain = create_retrieval_chain(retriever, combinar_docs_chain)       # Crea la cadena final que utilitza el "retriever" que és el que busca informació, i la cadena anterior que prepara la resposta
    
    return qa_chain

# Funció que fa la pregunta
import time
def fer_pregunta(qa_chain, pregunta: str):
    start_time = time.time()
    resultat = qa_chain.invoke({"input": pregunta})
    end_time = time.time()
    #Per fer proves a veure quant tarda en respondre
    print(f"Temps de resposta: {end_time - start_time:.2f} segons")
    return resultat


# Inicialitza la cadena de preguntes i respostes per a que es pugui executar des de chat.py més facilment
def inizialitzar_cadena():
    document = carregar_document("Documents/Informació_drogues.json")
    chunks = dividir_chunks(document)
    vectordb = crear_carregar_vectors(chunks)
    llm = connectar_llamacpp()
    qa_chain = crear_RAG(llm, vectordb)
    return qa_chain


if __name__ == "__main__":
    document = carregar_document("Documents/deepseek_json_20251002_c19308.json")
    chunks = dividir_chunks(document)
    vectordb = crear_carregar_vectors(chunks)
    llm = connectar_llamacpp()
    qa_chain = crear_RAG(llm, vectordb)

    # Prueba con diferentes tipos de preguntas
    preguntas = [
        "Quins són els efectes del cannabis?",
        "Quina és la dosi de cocaïna?",
        "Què passa si barrejo alcohol amb cocaïna?"
    ]
    
    for pregunta in preguntas:
        print(f"\n🧪 PROVA: {pregunta}")
        resultat = fer_pregunta(qa_chain, pregunta)
        print(f"✅ RESPOSTA: {resultat['answer'][:500]}...")  # Muestra primeros 500 chars