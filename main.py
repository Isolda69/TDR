import os
import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaLLM
import time

# Carregar l'arxiu de text:
def carregar_document(ruta_txt: str):                               #La part de str indica a VisualStudio que sera un "string" és a dir, text, per a que així pugui detectar els errors més facilment
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
    
    loader = TextLoader(ruta_txt, encoding='utf-8')                 #Crea una variable que sap llegir l'arxiu, la part d'encodig fa que entengui més tipus de caràcters
    documents = loader.load()                                       #Carrega el contingut de l’arxiu en una llista de documents
    return documents

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
        chunk_size = chunk_size,                                                         # Defineix la mida indicada en el paràmetre
        chunk_overlap = chunk_overlap                                                    # Defineix la superposició indicada en el paràmetre
    )
    chunks = splitter.split_documents(documents)                                         # Indiquem quin text ha de separar i utilitza una funció per mantenir millor el context
    return chunks


# Convertir els chunks en vectors i els guarda en una base de dades:
def crear_vectors(chunks, persist_directory: str = "vectordb"):                                     # Persist_directory és la carpeta on Chroma guarda els fitxers en el disc, per defecte definim la carpeta "vectordb"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")   # Indica el model que utilitzarà per a fer la traducció de text a vectors 
    if os.path.exists(persist_directory):                                                           # Comprova si la carpeta on guarda els vectors ja existeix
        vectordb = Chroma(                                                                          # Si existeix, carrega els vectors que ja hi ha guardats per reutilitzar-los i no tornar a crear-los 
            persist_directory=persist_directory,                                                    # Indica la carpeta on estan guardats els fitxers
            embedding_function=embedding_model)                                                     # Indica el model que s'ha utilitzat per crear els vectors
    
    else:                                                                                           # Si no existeix la carpeta crea els vectors des de zero
        vectordb = Chroma.from_documents(                                                           # Crea una base de dades de vectors a partir dels chunks utilitzant el model especificat abans, i especifica la carpeta on guarda els fitxers
            documents=chunks,                                                                       # Indica els documents que ha de convertir en vectors
            embedding=embedding_model,                                                              # Indica el model que s'ha d'utilitzar per crear els vectors
            persist_directory=persist_directory                                                     # Indica la carpeta on guardar els fitxers
        )                                                                          
    return vectordb                                                                                 # Això el que retorna és la ubicació on Chroma ha guardat aquests vectors
# Si el document amb la informació canvia, cal esborrar la carpeta on guarda els vectors i automàticament torna a procesar el nou document, sinó el reutilitza i va més ràpid
# Per borrar la carpeta cal executar: "rm -rf vectordb"

# Carregar la base de dades vectorial existent, només la podem utilitzar si sabem segur que ja ha estat creada, sinó hi haurà un error
def carregar_vectordb():                                                                            # No necessita cap paràmetre perquè sempre carregarà la mateixa carpeta
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")   # Indica el model que utilitzarà per a fer la traducció de text a vectors
    vectordb = Chroma(persist_directory="vectordb", embedding_function=embedding_model)             # Carrega la base de dades vectorial que ja existeix a la carpeta "vectordb"
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

# Connectar Ollama amb el model que hem triat i instal·lat localment:
def connectar_llamacpp(model_name: str = "mistral:7b"):                   # En el paràmetre indiquem el nom del model que volem utilitzar
    llm = OllamaLLM(                                                      # Crea la connexió amb Ollama
        model=model_name,                                                 # Indica quin model s'ha d'utilitzar
        temperature=0.3,                                                  # Defineix la temperatura del model, com més alta més creatiu serà el model
        top_p=0.9,                                                        # Defineix el top_p del model, com més alt més opcions tindrà per triar la següent paraula
        repeat_penalty=1.1)                                               # Defineix la penalització per repetició, com més alta menys repetirà paraules o frases
    return llm

# Crear el sistema RAG, unint la base vectorial de Chroma amb el model:
def crear_RAG(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})             # Busca els troços de text més semblants a la pregunta, en aquest cas n'hem definit 2
    
    qa_prompt = ChatPromptTemplate.from_template(                         # Crea el prompt inicial que s'enviarà al model juntament amb els fragments de text rellevants
        """                      
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
def fer_pregunta(qa_chain, pregunta: str):                          # Li passem de paràmetres el sistema qa_chain i la pregunta que volem fer
    start_time = time.time()                                        # Inicia el comptador de temps per saber quant triga en respondre
    resultat = qa_chain.invoke({"input": pregunta})                 # Crida la cadena de preguntes i respostes, i li passa la pregunta
    end_time = time.time()                                          # Acaba el comptador de temps
    #Per fer proves a veure quant tarda en respondre
    print(f"Temps de resposta: {end_time - start_time:.2f} segons") # Imprimeix el temps que ha trigat en respondre
    return resultat


# Inicialitza la cadena de preguntes i respostes per a que es pugui executar des de chat.py més facilment
def inicialitzar_cadena():
    document = carregar_document("Documents/Informació_drogues.json")
    chunks = dividir_chunks(document)
    vectordb = crear_vectors(chunks)
    llm = connectar_llamacpp()
    qa_chain = crear_RAG(llm, vectordb)
    return qa_chain


if __name__ == "__main__":
    document = carregar_document("Documents/deepseek_json_20251002_c19308.json")
    chunks = dividir_chunks(document)
    vectordb = crear_vectors(chunks)
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