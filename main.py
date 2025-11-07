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

# Carregar el fitxer de text:
def carregar_document(ruta_txt: str):                               # La part de str indica a VisualStudio que serà un "string" és a dir, text, perquè així pugui detectar els errors més fàcilment
    if ruta_txt.endswith('.json'):                                  # Si el fitxer és un json, utilitza aquesta part del codi
        with open(ruta_txt, 'r', encoding='utf-8') as f:            # Obre el fitxer i el guarda a la variable "f"
            dades = json.load(f)                                    # Guardem totes les dades en un diccionari                            
        substancies = dades.get("substàncies", {})                  # Extreu els valors de cada substància, si no hi ha la clau substàncies, retorna un diccionari buit
        metadades = dades.get("metadades", {})                      # Extreu els valors de les metadades, si no hi ha la clau metadades, retorna un diccionari buit
        documents = []                                              # Crea la llista que al final tindrà totes les dades ben estructurades i amb text normal
        for id_s, info in substancies.items():                      # Recorre un bucle per cada una de les substàncies i en guarda el nom i la infomació per separat
            contingut = format_mes_compacte(id_s, info, metadades)  # Amb la funció que extreu l'estructura, netegem la informació de cada element
            documents.append(Document(page_content=contingut))      # Anem afegint cada substància a la llista final
        return documents                                            # Si és un fitxer JSON, ja retorna la llista aquí i no farà les dues comandes següents, que són en cas que sigui un fitxer TXT
    
    loader = TextLoader(ruta_txt, encoding='utf-8')                 # Crea una variable que sap llegir el fitxer, la part d'encoding fa que entengui més tipus de caràcters
    documents = loader.load()                                       # Carrega el contingut de el fitxer en una llista de documents
    return documents

def format_mes_compacte(id_s, info, metadades):
    try:                                                            # Prova d'executar aquesta part, si no funciona executarà l'"except"
        nom = info.get("nom_cientific", "No especificat")           # Guarda en la variable "nom" el nom de la substància, si no el troba l'estableix com a no especificat
        
        # Efectes desitjats a curt termini
        efectes_pos = info.get("efectes_desitjats_curt", [])        # Busca els efectes desitjats a curt termini i si no els troba els estableix com una llista buida
        if isinstance(efectes_pos, list) and efectes_pos:           # Comprova que hi hagi efectes positius i dins aquest condicional mira si són una llista
            efectes_pos_str = ", ".join(efectes_pos[:3])            # En cas que ho siguin agafa només els tres primers elements i els ajunta amb comes
        else:
            efectes_pos_str = "efectes variables"                   # Si no és una llista o aquesta està buida, defineix els efectes com si fossin variables
        
        # Efectes no desitjats a curt termini  
        efectes_neg = info.get("efectes_no_desitjats_curt", [])    # Busca els efectes no desitjats a curt termini i si no els troba els estableix com una llista buida
        if isinstance(efectes_neg, list) and efectes_neg:          # Comprova que n'hi hagi i que siguin una llista
            efectes_neg_str = ", ".join(efectes_neg[:3])           # Si són una llista, n'agafa els tres primers
        else:
            efectes_neg_str = "efectes variables"                  # En cas que no n'hi hagi, posa que són variables
        
        dosi = info.get("dosi", "No especificada")                 # Busca la dosi de la substància, si no la troba ho indica
        durada = info.get("durada_efectes", "No especificada")     # Busca la durada dels efectes i si no n'hi ha ho diu
        
        # Text natural
        text = f"{id_s.capitalize()} ({nom}): Produeix {efectes_pos_str}. "    # Posa la primera lletra del nom en majúscula, li posa dos punt i la paraula "Produeix" seguida dels efectes 
        text += f"Efectes negatius: {efectes_neg_str}. "                       # Afegeix a la frase d'abans les paraules "Efectes negatius" seguides dels efectes negatius
        text += f"Dosi: {dosi}. Durada: {durada}."                             # Afegeix la dosi i la durada indicant-ho amb les paraules
        
        return text                                                            # Retorna la frase completa que conté totes les parts però en una afirmació curta
        
    except Exception as e:
        return f"{id_s}: Informació disponible sobre aquesta substància."      # Això només s'executa si falla alguna part del codi anterior, torna el nom de la substància i un missatge bàsic



# Dividir el text en fragments més petits "chunks":
def dividir_chunks(documents, chunk_size: int = 1500, chunk_overlap: int = 200):         # La superposició indica que el següent fragment agafi els últims 200 caràcters per tal de no tallar frases per la meitat
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
    return vectordb                                                                                 # Això retorna la ubicació on Chroma ha guardat aquests vectors
# Si el document amb la informació canvia, cal esborrar la carpeta on guarda els vectors i automàticament torna a processar el nou document, sinó el reutilitza i va més ràpid
# Per borrar la carpeta cal executar: "rm -rf vectordb"

# Carregar la base de dades vectorial existent, només la podem utilitzar si sabem segur que ja ha estat creada, sinó hi haurà un error
def carregar_vectordb():                                                                            # No necessita cap paràmetre perquè sempre carregarà la mateixa carpeta
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")   # Indica el model que utilitzarà per a fer la traducció de text a vectors
    vectordb = Chroma(persist_directory="vectordb", embedding_function=embedding_model)             # Carrega la base de dades vectorial que ja existeix a la carpeta "vectordb"
    return vectordb

# Inicialitza tot el sistema abans de començar a fer preguntes
def preescalfament():
    vectordb = carregar_vectordb()                                      # Carrega la base de dades vectorial existent
    llm = connectar_ollama()                                            # Connecta amb el model d'Ollama
    qa_chain = crear_RAG(llm, vectordb)                                 # Uneix el model amb la base de dades vectorial per crear el sistema RAG
    print("Preescalfant el model...")                                   # Imprimeix un missatge per indicar que està preescalfant el model
    try:                                                                # Prova d'executar aquesta part, si no funciona executarà l'"except", per tal de detectar errors
        fer_pregunta(qa_chain, "Hola")                                  # Fa una pregunta senzilla per preescalfar el model
        print("Model preescalfat correctament!")                        # Imprimeix un missatge per indicar que el preescalfament ha anat bé
    except Exception as e:                                              # Si hi ha algun error en el preescalfament, imprimeix un missatge indicant que hi ha un error
        print(f"S'ha produït un error durant el preescalfament: {e}")   # Imprimeix l'error que s'ha produït
    
    return qa_chain

# Connectar Ollama amb el model que hem triat i instal·lat localment:
def connectar_ollama(model_name: str = "mistral:7b"):                   # En el paràmetre indiquem el nom del model que volem utilitzar
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
        - Al final de la resposta, afegeix: "Recorda que la manera més segura d'evitar riscos és no consumir. Si decideixes consumir, informa't sempre i analitza la substància amb Energy Control per conèixer la seva composició real."

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


# Inicialitza la cadena de preguntes i respostes perquè es pugui executar des de chat.py més fàcilment
def inicialitzar_cadena():
    document = carregar_document("Documents/Informació_drogues.json")       # Carrega el document amb la informació
    chunks = dividir_chunks(document)                                       # Crea els chunks a partir del document
    vectordb = crear_vectors(chunks)                                        # Crea la base de dades vectorial a partir dels chunks
    llm = connectar_ollama()                                                # Connecta amb el model d'Ollama
    qa_chain = crear_RAG(llm, vectordb)                                     # Uneix el model amb la base de dades vectorial per crear el sistema RAG
    return qa_chain                                                         # Retorna la cadena completa perquè es pugui utilitzar en altres fitxers

