import customtkinter as ctk
import threading
from tkinter import scrolledtext # Aquesta funció fa que el text pugui baixar
from main import fer_pregunta, inicialitzar_chat # Hem d'importar les funcions que hem definit en el fitxer main

qa_chain = inicialitzar_chat()  # Inizialitza la cadena de preguntes i respostes

def enviar_pregunta(): # Funció que s'executarà quan en la interfície s'enviï la pregunta
        pregunta = entrada.get()
        if pregunta.strip() == "": # Si la preguta està buida, no retorna res
            return
        
        # Aquesta part es mostra automàticament, per a informar a l'usuari de l'estat de la resposta
        quadre_conversa.configure(state='normal') # Habilita el quadre de text per poder escriure
        
        quadre_conversa.insert(ctk.END, f"Tu: {pregunta}\n") # Insereix la pregunta de l'usuari al quadre de text, i fa un salt de línia
        quadre_conversa.insert(ctk.END, "Model pensant...\n") # Insereix el text "Model pensant..." al quadre de text, i fa un salt de línia
        quadre_conversa.see(ctk.END) # Baixa el quadre de text automàticament
        quadre_conversa.configure(state='disabled') # Torna a deshabilitar el quadre de text
        entrada.delete(0, ctk.END) # Esborra el text que hi ha a l'entrada
        
         # Aquesta part es fa en paral·lel, per a que la interfície no sembli que s'ha quedat penjada
        def processar_pregunta(): # Aquesta funció es crida en paral·lel, per a que la interfície no sembli que s'ha quedat penjada
            resultat = fer_pregunta(qa_chain, pregunta) # Crea una variable resultat que conté la resposta a la pregunta
            resposta_bot = resultat["answer"] # Escriu la resposta del diccionari que ha creat el model
            
            quadre_conversa.configure(state='normal') # Habilita el quadre de text per poder escriure
            quadre_conversa.delete("end-2l", "end-1l") # Esborra la línia on posava "Model pensant..."
            
            quadre_conversa.insert(ctk.END, f"Bot: {resposta_bot}\n\n") # Mostra la resposta al quadre de text, i fa un salt de línia
            quadre_conversa.see(ctk.END) # Baixa el quadre de text automàticament
            quadre_conversa.configure(state='disabled') # Torna a deshabilitar el quadre de text
        
        threading.Thread(target=processar_pregunta).start() # Crida la funció processar_pregunta en paral·lel
            
        
         

root = ctk.CTk() # Crea la finestra que obrira
root.geometry("2000x1900") # Defineix la mida que tindrà la finestra
root.title("Prevenció de riscos en les drogues") # Defineix el títol de la finestra

quadre_conversa = scrolledtext.ScrolledText(root, wrap="word", width=200, height=45, state='disabled') # Crea un quadre de text que mostra la conversa
quadre_conversa.pack(padx=10, pady=10) # Posa marges al quadre de text

entrada = ctk.CTkEntry(root, width=700) # Crea l'espai on escrius la pregunta
entrada.pack(padx=10, pady=10) # Defineix la entrada de la pregunta al centre i posa marges
entrada.bind("<Return>", lambda event: enviar_pregunta()) # Envia la pregunta quan es prem Enter


boto_enviar = ctk.CTkButton(root, text="Enviar", command=enviar_pregunta) # És el botó d'enviar la pregunta, que executa la funció enviar_pregunta
boto_enviar.pack(padx=10, pady=10) # Defineix el botó al centre i posa marges

root.mainloop() # Executa el fitxer main fins que l'usuari tanqui la finestra