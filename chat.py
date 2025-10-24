import customtkinter as ctk
import threading
from tkinter import scrolledtext # Aquesta funció fa que el text pugui baixar
from main import fer_pregunta, inicialitzar_chat # Hem d'importar les funcions que hem definit en el fitxer main

# Configuar aparença de la interfície
ctk.set_appearance_mode("light") # Defineix el mode clar 

# Definim els colors que farem servir
Color_principal = "#FFB6C1" # Rosa clar
Color_secundari = "#FF69B4"  # Rosa més fort  
Color_fosc = "#DB7093"  # Rosa fosc
Color_text = "#8B4513"  # Marrón clar, per fer contraste
Color_fons = "#FFF0F5"  # Rosa muolt clar


qa_chain = inicialitzar_chat()  # Inizialitza la cadena de preguntes i respostes


# Configurar la finestra principal
root = ctk.CTk() # Crea la finestra que obrira
root.geometry("1400x900") # Defineix la mida que tindrà la finestra
root.title("💊 SafeTrip - Asistent en reducció de riscos en les droges") # Defineix el títol de la finestra
root.configure(fg_color=Color_fons) # Defineix el color de fons de la finestra

# Centrar la finestra a la pantalla
root.update_idletasks() # Actualitza les tasques pendents de la finestra
amplada = root.winfo_width() # Agafa l'amplada de la finestra
height = root.winfo_height() # Agafa l'alçada de la finestra
x = (root.winfo_screenwidth() // 2) - (amplada // 2) # Calcula la posició x per centrar la finestra
y = (root.winfo_screenheight() // 2) - (height // 2) # Calcula la posició y per centrar la finestra
root.geometry(f"+{x}+{y}") # Defineix la posició de la finestra

# Crear marc decoratiu amb cantonades arrodonides, que contindrà tots els elements de la interfície
marc_principal = ctk.CTkFrame(
    root, # Defineix que el marc està dins de la finestra principal
    fg_color=Color_principal, # Color de fons del marc
    border_color=Color_secundari, # Color de la vora del marc
    border_width=3, # Amplada de la vora del marc
    corner_radius=25 # Radi de les cantonades arrodonides
)
marc_principal.pack(padx=20, pady=20, fill="both", expand=True) # Posa marges al marc i fa que s'ajusti a la finestra

# Posar el títol bonic
titol = ctk.CTkLabel(
    marc_principal, # Defineix que el títol està dins del marc principal
    text="💊 SafeTrip - Asistent en reducció de riscos en les droges",
    font=("Arial", 24, "bold"), # Defineix la font del títol
    text_color=Color_text, # Defineix el color del text del títol
    fg_color=Color_secundari, # Defineix el color de fons del títol
    corner_radius=20, # Defineix el radi de les cantonades arrodonides del títol
    height=50 # Defineix l'alçada del títol
)
titol.pack(padx=20, pady=(20, 10), fill="x") # Posa marges al títol i fa que s'ajusti a l'amplada













  

quadre_conversa = scrolledtext.ScrolledText(root, wrap="word", width=200, height=45, state='disabled') # Crea un quadre de text que mostra la conversa
quadre_conversa.pack(padx=10, pady=10) # Posa marges al quadre de text

entrada = ctk.CTkEntry(root, width=700) # Crea l'espai on escrius la pregunta
entrada.pack(padx=10, pady=10) # Defineix la entrada de la pregunta al centre i posa marges
entrada.bind("<Return>", lambda event: enviar_pregunta()) # Envia la pregunta quan es prem Enter


boto_enviar = ctk.CTkButton(root, text="Enviar", command=enviar_pregunta) # És el botó d'enviar la pregunta, que executa la funció enviar_pregunta
boto_enviar.pack(padx=10, pady=10) # Defineix el botó al centre i posa marges


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
            
        
         

root.mainloop() # Executa el fitxer main fins que l'usuari tanqui la finestra