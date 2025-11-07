import customtkinter as ctk
import threading
from tkinter import scrolledtext                    # Aquesta funció fa que el text pugui baixar
from main import fer_pregunta, preescalfament    # Hem d'importar les funcions que hem definit en el fitxer main

# Configurar aparença de la interfície
ctk.set_appearance_mode("light")            # Defineix el mode clar 

# Definim els colors que farem servir
Color_principal = "#FFB6C1"               # Rosa clar
Color_secundari = "#FF69B4"               # Rosa més fort  
Color_fosc = "#DB7093"                    # Rosa fosc
Color_text = "#8B4513"                    # Marró clar, per fer contrast
Color_fons = "#FFF0F5"                    # Rosa molt clar


# Inicialitza la cadena de preguntes i respostes
qa_chain = preescalfament()  


# Configurar la finestra principal
root = ctk.CTk()                                                            # Crea la finestra que obrirà
root.geometry("1200x700")                                                   # Defineix la mida que tindrà la finestra
root.title("💊 SafeTrip - Assistent en reducció de riscos en les droges")   # Defineix el títol de la finestra
ample = root.winfo_screenwidth()                                            # Agafa l'amplada de la pantalla
alt = root.winfo_screenheight()                                             # Agafa l'alçada de la pantalla
root.geometry(f"{ample}x{alt}+0+0")                                         # Fes que la finestra ocupi tota la pantalla
root.configure(fg_color=Color_fons)                                         # Defineix el color de fons de la finestra

# Crear marc decoratiu amb cantonades arrodonides, que contindrà tots els elements de la interfície
marc_principal = ctk.CTkFrame(
    root,                               # Defineix que el marc està dins de la finestra principal
    fg_color=Color_principal,           # Color de fons del marc
    border_color=Color_secundari,       # Color de la vora del marc
    border_width=3,                     # Amplada de la vora del marc
    corner_radius=25                    # Radi de les cantonades arrodonides
)
marc_principal.pack(padx=20, pady=20, fill="both", expand=True) # Posa marges al marc i fa que s'ajusti a la finestra

# Posar el títol bonic
titol = ctk.CTkLabel(
    marc_principal,                     # Defineix que el títol està dins del marc principal
    text="💊 SafeTrip - Assistent en reducció de riscos en les droges",
    font=("Arial", 24, "bold"),         # Defineix la font del títol
    text_color=Color_text,              # Defineix el color del text del títol
    fg_color=Color_secundari,           # Defineix el color de fons del títol
    corner_radius=20,                   # Defineix el radi de les cantonades arrodonides del títol
    height=50                           # Defineix l'alçada del títol
)
titol.pack(padx=20, pady=(20, 10), fill="x") # Posa marges al títol i fa que s'ajusti a l'amplada

# Subtítol
subtitol = ctk.CTkLabel(
    marc_principal,                     # Defineix que el subtítol està dins del marc principal
    text="Consulta lliurement: informació fiable sobre drogues, sense judicis.",
    font=("Arial", 14),                 # Defineix la font del subtítol
    text_color=Color_text,              # Defineix el color del text del subtítol
    fg_color=Color_principal,           # Defineix el color de fons del subtítol
    corner_radius=15,                   # Defineix el radi de les cantonades arrodonides del subtítol
    height=30                           # Defineix l'alçada del subtítol
)
subtitol.pack(padx=20, pady=(0, 15), fill="x") # Posa marges al subtítol i fa que s'ajusti a l'amplada

# Crea el quadre de conversa
quadre_conversa = scrolledtext.ScrolledText(
    marc_principal,                     # Defineix que el quadre de conversa està dins del marc principal
    wrap="word",                        # Fa que el text s'ajusti a l'amplada del quadre
    state='disabled',                   # Fa que el quadre de conversa no es pugui editar directament
    font=("Arial", 12),                 # Defineix la font del text del quadre de conversa
    bg=Color_fons,                      # Defineix el color de fons del quadre de conversa
    fg=Color_text,                      # Defineix el color del text del quadre de conversa
    relief="flat",                      # Defineix que el quadre de conversa no tingui relleu
    borderwidth=10,                     # Defineix l'amplada de la vora del quadre de conversa
    padx=12,                            # Defineix els marges horitzontals del text dins del quadre de conversa
    pady=12                             # Defineix els marges verticals del text dins del quadre de conversa
)
quadre_conversa.pack(padx=20, pady=10, fill="both", expand=True) # Posa marges al quadre de conversa i fa que s'ajusti a la finestra

# Configurar els texts de pregunta, resposta i espera
quadre_conversa.tag_config("usuari", foreground=Color_fosc, font=("Arial", 12, "bold"))         # Text de l'usuari
quadre_conversa.tag_config("bot", foreground=Color_text, font=("Arial", 12))                    # Text del bot
quadre_conversa.tag_config("espera", foreground=Color_secundari, font=("Arial", 11, "italic"))  # Text d'espera
quadre_conversa.tag_config("error", foreground="#FF0000", font=("Arial", 12, "bold"))         # Text d'error

# Missatge de benvinguda
quadre_conversa.configure(state='normal')                               # Habilita el quadre de text per poder escriure
quadre_conversa.insert(ctk.END, "Benvingut/da a SafeTrip!\n\n", "bot")  # Insereix el missatge de benvinguda al quadre de conversa
quadre_conversa.insert(ctk.END, "Aquest és un espai segur i confidencial, on pots preguntar lliurement sobre qualsevol substància.\n\n", "bot")
quadre_conversa.insert(ctk.END, "Sense por de ser jutjat/da\n", "bot")
quadre_conversa.insert(ctk.END, "Conversa privada i anònima\n", "bot")
quadre_conversa.insert(ctk.END, "Informació científica i real\n\n", "bot")
quadre_conversa.insert(ctk.END, "Exemples de preguntes:\n", "bot")
quadre_conversa.insert(ctk.END, "· Quins efectes té barrejar alcohol amb cannabis?\n", "bot")
quadre_conversa.insert(ctk.END, "· Com puc reduir riscs si prenc MDMA?\n", "bot")
quadre_conversa.insert(ctk.END, "· Què puc fer si algú té un mal viatge?\n", "bot")
quadre_conversa.configure(state='disabled')                             # Torna a deshabilitar el quadre de text

# Marc per la entrada de text
marc_entrada = ctk.CTkFrame(
    marc_principal,             # Defineix que el marc està dins del marc principal
    fg_color=Color_principal,   # Defineix el color de fons del marc
    corner_radius=20            # Defineix el radi de les cantonades arrodonides del marc
)
marc_entrada.pack(padx=20, pady=10, fill="x") # Posa marges al marc i fa que s'ajusti a l'amplada de la finestra

# Entrada de text per la pregunta
entrada = ctk.CTkEntry(
    marc_entrada,                   # Defineix que l'entrada està dins del marc d'entrada
    height=45,                      # Defineix l'alçada de l'entrada
    font=("Arial", 14),             # Defineix la font del text de l'entrada
    fg_color="#FFFFFF",           # Defineix el color de fons de l'entrada
    text_color=Color_text,          # Defineix el color del text de l'entrada
    border_color=Color_secundari,   # Defineix el color de la vora de l'entrada
    border_width=2,                 # Defineix l'amplada de la vora de l'entrada
    corner_radius=20,               # Defineix el radi de les cantonades arrodonides de l'entrada
    placeholder_text="💭 Escriu la teva pregunta...", # Text que apareix quan l'entrada està buida
)
entrada.pack(padx=(20, 10), pady=12, side="left", fill="x", expand=True) # Posa marges a l'entrada i fa que s'ajusti a l'amplada del marc

# Botó per enviar la pregunta
boto_enviar = ctk.CTkButton(
    marc_entrada,                   # Defineix que el botó està dins del marc d'entrada
    text="Enviar",                  # Text que apareix al botó
    font=("Arial", 14, "bold"),     # Defineix la font del text del botó
    fg_color=Color_secundari,       # Defineix el color de fons del botó
    hover_color=Color_fosc,         # Defineix el color de fons del botó quan es passa el ratolí per sobre
    text_color="#FFFFFF",         # Defineix el color del text del botó
    border_color="#FFFFFF",       # Defineix el color de la vora del botó
    border_width=2,                 # Defineix l'amplada de la vora del botó
    width=120,                      # Defineix l'amplada del botó
    height=45,                      # Defineix l'alçada del botó
    corner_radius=20,               # Defineix el radi de les cantonades arrodonides del botó
    command=lambda: enviar_pregunta()# Crida la funció enviar_pregunta quan es prem el botó
)
entrada.bind("<Return>", lambda event: enviar_pregunta()) # Fa que quan l'usuari premi la tecla Enter, s'enviï la pregunta         
boto_enviar.pack(padx=(10, 20), pady=12, side="right")    # Posa marges al botó i el situa a la dreta del marc

# Barra inferior amb missatge d'emergència
barra_inferior = ctk.CTkLabel(
    marc_principal, # Defineix que la barra inferior està dins del marc principal
    text="Si tens alguna emergència contacta amb els serveis mèdics d'urgència: 112. Pots consultar més informació a Energy Control: https://energycontrol.org/",
    font=("Arial", 12, "bold"),
    text_color="#FFFFFF",
    fg_color=Color_fosc,
    corner_radius=15,
    height=30
)
barra_inferior.pack(padx=20, pady=(10, 20), fill="x")


# Funció que envia les preguntes
def enviar_pregunta():                                          # Funció que s'executa quan l'usuari prem el botó d'enviar
        pregunta = entrada.get()                                # Agafa el text que hi ha a l'entrada
        if pregunta.strip() == "":                              # Si la pregunta està buida, no retorna res
            return
        
        if "Benvingut/da a SafeTrip!" in quadre_conversa.get(1.0, ctk.END):
            quadre_conversa.configure(state='normal')
            quadre_conversa.delete(1.0, ctk.END)               # Esborra el missatge de benvinguda si és la primera pregunta
        
        quadre_conversa.configure(state='normal')              # Habilita el quadre de text per poder escriure
        
        quadre_conversa.insert(ctk.END, f"Tu: {pregunta}\n")   # Insereix la pregunta de l'usuari al quadre de text, i fa un salt de línia
        quadre_conversa.insert(ctk.END, "Bot pensant...\n")    # Insereix el text "Bot pensant..." al quadre de text, i fa un salt de línia
        quadre_conversa.see(ctk.END)                           # Baixa el quadre de text automàticament
        quadre_conversa.configure(state='disabled')            # Torna a deshabilitar el quadre de text
        entrada.delete(0, ctk.END)                             # Esborra el text que hi ha a l'entrada

        def processar_pregunta():                       # Aquesta funció es crida en paral·lel, perquè la interfície no sembli que s'hagi quedat penjada
            resultat = fer_pregunta(qa_chain, pregunta) # Crea una variable resultat que conté la resposta a la pregunta
            resposta_bot = resultat["answer"]           # Escriu la resposta del diccionari que ha creat el model
            
            quadre_conversa.configure(state='normal')   # Habilita el quadre de text per poder escriure
            quadre_conversa.delete("end-2l", "end-1l")  # Esborra la línia on posava "Bot pensant..."
            
            quadre_conversa.insert(ctk.END, f"Bot: {resposta_bot}\n\n") # Mostra la resposta al quadre de text, i fa un salt de línia
            quadre_conversa.see(ctk.END)                                # Baixa el quadre de text automàticament
            quadre_conversa.configure(state='disabled')                 # Torna a deshabilitar el quadre de text
        
        threading.Thread(target=processar_pregunta).start()             # Crida la funció processar_pregunta en paral·lel
            
        
root.mainloop() # Executa el fitxer main fins que l'usuari tanqui la finestra