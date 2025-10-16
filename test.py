from main import inicialitzar_chat, fer_pregunta
import time
import subprocess

def test_velocidad():
    print("🎯 INICIANDO TEST DE VELOCIDAD")
    print("=" * 50)
    
    # 1. Inicialización
    print("🔄 Inicializando sistema...")
    start_init = time.time()
    qa_chain = inicialitzar_chat()
    end_init = time.time()
    print(f"✅ Inicialización: {end_init - start_init:.2f}s")
    print()
    
    # 2. Monitoreo GPU antes
    print("📊 Estado GPU ANTES de preguntas:")
    result_before = subprocess.run(['rocm-smi'], capture_output=True, text=True)
    for line in result_before.stdout.split('\n'):
        if 'GPU%' in line or 'VRAM%' in line:
            print(line)
    print()
    
    # 3. Preguntas de prueba
    preguntas = [
        "Quins són els efectes del cannabis?",
        "Quina és la dosi de cocaïna?",
        "Què passa si barrejo alcohol amb cocaïna?"
    ]
    
    tiempos = []
    
    for i, pregunta in enumerate(preguntas, 1):
        print(f"🧪 PREGUNTA {i}: {pregunta}")
        start_pregunta = time.time()
        resultado = fer_pregunta(qa_chain, pregunta)
        end_pregunta = time.time()
        
        tiempo_pregunta = end_pregunta - start_pregunta
        tiempos.append(tiempo_pregunta)
        
        print(f"⏱️  Tiempo respuesta: {tiempo_pregunta:.2f}s")
        print(f"📝 Respuesta: {resultado['answer'][:100]}...")
        print()
    
    # 4. Estadísticas finales
    print("📈 ESTADÍSTICAS FINALES:")
    print(f"📊 Tiempo promedio: {sum(tiempos)/len(tiempos):.2f}s")
    print(f"⚡ Mejor tiempo: {min(tiempos):.2f}s")
    print(f"🐌 Peor tiempo: {max(tiempos):.2f}s")
    print()
    
    # 5. Monitoreo GPU después
    print("📊 Estado GPU DESPUÉS de preguntas:")
    result_after = subprocess.run(['rocm-smi'], capture_output=True, text=True)
    for line in result_after.stdout.split('\n'):
        if 'GPU%' in line or 'VRAM%' in line:
            print(line)

if __name__ == "__main__":
    test_velocidad()