import requests
import time
import subprocess

def test_gpu():
    print("🎯 Test final de GPU AMD...")
    
    # Verificar GPU antes
    result_before = subprocess.run(['rocm-smi'], capture_output=True, text=True)
    print("📊 GPU ANTES:")
    for line in result_before.stdout.split('\n'):
        if 'GPU%' in line or 'VRAM%' in line:
            print(line)
    
    # Test con API directa
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b",
        "prompt": "Responde en una palabra: hola",
        "stream": False
    }
    
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end = time.time()
        
        if response.status_code == 200:
            tiempo = end - start
            print(f"✅ Tiempo: {tiempo:.2f}s")
            print(f"📝 Respuesta: {response.json()['response']}")
            
            # Verificar GPU después
            result_after = subprocess.run(['rocm-smi'], capture_output=True, text=True)
            print("📊 GPU DESPUÉS:")
            for line in result_after.stdout.split('\n'):
                if 'GPU%' in line or 'VRAM%' in line:
                    print(line)
            
            if tiempo < 3:
                print("🎉 ¡GPU FUNCIONANDO CORRECTAMENTE!")
            else:
                print("⚠️  Posiblemente usando CPU")
        else:
            print(f"❌ Error HTTP: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

test_gpu()