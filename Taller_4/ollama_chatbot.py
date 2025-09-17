import requests
import json
import time
from typing import List, Dict, Optional
import os

class OllamaChatbot:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Inicializa el chatbot con Ollama
        
        Args:
            model_name: Nombre del modelo a usar (llama3.2, mistral, codellama, etc.)
            base_url: URL base de la API de Ollama
        """
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history: List[Dict] = []
        self.temperature: float = 0.7  # default creative level
        self.base_system_prompt = """Eres un asistente virtual inteligente y amigable. 
        Tu nombre es ChatBot Académico y tu propósito es ayudar a estudiantes con sus dudas académicas.
        Respondes de manera clara, educativa y siempre mantienes un tono profesional pero cercano.
        Si no sabes algo, lo admites honestamente y sugieres dónde podrían encontrar la información."""
        
        # Cargar prompts especializados
        self.prompts: List[Dict] = self._load_prompts()
        self.active_prompt_id: Optional[int] = None
        
        # Personalización del chatbot
        self.system_prompt = self.base_system_prompt

    def _load_prompts(self) -> List[Dict]:
        """Carga prompts académicos desde archivo externo."""
        filename = os.path.join(os.path.dirname(__file__), 'prompts_estudio.txt')
        prompts: List[Dict] = []
        if not os.path.exists(filename):
            return prompts
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    if len(parts) < 4:
                        continue
                    try:
                        pid = int(parts[0])
                    except ValueError:
                        continue
                    prompts.append({
                        'id': pid,
                        'categoria': parts[1].strip(),
                        'titulo': parts[2].strip(),
                        'prompt': parts[3].strip()
                    })
        except Exception as e:
            print(f"Advertencia: no se pudieron cargar prompts: {e}")
        return prompts
        
    def check_ollama_status(self) -> bool:
        """Verifica si Ollama está ejecutándose"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_available_models(self) -> List[str]:
        """Lista los modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except requests.exceptions.RequestException:
            return []
    
    def pull_model(self, model_name: str = None) -> bool:
        """Descarga un modelo si no está disponible"""
        model = model_name or self.model_name
        try:
            print(f"Descargando modelo {model}... (esto puede tomar varios minutos)")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        print(f"\r{data['status']}", end='', flush=True)
                    if data.get('status') == 'success':
                        print(f"\n✓ Modelo {model} descargado exitosamente")
                        return True
            return False
        except Exception as e:
            print(f"Error descargando modelo: {e}")
            return False
    
    def generate_response(self, user_message: str) -> str:
        """Genera una respuesta usando el modelo de Ollama"""
        try:
            # Construir el prompt con historial
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_message})
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result['message']['content']
                
                # Actualizar historial
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": bot_response})
                
                # Mantener solo los últimos 10 intercambios para no sobrecargar el contexto
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                return bot_response
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error de conexión: {e}"
        except Exception as e:
            return f"Error inesperado: {e}"
    
    def clear_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []
        print("✓ Historial de conversación limpiado")
    
    def change_model(self, new_model: str):
        """Cambia el modelo actual"""
        available_models = self.list_available_models()
        if new_model in [m.split(':')[0] for m in available_models]:
            self.model_name = new_model
            self.clear_history()  # Limpiar historial al cambiar modelo
            print(f"✓ Modelo cambiado a: {new_model}")
        else:
            print(f"✗ Modelo {new_model} no disponible. Modelos disponibles: {available_models}")

    # -------------------------- Gestión de Prompts --------------------------
    def list_prompts(self, page: int = 1, page_size: int = 10):
        total = len(self.prompts)
        if total == 0:
            print("No hay prompts cargados.")
            return
        pages = (total + page_size - 1) // page_size
        page = max(1, min(page, pages))
        start = (page - 1) * page_size
        end = start + page_size
        print(f"Prompts (página {page}/{pages}) - total {total}")
        for p in self.prompts[start:end]:
            activo = ' *' if p['id'] == self.active_prompt_id else ''
            print(f"[{p['id']:>3}] {p['categoria']:<12} | {p['titulo']}{activo}")
        if page < pages:
            print(f"/prompts {page+1} para siguiente página")

    def use_prompt(self, pid: int):
        for p in self.prompts:
            if p['id'] == pid:
                self.system_prompt = self.base_system_prompt + "\n\nInstrucción especializada activa:\n" + p['prompt']
                self.active_prompt_id = pid
                self.clear_history()
                print(f"✓ Prompt {pid} activado: {p['titulo']}")
                return
        print(f"✗ Prompt con id {pid} no encontrado")

    def search_prompts(self, keyword: str):
        keyword_lower = keyword.lower()
        matches = [p for p in self.prompts if keyword_lower in p['prompt'].lower() or keyword_lower in p['titulo'].lower() or keyword_lower in p['categoria'].lower()]
        if not matches:
            print("Sin resultados")
            return
        for p in matches[:30]:  # limitar salida
            print(f"[{p['id']}] {p['categoria']} - {p['titulo']}")
        if len(matches) > 30:
            print(f"... {len(matches)-30} más")

    def list_categories(self):
        cats = sorted({p['categoria'] for p in self.prompts})
        print("Categorías:")
        for c in cats:
            count = sum(1 for p in self.prompts if p['categoria'] == c)
            print(f"- {c} ({count})")

    def reset_prompt(self):
        self.system_prompt = self.base_system_prompt
        self.active_prompt_id = None
        self.clear_history()
        print("✓ Prompt base restaurado")

    def adjust_prompt(self, extra: str):
        self.system_prompt += "\n\nAjuste contextual temporal: " + extra.strip()
        print("✓ Ajuste añadido al system prompt (solo afectará a futuros mensajes)")

    # -------------------------- Configuración --------------------------
    def set_temperature(self, value: float):
        try:
            if not (0 <= value <= 2):
                print("Valor fuera de rango (0 - 2)")
                return
            self.temperature = float(value)
            print(f"✓ Temperature establecida en {self.temperature}")
        except ValueError:
            print("Valor inválido")

    def show_config(self):
        print("Configuración actual:")
        print(f"- Modelo: {self.model_name}")
        print(f"- Temperature: {self.temperature}")
        print(f"- Prompt activo: {'Base' if self.active_prompt_id is None else self.active_prompt_id}")
        print(f"- Longitud historial: {len(self.conversation_history)} mensajes")

    # -------------------------- Logging --------------------------
    def log_interaction(self, user: str, assistant: str):
        try:
            log_path = os.path.join(os.path.dirname(__file__), 'chat_logs.txt')
            with open(log_path, 'a', encoding='utf-8') as f:
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(json.dumps({
                    'timestamp': ts,
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'prompt_id': self.active_prompt_id,
                    'user': user,
                    'assistant': assistant[:4000]
                }, ensure_ascii=False) + '\n')
        except Exception:
            pass

def main():
    print("=" * 60)
    print("🤖 CHATBOT ACADÉMICO CON OLLAMA")
    print("=" * 60)
    
    # Inicializar chatbot
    chatbot = OllamaChatbot()
    
    # Verificar si Ollama está ejecutándose
    print("Verificando conexión con Ollama...")
    if not chatbot.check_ollama_status():
        print("❌ Error: Ollama no está ejecutándose.")
        print("\nPara iniciar Ollama:")
        print("1. Instala Ollama desde: https://ollama.ai")
        print("2. Ejecuta: ollama serve")
        print("3. En otra terminal ejecuta: ollama pull llama3.2")
        return
    
    print("✓ Ollama está ejecutándose")
    
    # Verificar modelos disponibles
    models = chatbot.list_available_models()
    if not models:
        print("No hay modelos instalados. Descargando llama3.2...")
        if not chatbot.pull_model("llama3.2"):
            print("❌ Error descargando el modelo")
            return
    else:
        print(f"✓ Modelos disponibles: {[m.split(':')[0] for m in models]}")
    
    print("\n" + "=" * 60)
    print("💬 ¡Chatbot listo! Escribe 'salir' para terminar")
    print("Comandos especiales:")
    print("- /limpiar: Borrar historial de conversación")
    print("- /modelos: Ver modelos disponibles")
    print("- /cambiar [modelo]: Cambiar modelo actual")
    print("- /prompts [pag]: Listar prompts académicos")
    print("- /usar [id]: Activar prompt por id")
    print("- /buscar [palabra]: Buscar en prompts")
    print("- /categorias: Listar categorías de prompts")
    print("- /reset: Restaurar prompt base")
    print("- /ajustar [texto]: Añadir ajuste temporal al prompt")
    print("- /temp [0-2]: Ajustar temperatura del modelo")
    print("- /config: Ver configuración actual")
    print("=" * 60)
    
    # Loop principal del chat
    while True:
        try:
            user_input = input("\n👤 Tú: ").strip()
            
            if not user_input:
                continue
            
            # Comandos especiales
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego! 👋")
                break
            
            elif user_input.lower() == '/limpiar':
                chatbot.clear_history()
                continue
            
            elif user_input.lower() == '/modelos':
                models = chatbot.list_available_models()
                print(f"Modelos disponibles: {[m.split(':')[0] for m in models]}")
                print(f"Modelo actual: {chatbot.model_name}")
                continue
            
            elif user_input.lower().startswith('/cambiar'):
                parts = user_input.split()
                if len(parts) > 1:
                    chatbot.change_model(parts[1])
                else:
                    print("Uso: /cambiar [nombre_del_modelo]")
                continue

            elif user_input.lower().startswith('/prompts'):
                parts = user_input.split()
                page = 1
                if len(parts) > 1 and parts[1].isdigit():
                    page = int(parts[1])
                chatbot.list_prompts(page=page)
                continue

            elif user_input.lower().startswith('/usar'):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    chatbot.use_prompt(int(parts[1]))
                else:
                    print("Uso: /usar [id]")
                continue

            elif user_input.lower().startswith('/buscar'):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    chatbot.search_prompts(parts[1])
                else:
                    print("Uso: /buscar [palabra_clave]")
                continue

            elif user_input.lower() == '/categorias':
                chatbot.list_categories()
                continue

            elif user_input.lower() == '/reset':
                chatbot.reset_prompt()
                continue

            elif user_input.lower().startswith('/ajustar'):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    chatbot.adjust_prompt(parts[1])
                else:
                    print("Uso: /ajustar [texto contextual]")
                continue

            elif user_input.lower().startswith('/temp'):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        chatbot.set_temperature(float(parts[1]))
                    except ValueError:
                        print("Uso: /temp [valor 0 a 2]")
                else:
                    print("Uso: /temp [valor 0 a 2]")
                continue

            elif user_input.lower() == '/config':
                chatbot.show_config()
                continue
            
            # Generar respuesta
            print("\n🤖 Bot: ", end="", flush=True)
            start_time = time.time()
            
            response = chatbot.generate_response(user_input)
            
            end_time = time.time()
            print(response)
            print(f"\n⏱️ Tiempo de respuesta: {end_time - start_time:.2f}s")
            chatbot.log_interaction(user_input, response)
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
