import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import time
from ollama_chatbot import OllamaChatbot

class ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 ChatBot Académico")
        
        # Adaptación a tamaño de pantalla
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Tamaño inicial basado en pantalla (80% máximo)
        init_width = min(1000, int(screen_width * 0.8))
        init_height = min(700, int(screen_height * 0.8))
        
        # Centrar ventana
        x = (screen_width - init_width) // 2
        y = (screen_height - init_height) // 2
        
        self.root.geometry(f"{init_width}x{init_height}+{x}+{y}")
        self.root.minsize(600, 400)  # Más pequeño para móviles/tablets
        
        # Colores modernos
        self.colors = {
            'bg_main': '#1a1a1a',
            'bg_secondary': '#2d2d2d', 
            'bg_chat': '#0f0f0f',
            'user_bubble': '#0084ff',
            'bot_bubble': '#3a3a3c',
            'text_primary': '#ffffff',
            'text_secondary': '#b3b3b3',
            'accent': '#00d4aa',
            'border': '#404040'
        }
        
        self.root.configure(bg=self.colors['bg_main'])

        self.chatbot = OllamaChatbot()
        
        # Fuentes
        self.font_chat = font.Font(family="Segoe UI", size=10)
        self.font_input = font.Font(family="Segoe UI", size=11)
        self.font_header = font.Font(family="Segoe UI", size=9, weight="bold")

        # Verificar conexión
        if not self.chatbot.check_ollama_status():
            messagebox.showerror("Error", "Ollama no está ejecutándose. Inicia ollama serve.")
        
        self._build_layout()
        self._bind_events()
        self._add_welcome_message()

    def _build_layout(self):
        # Header con título y controles
        self.header_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=60)
        self.header_frame.pack(fill=tk.X, padx=0, pady=0)
        self.header_frame.pack_propagate(False)

        # Título del chat
        title_label = tk.Label(self.header_frame, text="🤖 ChatBot Académico", 
                              font=self.font_header, bg=self.colors['bg_secondary'], 
                              fg=self.colors['text_primary'])
        title_label.pack(side=tk.LEFT, padx=15, pady=15)

        # Controles en header - adaptativos
        controls_frame = tk.Frame(self.header_frame, bg=self.colors['bg_secondary'])
        controls_frame.pack(side=tk.RIGHT, padx=15, pady=10)

        # Botón prompts - más compacto en pantallas pequeñas
        window_width = self.root.winfo_width() if self.root.winfo_width() > 1 else 1000
        btn_text = "📚" if window_width < 800 else "📚 Prompts"
        self.prompt_btn = tk.Button(controls_frame, text=btn_text, 
                                   command=self.open_prompts_window,
                                   bg=self.colors['accent'], fg='white', 
                                   relief=tk.FLAT, padx=15, pady=5,
                                   font=self.font_chat)
        self.prompt_btn.pack(side=tk.LEFT, padx=5)

        # Configuración - más compacto en pantallas pequeñas  
        btn_text = "⚙️" if window_width < 800 else "⚙️ Config"
        self.config_btn = tk.Button(controls_frame, text=btn_text, 
                                   command=self.open_config_window,
                                   bg=self.colors['bot_bubble'], fg='white',
                                   relief=tk.FLAT, padx=15, pady=5,
                                   font=self.font_chat)
        self.config_btn.pack(side=tk.LEFT, padx=5)

        # Área principal de chat
        self.chat_frame = tk.Frame(self.root, bg=self.colors['bg_chat'])
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Canvas para scroll personalizado  
        self.canvas = tk.Canvas(self.chat_frame, bg=self.colors['bg_chat'], 
                               highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors['bg_chat'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=1)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Hacer que el scrollable_frame se expanda al ancho del canvas
        def configure_scroll_region(event):
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas.find_all()[0], width=canvas_width)
        self.canvas.bind('<Configure>', configure_scroll_region)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Input area en la parte inferior
        self.input_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=80)
        self.input_frame.pack(fill=tk.X, padx=0, pady=0)
        self.input_frame.pack_propagate(False)

        # Container para el input
        input_container = tk.Frame(self.input_frame, bg=self.colors['bg_secondary'])
        input_container.pack(fill=tk.X, padx=20, pady=15)

        # Entry estilizado
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(input_container, textvariable=self.entry_var,
                             font=self.font_input, bg=self.colors['bg_main'],
                             fg=self.colors['text_primary'], relief=tk.FLAT,
                             bd=10, insertbackground=self.colors['text_primary'])
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)

        # Botón enviar moderno
        self.send_btn = tk.Button(input_container, text="➤", 
                                 command=self.send_message,
                                 bg=self.colors['user_bubble'], fg='white',
                                 relief=tk.FLAT, padx=20, pady=8,
                                 font=font.Font(size=14, weight="bold"))
        self.send_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Status bar
        self.status_frame = tk.Frame(self.root, bg=self.colors['bg_main'], height=25)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Listo")
        self.status_label = tk.Label(self.status_frame, textvariable=self.status_var, 
                                    bg=self.colors['bg_main'], fg=self.colors['text_secondary'],
                                    font=font.Font(size=8), anchor='w')
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

    def _bind_events(self):
        self.entry.bind('<Return>', lambda e: self.send_message())
        self.root.bind('<Control-Return>', lambda e: self.send_message())
        # Scroll con mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Redimensionar ventana - actualizar wraplength de mensajes
        self.root.bind('<Configure>', self._on_window_resize)

    def _on_window_resize(self, event):
        """Manejar redimensionamiento de ventana para adaptabilidad"""
        if event.widget == self.root:
            # Reajustar canvas
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _add_welcome_message(self):
        self._add_message("bot", "¡Hola! Soy tu ChatBot Académico 🤖\n\nPuedo ayudarte con:\n• Matemáticas y cálculo\n• Programación y algoritmos\n• Ciencias naturales\n• Redacción académica\n• Y mucho más...\n\nUsa el botón 📚 Prompts para especializarme en un área.")

    def _add_message(self, sender: str, text: str, is_typing: bool = False):
        # Container para el mensaje con padding adecuado
        msg_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg_chat'])
        msg_container.pack(fill=tk.X, padx=0, pady=5)
        
        # Calcular ancho máximo basado en ventana (responsivo)
        window_width = self.root.winfo_width() if self.root.winfo_width() > 1 else 1000
        max_width = max(250, int(window_width * 0.65))  # 65% del ancho de ventana, mínimo 250px
        
        if sender == "user":
            # Frame contenedor para alinear a la derecha
            right_frame = tk.Frame(msg_container, bg=self.colors['bg_chat'])
            right_frame.pack(fill=tk.X, padx=20)
            
            # Mensaje del usuario (derecha, azul)
            msg_frame = tk.Frame(right_frame, bg=self.colors['user_bubble'], 
                               relief=tk.FLAT, bd=0)
            msg_frame.pack(side=tk.RIGHT)
            
            label = tk.Label(msg_frame, text=text, bg=self.colors['user_bubble'],
                           fg='white', font=self.font_chat, wraplength=max_width-50,
                           justify=tk.LEFT, padx=12, pady=8, anchor='w')
            label.pack()
            
        else:
            # Frame contenedor para alinear a la izquierda
            left_frame = tk.Frame(msg_container, bg=self.colors['bg_chat'])
            left_frame.pack(fill=tk.X, padx=20)
            
            # Mensaje del bot (izquierda, gris)
            msg_frame = tk.Frame(left_frame, bg=self.colors['bot_bubble'],
                               relief=tk.FLAT, bd=0)
            msg_frame.pack(side=tk.LEFT)
            
            if is_typing:
                # Indicador de escribiendo
                self.typing_label = tk.Label(msg_frame, text="●●●", bg=self.colors['bot_bubble'],
                                           fg=self.colors['text_secondary'], font=self.font_chat,
                                           padx=12, pady=8)
                self.typing_label.pack()
                self.typing_container = msg_container
                self._animate_typing()
            else:
                label = tk.Label(msg_frame, text=text, bg=self.colors['bot_bubble'],
                               fg='white', font=self.font_chat, wraplength=max_width,
                               justify=tk.LEFT, padx=12, pady=8, anchor='w')
                label.pack()
        
        # Auto scroll al final
        self.root.after(10, self._scroll_to_bottom)
        return msg_container

    def _animate_typing(self):
        """Anima los puntos de 'escribiendo...' como WhatsApp"""
        if hasattr(self, 'typing_label') and self.typing_label.winfo_exists():
            current = self.typing_label.cget('text')
            if current == "●":
                next_text = "●●"
            elif current == "●●":
                next_text = "●●●"
            else:
                next_text = "●"
            
            self.typing_label.config(text=next_text)
            # Continuar animación cada 500ms
            self.root.after(500, self._animate_typing)

    def _remove_typing_indicator(self):
        """Elimina el indicador de escribiendo"""
        if hasattr(self, 'typing_container') and self.typing_container.winfo_exists():
            self.typing_container.destroy()
            delattr(self, 'typing_container')
            if hasattr(self, 'typing_label'):
                delattr(self, 'typing_label')

    def _scroll_to_bottom(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def open_config_window(self):
        """Ventana de configuración moderna"""
        win = tk.Toplevel(self.root)
        win.title("⚙️ Configuración")
        win.geometry("400x300")
        win.configure(bg=self.colors['bg_main'])
        win.resizable(False, False)
        
        # Centrar ventana
        win.transient(self.root)
        win.grab_set()

        # Modelo
        tk.Label(win, text="Modelo:", bg=self.colors['bg_main'], 
                fg=self.colors['text_primary'], font=self.font_header).pack(pady=(20,5))
        
        self.model_var = tk.StringVar(value=self.chatbot.model_name)
        model_frame = tk.Frame(win, bg=self.colors['bg_main'])
        model_frame.pack(pady=5)
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                       width=25, state='readonly')
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        refresh_btn = tk.Button(model_frame, text="🔄", command=self.refresh_models,
                               bg=self.colors['accent'], fg='white', relief=tk.FLAT)
        refresh_btn.pack(side=tk.LEFT, padx=5)

        # Temperature
        tk.Label(win, text="Creatividad (Temperature):", bg=self.colors['bg_main'],
                fg=self.colors['text_primary'], font=self.font_header).pack(pady=(20,5))
        
        temp_frame = tk.Frame(win, bg=self.colors['bg_main'])
        temp_frame.pack(pady=10, fill=tk.X, padx=50)
        
        self.temp_var = tk.DoubleVar(value=self.chatbot.temperature)
        self.temp_scale = tk.Scale(temp_frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL,
                                  variable=self.temp_var, command=self._temp_changed,
                                  bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                                  highlightthickness=0, resolution=0.1)
        self.temp_scale.pack(fill=tk.X)
        
        self.temp_value_label = tk.Label(temp_frame, text=f"{self.chatbot.temperature:.1f}",
                                        bg=self.colors['bg_main'], fg=self.colors['accent'],
                                        font=self.font_header)
        self.temp_value_label.pack(pady=5)

        # Botones
        btn_frame = tk.Frame(win, bg=self.colors['bg_main'])
        btn_frame.pack(pady=20)
        
        reset_btn = tk.Button(btn_frame, text="🔄 Reset Prompt", command=self.reset_prompt,
                             bg=self.colors['bot_bubble'], fg='white', relief=tk.FLAT,
                             padx=15, pady=8, font=self.font_chat)
        reset_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = tk.Button(btn_frame, text="✓ Cerrar", command=win.destroy,
                             bg=self.colors['accent'], fg='white', relief=tk.FLAT,
                             padx=15, pady=8, font=self.font_chat)
        close_btn.pack(side=tk.LEFT, padx=10)
        
        self.refresh_models()

    def refresh_models(self):
        models = self.chatbot.list_available_models()
        base_names = [m.split(':')[0] for m in models]
        if not base_names:
            messagebox.showwarning("Modelos", "No se detectaron modelos. Usa la terminal para hacer pull.")
            return
        if hasattr(self, 'model_combo'):
            self.model_combo['values'] = base_names
            if self.chatbot.model_name not in base_names:
                self.model_var.set(base_names[0])
        self.status_var.set("Modelos actualizados")

    def _temp_changed(self, val):
        try:
            v = float(val)
            self.chatbot.set_temperature(v)
            if hasattr(self, 'temp_value_label'):
                self.temp_value_label.config(text=f"{v:.1f}")
        except ValueError:
            pass

    def reset_prompt(self):
        self.chatbot.reset_prompt()
        self.status_var.set("Prompt base activo")

    def open_prompts_window(self):
        win = tk.Toplevel(self.root)
        win.title("📚 Prompts Académicos")
        win.geometry("700x500")
        win.configure(bg=self.colors['bg_main'])
        win.transient(self.root)
        win.grab_set()

        # Header
        header = tk.Frame(win, bg=self.colors['bg_secondary'], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="📚 Selecciona un Prompt Especializado", 
                font=self.font_header, bg=self.colors['bg_secondary'], 
                fg=self.colors['text_primary']).pack(pady=15)

        # Search
        search_frame = tk.Frame(win, bg=self.colors['bg_main'])
        search_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(search_frame, text="🔍 Buscar:", bg=self.colors['bg_main'], 
                fg=self.colors['text_primary']).pack(side=tk.LEFT)
        
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var,
                               bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                               relief=tk.FLAT, font=self.font_chat)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, ipady=5)

        # Lista con estilo
        list_frame = tk.Frame(win, bg=self.colors['bg_main'])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Style para treeview
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Treeview", 
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['text_primary'],
                       fieldbackground=self.colors['bg_secondary'],
                       borderwidth=0)
        style.configure("Custom.Treeview.Heading",
                       background=self.colors['bg_main'],
                       foreground=self.colors['text_primary'])

        tree = ttk.Treeview(list_frame, columns=("Categoria", "Titulo"), 
                           show='headings', style="Custom.Treeview")
        tree.heading("Categoria", text="Categoría")
        tree.heading("Titulo", text="Título")
        tree.column("Categoria", width=120)
        tree.column("Titulo", width=400)
        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Scrollbar para tree
        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=tree_scroll.set)

        # Fill
        def load_items(filter_text: str = ""):
            tree.delete(*tree.get_children())
            ft = filter_text.lower()
            for p in self.chatbot.prompts:
                if ft and (ft not in p['prompt'].lower() and ft not in p['titulo'].lower() and ft not in p['categoria'].lower()):
                    continue
                tree.insert('', tk.END, iid=str(p['id']), values=(p['categoria'], p['titulo']))
        load_items()

        def on_search(*_):
            load_items(search_var.get())
        search_var.trace_add('write', on_search)

        # Vista previa
        preview_frame = tk.Frame(win, bg=self.colors['bg_main'])
        preview_frame.pack(fill=tk.X, padx=20, pady=(0,10))
        
        tk.Label(preview_frame, text="Vista previa:", bg=self.colors['bg_main'],
                fg=self.colors['text_secondary'], font=self.font_chat).pack(anchor='w')
        
        detail_text = tk.Text(preview_frame, height=4, wrap=tk.WORD, state=tk.DISABLED, 
                             bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                             relief=tk.FLAT, font=self.font_chat)
        detail_text.pack(fill=tk.X, pady=5)

        def show_detail(event):
            sel = tree.selection()
            if not sel:
                return
            pid = int(sel[0])
            prompt = next((p for p in self.chatbot.prompts if p['id'] == pid), None)
            if prompt:
                detail_text.configure(state=tk.NORMAL)
                detail_text.delete('1.0', tk.END)
                detail_text.insert(tk.END, prompt['prompt'])
                detail_text.configure(state=tk.DISABLED)
        tree.bind('<<TreeviewSelect>>', show_detail)

        # Botones finales
        btn_frame = tk.Frame(win, bg=self.colors['bg_main'])
        btn_frame.pack(pady=15)
        
        def activate_prompt():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Selecciona", "Selecciona un prompt")
                return
            pid = int(sel[0])
            self.chatbot.use_prompt(pid)
            self.status_var.set(f"Prompt {pid} activado")
            win.destroy()
            
        tk.Button(btn_frame, text="✓ Activar Prompt", command=activate_prompt,
                 bg=self.colors['accent'], fg='white', relief=tk.FLAT,
                 padx=20, pady=10, font=self.font_chat).pack(side=tk.LEFT, padx=10)
                 
        tk.Button(btn_frame, text="✕ Cancelar", command=win.destroy,
                 bg=self.colors['bot_bubble'], fg='white', relief=tk.FLAT,
                 padx=20, pady=10, font=self.font_chat).pack(side=tk.LEFT, padx=10)

    def send_message(self):
        text = self.entry_var.get().strip()
        if not text:
            return

        # Cambio de modelo si seleccionó otro en combo
        if hasattr(self, 'model_var'):
            selected_model = self.model_var.get().strip()
            if selected_model and selected_model != self.chatbot.model_name:
                self.chatbot.change_model(selected_model)
                self.status_var.set(f"Modelo cambiado a {selected_model}")

        self.entry_var.set("")
        self._add_message("user", text)
        
        # Mostrar indicador de escribiendo
        self._add_message("bot", "", is_typing=True)
        self.status_var.set("🤖 Generando respuesta...")
        self.send_btn.config(state=tk.DISABLED, text="⏳")

        threading.Thread(target=self._generate_response_thread, args=(text,), daemon=True).start()

    def _generate_response_thread(self, user_text: str):
        start = time.time()
        response = self.chatbot.generate_response(user_text)
        elapsed = time.time() - start

        def finish():
            # Eliminar indicador de escribiendo
            self._remove_typing_indicator()
            # Agregar respuesta real
            self._add_message("bot", response)
            self.status_var.set(f"✓ Respuesta generada en {elapsed:.1f}s")
            self.send_btn.config(state=tk.NORMAL, text="➤")
        self.root.after(0, finish)


def main():
    root = tk.Tk()
    app = ChatGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
