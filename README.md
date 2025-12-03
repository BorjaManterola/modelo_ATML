PAINE - Asistente local para trekking en Torres del Paine
==========================================================

Asistente de texto basado en un LoRA finetuned sobre TinyLlama-1.1B-Chat. Incluye modo terminal y una interfaz gráfica local sencilla con Gradio.

Requisitos rápidos
------------------
- Python 3.11+ y `pip` o `uv`.
- Dependencias en `pyproject.toml` (incluye `gradio`, `torch`, `transformers`, `peft`, etc.). Instala con `pip install -e .` o `uv sync`.
- Los pesos del adaptador LoRA ya están en `modelo_finetuned_local/` (no requiere descarga adicional).

Uso en terminal (CLI)
---------------------
```bash
uv run python inference.py --prompt "¿Qué equipo necesito para el circuito W en marzo?" --max_new_tokens 200
```
Luego queda en modo interactivo; escribe `exit`/`salir` para terminar.

Interfaz HTML liviana
---------------------
```bash
uv run python web_server.py --port 8000
```
Luego abre `http://127.0.0.1:8000` y usa la página HTML con estilo personalizado. El endpoint `POST /api/chat` recibe `{message, history, max_new_tokens, temperature, top_p}`.

Notas
-----
- Detecta GPU (`cuda`) o `mps` automáticamente; en CPU funciona pero será más lento.
- Si el modelo no tiene certeza, debería advertirlo y sugerir consultar fuentes oficiales/guardaparques.
