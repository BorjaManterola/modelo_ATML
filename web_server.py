"""Servidor ligero con página HTML para conversar con PAINE."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List

from inference import generate_text, load_model_and_tokenizer

STATIC_DIR = Path(__file__).parent / "static"

SYSTEM_MESSAGE = (
    "Eres PAINE, un guía local que ayuda a mochileros a planificar y resolver "
    "dudas sobre el trekking en Torres del Paine. Responde con claridad, "
    "seguridad y consejos prácticos. Si algo no lo sabes con certeza, di que no "
    "estás seguro e invita a consultar a guardaparques u oficinas oficiales."
)

# Cargar modelo y tokenizer una sola vez al iniciar el servidor
MODEL, TOKENIZER, DEVICE = load_model_and_tokenizer()


def _history_to_messages(history: List[Dict[str, str]], user_message: str) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for turn in history:
        user = turn.get("user")
        bot = turn.get("bot")
        if user:
            messages.append({"role": "user", "content": user})
        if bot:
            messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_message})
    return messages


class PaineHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Logs estándar (status, path) permanecen, sin ruido adicional
        return super().log_message(format, *args)

    def _json_response(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self.send_error(404, "Endpoint no encontrado")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body or "{}")
        except Exception as exc:
            self._json_response({"error": f"JSON inválido: {exc}"}, status=400)
            return

        message = (payload.get("message") or "").strip()
        if not message:
            self._json_response({"error": "El mensaje no puede estar vacío."}, status=400)
            return

        history = payload.get("history") or []
        if not isinstance(history, list):
            history = []

        max_new_tokens = int(payload.get("max_new_tokens", 220))
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.9))

        messages = _history_to_messages(history, message)
        try:
            reply = generate_text(
                model=MODEL,
                tokenizer=TOKENIZER,
                device=DEVICE,
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:  # pragma: no cover - solo registros en runtime
            self._json_response(
                {"error": f"Ocurrió un error durante la generación: {exc}"},
                status=500,
            )
            return

        self._json_response({"reply": reply})


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    if not STATIC_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio estático en {STATIC_DIR}")

    server = ThreadingHTTPServer((host, port), PaineHandler)

    print(
        f"\nServidor listo: http://{host}:{port}\n"
        "Abre la página en tu navegador y conversa con PAINE.\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando servidor...")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Servidor HTML liviano para PAINE")
    parser.add_argument("--host", default="127.0.0.1", help="Host a enlazar (por defecto 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto HTTP (por defecto 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
