from typing import List, Tuple

import gradio as gr

from inference import generate_text, load_model_and_tokenizer

# Cargar modelo y tokenizer una sola vez al iniciar la app
MODEL, TOKENIZER, DEVICE = load_model_and_tokenizer()

SYSTEM_MESSAGE = (
    "Eres PAINE, un guía local que ayuda a mochileros a planificar y resolver "
    "dudas sobre el trekking en Torres del Paine. Responde con claridad, "
    "seguridad y consejos prácticos. Si algo no lo sabes con certeza, di que no "
    "estás seguro e invita a consultar a guardaparques u oficinas oficiales."
)


def _history_to_messages(
    history: List[Tuple[str, str]], user_message: str
) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for user, bot in history:
        if user:
            messages.append({"role": "user", "content": user})
        if bot:
            messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_message})
    return messages


def paine_reply(
    message: str,
    history: List[Tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = _history_to_messages(history, message)
    try:
        return generate_text(
            model=MODEL,
            tokenizer=TOKENIZER,
            device=DEVICE,
            messages=messages,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )
    except Exception as exc:
        return f"Ocurrió un error durante la generación: {exc}"


def main():
    chatbot = gr.Chatbot(
        height=520,
        show_label=False,
        avatar_images=(None, None),
    )

    interface = gr.ChatInterface(
        fn=paine_reply,
        title="PAINE - Guía de Trekking",
        description=(
            "Asistente local (offline) para dudas sobre trekking en Torres del Paine. "
            "Incluye recomendaciones de seguridad y logística."
        ),
        chatbot=chatbot,
        examples=[
            ["¿Cómo organizo el circuito W en 5 días si parto en Paine Grande?", 220, 0.7, 0.9],
            ["¿Qué equipo mínimo necesito en marzo? Llevo carpa y saco de dormir.", 220, 0.7, 0.9],
            ["¿Dónde puedo reabastecer agua y qué hago en caso de emergencia?", 220, 0.7, 0.9],
            ["¿Necesito reservar campamentos con cuánta anticipación en temporada alta?", 220, 0.7, 0.9],
        ],
        additional_inputs=[
            gr.Slider(
                minimum=64,
                maximum=512,
                value=220,
                step=8,
                label="Máx. tokens nuevos",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.05,
                label="Temperatura",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p",
            ),
        ],
        textbox=gr.Textbox(
            placeholder="Pregunta algo sobre rutas, equipo, clima, reservas o seguridad...",
            label="",
        ),
        cache_examples=False,
    )

    interface.launch()


if __name__ == "__main__":
    main()
