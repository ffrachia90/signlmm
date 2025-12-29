import gradio as gr
import os

def echo(video):
    if video is None:
        return "No hay video"
    return "Video recibido correctamente (Modo Simple)"

# Forzar configuración de red para evitar el error de localhost
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

print("🚀 Iniciando Demo Simplificada...")

with gr.Blocks() as demo:
    gr.Markdown("# 🤟 SignLMM - Demo Simple")
    
    with gr.Row():
        vid = gr.Video(label="Video")
        out = gr.Textbox(label="Resultado")
    
    btn = gr.Button("Procesar")
    btn.click(echo, inputs=vid, outputs=out)

# Lanzamiento ultra-robusto
try:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False 
    )
except Exception as e:
    print(f"❌ Error al lanzar: {e}")
    # Intento desesperado con localhost si 0.0.0.0 falla
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
