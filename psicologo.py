import streamlit as st
import os
import pickle
from crewai import Agent, LLM
from crewai_tools import ScrapeWebsiteTool
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import re

#Ejemplo de uso de Transformers para resumen 
#from transformers import pipeline
#
#summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
#
#def resumir_texto(texto, max_length=150, min_length=30):
#    try:
#        resumen = summarizer(texto, max_length=max_length, min_length=min_length, do_sample=False)
#        return resumen[0]['summary_text']
#    except Exception as e:
#        return texto[:max_length]


MEMORY_FILE = "chat_memory.pkl"

def cargar_memoria():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return []

def guardar_memoria(chat_history):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(chat_history, f)

@st.cache_data(ttl=3600)
def obtener_contexto(urls):
    textos = []
    for url in urls:
        tool = ScrapeWebsiteTool(url)
        contenido = tool.run()
        textos.append(f"Fuente: {url}\n{contenido}")
    return "\n\n".join(textos)

urls_contexto = [
    "https://www.mentalhealth.gov/",
    "https://www.mscbs.gob.es/",
    "https://www.isciii.es/"      
]
texto_contexto = obtener_contexto(urls_contexto)

LOCALE = "España"

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = cargar_memoria()

def _init_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = cargar_memoria()
    if "psy_progress" not in st.session_state:
        st.session_state["psy_progress"] = 0
_init_state()

with st.sidebar:
    st.markdown("## Línea de atención a la conducta suicida: 024")
    st.warning("⚠️ Esta herramienta no sustituye la atención profesional, en caso de crisis o urgencia, comunícate con los servicios de emergencia locales.")

    st.sidebar.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Limpiar chat")
        if st.button(" 🗑️"):
            st.session_state["chat_history"] = []
            guardar_memoria(st.session_state["chat_history"])
            st.success("Chat limpiado.")
    with col2:
        st.markdown("### Reiniciar diagnóstico")
        if st.button(" 🔄"):
            st.session_state["psy_progress"] = 0
            st.success("Diagnóstico reiniciado.")

    st.sidebar.markdown("---")
    model_options = {
        "groq/gemma2-9b-it": "Modelo ligero para conversaciones rápidas",
        "llama3-8b-8192": "Modelo avanzado para análisis complejos",
        "llama-3.3-70b-versatile": "Modelo versatil multilingüe"
    }
    selected_model = st.selectbox("Selecciona un modelo", list(model_options.keys()))
    st.caption(model_options[selected_model])

    st.sidebar.markdown("---")
    prog = st.session_state["psy_progress"]
    st.progress(prog)
    st.markdown(f"**Seguridad del diagnóstico:** {prog}%")
    
    st.sidebar.markdown("---")
    rating = st.slider("Valora nuestro servicio", 1, 5, 3)
    feedback = st.text_area("¿Cómo podemos mejorar?", max_chars=250)
    if st.button("Enviar Comentarios"):
        if feedback.strip():
            with open("feedback_logs.txt", "a") as f:
                f.write(f"Valoración: {rating}\nComentarios: {feedback.strip()}\n\n")
            st.success("¡Gracias por tus comentarios!")
        else:
            st.error("Por favor, ingresa un comentario antes de enviar.")

header_color = "#2a4d69"
text_color = "#4a4a4a"
st.markdown(f"""
    <style>
    .header-title {{
        color: {header_color};
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }}
    .main .block-container p {{
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 class='header-title'>Consulta Virtual Psicológica</h1>", unsafe_allow_html=True)

header_img_path = "/Users/nicolas/crewai/Psiconline.webp"
if os.path.exists(header_img_path):
    st.image(header_img_path, use_container_width=True)

@st.cache_resource
def init_llm_and_agents(model_name):
    llm = LLM(
        model=model_name,
        temperature=0.5,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("CREW_API_KEY")
    )
    psicologo = Agent(
        role="Psicólogo Virtual",
        goal=(
            "Proporcionar un espacio de apoyo emocional mediante escucha activa y empatía."
            "Explorar las preocupaciones del usuario con preguntas abiertas y adaptativas."
            "Validar y reformular emociones para facilitar la autoexploración."
            "Preparar información relevante para el psiquiatra."
        ),
        backstory=(
            "Tras años de práctica en entornos clínicos y comunitarios, desarrollaste una profunda comprensión de la importancia de la empatía"
            "y la escucha activa. Tu formación en terapia humanista y enfoque centrado en la persona te ha permitido acompañar a individuos en momentos críticos,"
            "ayudándoles a explorar sus emociones y pensamientos de manera segura. Estas experiencias han cimentado tu compromiso con el apoyo emocional"
            "y la creación de espacios de confianza."
        ),
        allow_delegation=False,
        verbose=True,
        tools=[],
        llm=llm
    )
    psiquiatra = Agent(
        role="Psiquiatra Virtual",
        goal=(
            "Analizar de forma experta el historial emocional y cognitivo del paciente."
            "Aplicar criterios diagnósticos clínicos (ICD-11)."
            "Con datos suficientes emitir un diagnóstico estructurado que guíe el tratamiento."
        ),
        backstory=(
            "Con una sólida formación en psicofarmacología y diagnóstico clínico según la ICD-11, has trabajado en diversos entornos hospitalarios y ambulatorios."
            "Tu experiencia te ha enseñado la importancia de un análisis riguroso y basado en evidencia, especialmente en casos complejos. "
            "Esta trayectoria te ha llevado a adoptar un enfoque meticuloso y reservado, "
            "interviniendo únicamente cuando la información disponible es suficiente para emitir un diagnóstico confiable."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    recomendador = Agent(
        role="Recomendador de Tratamientos",
        goal=(
            "Ofrecer intervenciones terapéuticas y recursos de apoyo efectivos basados en el diagnóstico psiquiátrico."
            "Priorizar terapias con evidencia clínica sólida y adaptabilidad a las necesidades del paciente."
        ),
        backstory=(
            "A lo largo de tu carrera, has integrado conocimientos de terapias basadas en evidencia como la TCC, Mindfulness y EMDR, "
            "adaptándolos a las necesidades individuales de cada paciente. Has colaborado con equipos multidisciplinarios, "
            "diseñando planes de tratamiento personalizados que combinan intervenciones terapéuticas y recursos psicoeducativos. "
            "Tu enfoque se centra en ofrecer recomendaciones claras y accionables, fundamentadas en la evidencia clínica y la comprensión profunda de cada caso."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    return llm, psicologo, psiquiatra, recomendador

llm, psicologo, psiquiatra, recomendador = init_llm_and_agents(selected_model)

def construir_prompt_psicologo(historial, mensaje_usuario, instr_psiquiatra=None):
    bloques = [
        f"Contexto adicional:\n{texto_contexto}",
        f"Contexto geográfico: {LOCALE}. Trabajas en {LOCALE}.",
        "Historial de la conversación:\n" + "\n".join(msg.content for msg in historial),
        f"Mensaje actual del usuario:\n{mensaje_usuario}",
        "Instrucciones: Responde de forma empática y de apoyo."
    ]
    if instr_psiquiatra:
        bloques.append("Instrucciones del psiquiatra:\n" + instr_psiquiatra)

    prompt = "\n\n".join(bloques)
    return (
        f"Rol: Psicólogo Virtual\n"
        f"Objetivo: {psicologo.goal}\n"
        f"Historia: {psicologo.backstory}\n\n"
        + prompt
    )


def construir_prompt_psiquiatra(historial):
    header = (
        f"Rol: Psiquiatra Virtual\n"
        f"Objetivo: {psiquiatra.goal}\n"
        f"Historia: {psiquiatra.backstory}\n\n"
    )

    instructions = (
        "Instrucciones:\n"
        "- Emite un diagnóstico estructurado solo cuando tu nivel de confianza sea >= 95%.\n"
        "- Incluye siempre 'Nivel de confianza: X%' con un valor numérico entre 0 y 100.\n"
        "- Secciones:\n"
        "  1. Resumen ejecutivo\n"
        "  2. Criterios diagnósticos\n"
        "  3. Evidencia del historial\n"
        "  4. Nivel de confianza: X%\n"
    )

    body = (
        "Historial completo de la conversación:\n"
        + "\n".join(f"- {msg.content}" for msg in historial)
        + "\n\nPor favor, atiende las instrucciones anteriores."
    )

    return header + instructions + body


def construir_prompt_recomendador(diagnostico):
    prompt = (
        f"Diagnóstico interno:\n{diagnostico}\n\n"
        f"Contexto geográfico: {LOCALE}. Trabajas en {LOCALE}.\n\n"
        "Instrucciones: A partir de este diagnóstico, genera recomendaciones de tratamiento altamente efectivas y adaptadas al paciente. "
        "Justifica cada recomendación en una frase y formatea la salida en viñetas"
    )
    return (
        f"Rol: Recomendador de Tratamientos\n"
        f"Objetivo: {recomendador.goal}\n"
        f"Historia: {recomendador.backstory}\n\n"
        + prompt
    )

async def call_llm_async(prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.call, [{"role": "system", "content": prompt}])

async def coordinar_interaccion_async(historial, mensaje_usuario):
    prompt_psiquiatra = construir_prompt_psiquiatra(historial)
    respuesta_psiquiatra = await call_llm_async(prompt_psiquiatra)
    historial.append(AIMessage(content=f"**Psiquiatra Virtual:** {respuesta_psiquiatra}"))

    match = re.search(r"Nivel de confianza:\s*(\d{1,3})%", respuesta_psiquiatra)
    nivel = int(match.group(1)) if match else 0
    st.session_state["psy_progress"] = nivel

    respuesta_recomendador = None
    umbral = 95

    if nivel >= umbral:
        prompt_rec = construir_prompt_recomendador(respuesta_psiquiatra)
        respuesta_recomendador = await call_llm_async(prompt_rec)
        historial.append(AIMessage(content=f"**Recomendador Virtual:** {respuesta_recomendador}"))
        return None, respuesta_psiquiatra, respuesta_recomendador

    instr = None
    if "Preguntas para el psicólogo:" in respuesta_psiquiatra:
        instr = respuesta_psiquiatra.split("Preguntas para el psicólogo:")[1].strip()
    prompt_psicologo = construir_prompt_psicologo(historial, mensaje_usuario, instr_psiquiatra=instr)
    respuesta_psicologo = await call_llm_async(prompt_psicologo)
    historial.append(AIMessage(content=f"**Psicólogo Virtual:** {respuesta_psicologo}"))

    return respuesta_psicologo, respuesta_psiquiatra, None


def display_history(history):
    umbral = 95
    nivel = st.session_state.get("psy_progress", 0)

    for msg in history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            content = msg.content

            if content.startswith("**Psiquiatra Virtual:**"):
                if nivel >= umbral:
                    with st.chat_message("assistant", avatar="/Users/nicolas/crewai/avatar2.png"):
                        st.markdown(content)
                continue

            if content.startswith("**Recomendador Virtual:**") and nivel >= umbral:
                with st.chat_message("assistant", avatar="/Users/nicolas/crewai/avatar3.png"):
                    st.markdown(content)
                continue

            with st.chat_message("assistant", avatar="/Users/nicolas/crewai/avatar.jpg"):
                st.markdown(content)


# Render inicial
display_history(st.session_state["chat_history"])

user_input = st.chat_input("Escribe tu mensaje aquí...")
if user_input:
    st.session_state["chat_history"].append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Procesando la consulta..."):
        respuesta_psicologo, respuesta_psiquiatra, respuesta_recomendador = asyncio.run(
            coordinar_interaccion_async(st.session_state["chat_history"], user_input)
        )

    nivel = st.session_state.get("psy_progress", 0)
    umbral = 95

    if nivel >= umbral:
        with st.chat_message("assistant", avatar="/Users/nicolas/crewai/avatar2.png"):
            st.markdown(f"**Psiquiatra Virtual:** {respuesta_psiquiatra}")
        if respuesta_recomendador:
            with st.chat_message("assistant", avatar="/Users/nicolas/crewai/avatar3.png"):
                st.markdown(f"**Recomendador Virtual:** {respuesta_recomendador}")
    else:
        with st.chat_message("assistant", avatar="/Users/nicolas/crewai/avatar.jpg"):
            st.markdown(f"**Psicólogo Virtual:** {respuesta_psicologo}")

    

    guardar_memoria(st.session_state["chat_history"])
    st.rerun()

