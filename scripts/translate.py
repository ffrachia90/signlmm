"""
SignLMM POC - Traducción con LLM
================================
Traduce glosas de lengua de señas a español natural usando LLM.

Uso:
    from translate import SignTranslator
    translator = SignTranslator()
    result = translator.translate(["YO", "IR", "CASA", "MAÑANA"])
    # → "Mañana voy a mi casa"
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class SignTranslator:
    """
    Traduce secuencias de glosas de lengua de señas a español natural.
    
    Soporta múltiples proveedores de LLM:
    - OpenAI (GPT-3.5, GPT-4)
    - Gemini (Google)
    - Local (Ollama)
    """
    
    SYSTEM_PROMPT = """Eres un traductor experto de Lengua de Señas a español.

Tu tarea es traducir GLOSAS (representación escrita de señas) a español natural y fluido.

IMPORTANTE sobre las glosas:
- Las glosas están en MAYÚSCULAS
- El orden gramatical es diferente al español (generalmente SOV: Sujeto-Objeto-Verbo)
- No tienen artículos ni preposiciones
- Los verbos no están conjugados
- Las preguntas se indican con la palabra interrogativa al final o al principio

Ejemplos de traducciones:
- "YO IR CASA" → "Voy a casa" o "Voy a mi casa"
- "TÚ NOMBRE QUÉ" → "¿Cómo te llamas?" o "¿Cuál es tu nombre?"
- "AYER YO COMER PIZZA" → "Ayer comí pizza"
- "MAÑANA YO TRABAJAR NO" → "Mañana no trabajo"
- "TÚ QUERER CAFÉ" → "¿Quieres café?"

Reglas:
1. Traduce a español natural y fluido
2. Mantén el significado original
3. Adapta el tiempo verbal según el contexto (AYER=pasado, MAÑANA=futuro, etc.)
4. Si hay signos de pregunta (QUÉ, QUIÉN, DÓNDE, etc.), formula como pregunta
5. Responde SOLO con la traducción, sin explicaciones"""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Inicializa el traductor.
        
        Args:
            provider: "openai", "gemini", o "local"
            model: Modelo específico (default según provider)
            api_key: API key (o usar variable de entorno)
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.model = model or "gpt-3.5-turbo"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self._init_openai()
        elif self.provider == "gemini":
            self.model = model or "gemini-pro"
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            self._init_gemini()
        elif self.provider == "local":
            self.model = model or "llama2"
            self._init_local()
        else:
            raise ValueError(f"Provider no soportado: {provider}")
    
    def _init_openai(self):
        """Inicializa cliente OpenAI."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Instala openai: pip install openai")
    
    def _init_gemini(self):
        """Inicializa cliente Gemini."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("Instala google-generativeai: pip install google-generativeai")
    
    def _init_local(self):
        """Inicializa cliente local (Ollama)."""
        try:
            import requests
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            # Verificar que Ollama está corriendo
            requests.get(f"{self.ollama_url}/api/tags", timeout=5)
        except Exception as e:
            print(f"Warning: No se pudo conectar a Ollama: {e}")
    
    def translate(
        self,
        glosas,
        sign_language: str = "LSA",
        context: Optional[str] = None
    ) -> str:
        """
        Traduce una secuencia de glosas a español.
        
        Args:
            glosas: Lista de glosas o string separado por espacios
            sign_language: Lengua de señas (LSA, LSE, LSU)
            context: Contexto adicional opcional
            
        Returns:
            Traducción en español
        """
        # Normalizar input
        if isinstance(glosas, list):
            glosas_str = " ".join(glosas)
        else:
            glosas_str = glosas
        
        # Construir prompt
        user_prompt = f"Traduce esta secuencia de glosas de {sign_language} a español:\n\n{glosas_str}"
        
        if context:
            user_prompt += f"\n\nContexto adicional: {context}"
        
        # Llamar al LLM según provider
        if self.provider == "openai":
            return self._translate_openai(user_prompt)
        elif self.provider == "gemini":
            return self._translate_gemini(user_prompt)
        elif self.provider == "local":
            return self._translate_local(user_prompt)
    
    def _translate_openai(self, user_prompt: str) -> str:
        """Traduce usando OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    def _translate_gemini(self, user_prompt: str) -> str:
        """Traduce usando Gemini."""
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
        response = self.client.generate_content(full_prompt)
        return response.text.strip()
    
    def _translate_local(self, user_prompt: str) -> str:
        """Traduce usando Ollama local."""
        import requests
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{self.SYSTEM_PROMPT}\n\n{user_prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise Exception(f"Error de Ollama: {response.text}")


class SimpleTranslator:
    """
    Traductor simple basado en reglas (sin LLM).
    Útil para testing o cuando no hay API disponible.
    """
    
    # Mapeo básico de glosas a español
    VOCABULARY = {
        # Saludos
        "HOLA": "hola",
        "CHAU": "chau",
        "BUENOS-DÍAS": "buenos días",
        "BUENAS-TARDES": "buenas tardes",
        "BUENAS-NOCHES": "buenas noches",
        "GRACIAS": "gracias",
        "POR-FAVOR": "por favor",
        "PERDÓN": "perdón",
        "DE-NADA": "de nada",
        
        # Pronombres
        "YO": "yo",
        "TÚ": "tú",
        "ÉL": "él",
        "ELLA": "ella",
        "NOSOTROS": "nosotros",
        "ELLOS": "ellos",
        
        # Verbos
        "SER": "soy/es",
        "ESTAR": "estoy/está",
        "TENER": "tengo/tiene",
        "QUERER": "quiero/quiere",
        "PODER": "puedo/puede",
        "IR": "voy/va",
        "VENIR": "vengo/viene",
        "COMER": "como/come",
        "BEBER": "bebo/bebe",
        "DORMIR": "duermo/duerme",
        "TRABAJAR": "trabajo/trabaja",
        "ESTUDIAR": "estudio/estudia",
        
        # Sustantivos
        "CASA": "casa",
        "FAMILIA": "familia",
        "TRABAJO": "trabajo",
        "ESCUELA": "escuela",
        "COMIDA": "comida",
        "AGUA": "agua",
        "PERSONA": "persona",
        "AMIGO": "amigo",
        
        # Preguntas
        "QUÉ": "qué",
        "QUIÉN": "quién",
        "DÓNDE": "dónde",
        "CUÁNDO": "cuándo",
        "CÓMO": "cómo",
        "POR-QUÉ": "por qué",
        
        # Tiempo
        "AYER": "ayer",
        "HOY": "hoy",
        "MAÑANA": "mañana",
        "AHORA": "ahora",
        
        # Otros
        "SÍ": "sí",
        "NO": "no",
        "BIEN": "bien",
        "MAL": "mal",
        "MÁS": "más",
        "MENOS": "menos",
    }
    
    def translate(self, glosas, **kwargs) -> str:
        """Traducción simple palabra por palabra."""
        if isinstance(glosas, str):
            glosas = glosas.split()
        
        translated = []
        for glosa in glosas:
            glosa_upper = glosa.upper()
            if glosa_upper in self.VOCABULARY:
                translated.append(self.VOCABULARY[glosa_upper])
            else:
                translated.append(glosa.lower())
        
        result = " ".join(translated)
        
        # Capitalizar primera letra
        if result:
            result = result[0].upper() + result[1:]
        
        return result


def get_translator(use_llm: bool = True, **kwargs):
    """
    Factory function para obtener el traductor apropiado.
    
    Args:
        use_llm: Si es True, intenta usar LLM; si falla, usa SimpleTranslator
        **kwargs: Argumentos para SignTranslator
    """
    if use_llm:
        try:
            return SignTranslator(**kwargs)
        except Exception as e:
            print(f"Warning: No se pudo inicializar LLM ({e}). Usando traductor simple.")
            return SimpleTranslator()
    return SimpleTranslator()


# Test
if __name__ == "__main__":
    # Probar traductor simple
    simple = SimpleTranslator()
    
    test_cases = [
        "HOLA",
        "YO IR CASA",
        "TÚ QUERER COMER QUÉ",
        "MAÑANA YO TRABAJAR",
        "GRACIAS AMIGO",
    ]
    
    print("=== Traductor Simple ===\n")
    for glosas in test_cases:
        result = simple.translate(glosas)
        print(f"  {glosas}")
        print(f"  → {result}\n")
    
    # Probar con LLM si hay API key
    if os.getenv("OPENAI_API_KEY"):
        print("\n=== Traductor LLM (OpenAI) ===\n")
        llm = SignTranslator(provider="openai")
        for glosas in test_cases:
            result = llm.translate(glosas)
            print(f"  {glosas}")
            print(f"  → {result}\n")


