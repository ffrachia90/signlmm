<style>
    @page {
        size: A4;
        margin: 2.5cm;
    }
    body {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 12pt;
        line-height: 1.6;
        text-align: justify;
    }
    h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 25px; border-bottom: 1px solid #ddd; }
    h3 { color: #516173; margin-top: 20px; }
    code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 4px; font-family: "Courier New", monospace; font-size: 0.9em; }
    pre { background-color: #f8f9fa; padding: 15px; border: 1px solid #e1e4e8; border-radius: 6px; overflow-x: auto; }
</style>

# MEMORÁNDUM TÉCNICO

**PARA:** Fabián Curzio & Equipo Directivo, ELdeS  
**DE:** Equipo de Ingeniería & IA  
**FECHA:** 29 de Diciembre, 2025  
**ASUNTO:** Estrategia Técnica para el Desarrollo del Primer Large Multimodal Model (LMM) de Lengua de Señas

---

## 1. RESUMEN EJECUTIVO

El presente documento detalla la viabilidad técnica y hoja de ruta para el desarrollo de un modelo de Inteligencia Artificial propietario capaz de interpretar Lengua de Señas (LSA, LSE, LSU) a partir de video en tiempo real.

A diferencia de los enfoques tradicionales de IA generativa basados en texto (LLMs), este proyecto propone la creación de un **Modelo Multimodal (LMM)**. Este sistema no solo procesará información textual, sino que tendrá la capacidad de "ver" y comprender visualmente la ejecución de señas por parte de los usuarios.

La ventaja competitiva de ELdeS reside en su acceso exclusivo a una base de usuarios activa. Mientras que los gigantes tecnológicos carecen de datos específicos de dominio, ELdeS posee la capacidad de recolectar el dataset de video más grande del mundo en este nicho, lo cual garantiza una barrera de entrada casi insuperable para competidores futuros.

---

## 2. ANÁLISIS DE FACTIBILIDAD Y TECNOLOGÍA

### 2.1 Viabilidad Técnica
Tras un análisis exhaustivo de los recursos actuales, determinamos que el proyecto tiene una **factibilidad técnica ALTA**. Las tecnologías fundamentales requeridas para este desarrollo (Visión por Computadora y Redes Neuronales Secuenciales) han alcanzado un punto de madurez que permite su implementación sin necesidad de investigación científica base, reduciendo significativamente los riesgos del proyecto.

### 2.2 Arquitectura Propuesta: ¿Por qué un LMM?
Es crucial distinguir entre dos tipos de tecnologías para alinear las expectativas:
*   **LLM (Large Language Model):** Modelos como GPT-4 operan exclusivamente con texto. Son incapaces de procesar la entrada nativa de la comunidad sorda (video).
*   **LMM (Large Multimodal Model):** Es la arquitectura seleccionada para este proyecto. Permite inyectar secuencias de video directamente en el modelo, el cual aprende a asociar patrones de movimiento (manos, cuerpo, gestos faciales) con significados semánticos profundos.

---

## 3. PROTOCOLO DE ADQUISICIÓN DE DATOS

Para entrenar este modelo, es imperativo activar un protocolo de recolección de datos aprovechando el uso natural de la plataforma ELdeS. No se requiere etiquetado manual costoso; el sistema aprovechará el "aprendizaje autodefinido" de las lecciones actuales.

A continuación, se define el estándar técnico para el almacenamiento de cada intento de práctica de los usuarios:

**Requisito de Datos por Instancia:**
Cada vez que un usuario realiza un ejercicio en la plataforma, el sistema deberá almacenar:
1.  **El Video Crudo:** Formato MP4/WebM, resolución mínima de 720p a 30fps.
2.  **Metadata Contextual (Etiquetado Automático):** Un archivo JSON asociado que contenga la "Verdad Terreno" (Ground Truth) derivada de la lección activa.

**Estructura de Metadata Requerida:**
```json
{
  "archivo_video": "LSU_HOLA_user882_20251229.mp4",
  "etiquetado": {
    "seña_objetivo": "HOLA",         // Dato crítico: Qué seña se solicitó
    "idioma": "LSU",                 // Lengua de Señas Uruguaya/Argentina/Española
    "resultado_evaluador": true,     // Si el sistema actual lo marcó como correcto
    "nivel_confianza": 0.95          // Score numérico del evaluador actual
  },
  "contexto_tecnico": {
    "user_id_hash": "a1b2c3d4",      // ID anonimizado
    "timestamp": "2025-12-29T15:30:00Z",
    "dispositivo": "webcam_720p"
  }
}
```

---

## 4. PLAN DE TRABAJO: PRUEBA DE CONCEPTO (POC)

Se propone un plan de ejecución acelerado de **4 semanas** para validar la tecnología de extremo a extremo con una inversión mínima de recursos.

### Semana 1: Ingesta y "Semilla de Datos"
El objetivo es recolectar un dataset inicial de 1,000 videos cubriendo 50 señas fundamentales.
*   *Acción:* Instrumentar la plataforma web para guardar silenciosamente los videos exitosos de usuarios seleccionados (o realizar pruebas internas).
*   *Resultado:* Un repositorio de datos crudos listo para procesamiento.

### Semana 2: Procesamiento y Vectorización
Transformaremos los videos pesados en estructuras de datos optimizadas para IA.
*   *Acción:* Implementar pipeline de extracción de "Landmarks" (puntos clave) utilizando MediaPipe Holistic. Esto anonimiza visualmente al usuario y reduce el tamaño de los datos en un 99%.
*   *Resultado:* Dataset procesado y normalizado.

### Semana 3: Entrenamiento del Modelo Púrpura
Desarrollo y entrenamiento del primer motor de inteligencia artificial.
*   *Acción:* Entrenar una red neuronal (LSTM Bidireccional) para clasificar las 50 señas basándose puramente en el movimiento.
*   *Resultado:* Archivo de modelo (`.pt`) capaz de distinguir señas con una precisión estimada >80%.

### Semana 4: Integración y Demostración
Validación final ante stakeholders.
*   *Acción:* Desplegar una interfaz web interna donde se pueda subir un video nunca antes visto y observar cómo el sistema lo traduce a texto en tiempo real.
*   *Resultado:* POC funcional aprobada.

---

## 5. RECURSOS Y PRESUPUESTO POC

La ejecución de esta fase inicial requiere recursos contenidos:
*   **Capital Humano:** 1 Ingeniero de ML/Backend (Part-time o Full-time).
*   **Infraestructura Cloud:**
    *   Almacenamiento (S3/GCS): ~100GB.
    *   Cómputo (GPU): Instancia t4/a10g (Costo aprox. < $200 USD/mes).

## 6. CONCLUSIÓN

La tecnología necesaria para construir el **Traductor Universal de Lengua de Señas** ya existe. La única pieza faltante eran los datos, y ELdeS es la única entidad posicionada para obtenerlos. Ejecutar este plan no solo es técnicamente viable, sino que representa la oportunidad estratégica más relevante de la compañía para consolidarse como líder global en tecnología de accesibilidad.

---
**Generado por:** Equipo de Desarrollo & IA  
**Estado:** Listo para ejecución inmediata
