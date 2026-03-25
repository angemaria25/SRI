# PaperScan

Sistema de Recuperacion de Informacion orientado a investigacion cientifica y academica.

## Punto de entrada
La aplicacion inicia desde `main.py`.

## Estructura principal
- `main.py`: orquestacion del flujo principal.
- `modulos/`: modulos funcionales del sistema.
- `interfaz/`: espacio reservado para la interfaz visual.
- `datos/`: corpus, indices, vectores y resultados del sistema.
- `local/`: base documental en PDF (`local/local_papers`) y script de descarga.
- `documentacion/`: documentacion tecnica en formato LNCS.
- `Dockerfile` y `docker-compose.yml`: despliegue reproducible.

## Modulos del sistema
### Modulos imprescindibles implementados
- `modulos/adquisicion_datos`: construccion de corpus local desde PDF y adquisicion web desde arXiv API.
- `modulos/indexacion`: preprocesamiento e indice invertido.
- `modulos/recuperacion`: recuperador no basico (modelo probabilistico de lenguaje).
- `modulos/base_vectorial`: base vectorial inicial con TF-IDF.

### Soporte de respaldo web implementado
- `modulos/busqueda_web`: consulta en arXiv cuando los resultados locales son insuficientes.

## Flujo funcional implementado (Primera Entrega)
1. El sistema toma como base los PDF de `local/local_papers`.
2. Construye `datos/brutos/corpus_local.jsonl` con texto extraido de PDF.
3. Construye indice invertido en `datos/indices`.
4. Construye base vectorial inicial en `datos/base_vectorial`.
5. Ejecuta recuperacion local con modelo de lenguaje y similitud vectorial.
6. Guarda resultados y estadisticas en `datos/procesados`.
7. Si hay pocos resultados locales, activa respaldo web en arXiv y guarda `datos/brutos/corpus_web_arxiv.jsonl`.

### Modulos imprescindibles creados y pendientes
- `modulos/rag`
- `modulos/posicionamiento`

### Modulos opcionales creados y pendientes
- `modulos/expansion_retroalimentacion`
- `modulos/multimodal`
- `modulos/recomendacion`
- `modulos/evaluacion`

## Ejecucion local
Instalacion de dependencias:

```bash
pip install -r requirements.txt
```

Ejecucion del sistema:

```bash
python main.py --consulta "tu consulta" --max-local 3000 --max-web 30
```

Descarga local de documentos PDF:

```bash
python local/scrapping.py
```

## Ejecucion con Docker
Construccion de imagen:

```bash
docker build -t paperscan .
```

Ejecucion del sistema principal:

```bash
docker run --rm -v ${PWD}/datos:/app/datos -v ${PWD}/local:/app/local paperscan python main.py --consulta "tu consulta"
```

Ejecucion del descargador local:

```bash
docker run --rm -v ${PWD}/local:/app/local paperscan python local/scrapping.py
```

## Ejecucion con Docker Compose
Sistema principal:

```bash
docker compose up sistema_principal
```

Con parametros personalizados:

```bash
CONSULTA_USUARIO="busqueda de evidencia clinica" MAX_DOCUMENTOS_LOCALES=2000 MAX_RESULTADOS_WEB=20 docker compose up sistema_principal
```

Descargador local:

```bash
docker compose up descargador_local
```

## Referencia del modelo recuperador
- Croft, W. B., Metzler, D., Strohman, T. (2010). Search Engines: Information Retrieval in Practice. Addison-Wesley.

## Evidencias generadas por el sistema
- `datos/brutos/corpus_local.jsonl`
- `datos/indices/indice_invertido.json`
- `datos/base_vectorial/vectores.npy`
- `datos/procesados/estadisticas_corpus_local.json`
- `datos/procesados/resultados_locales_modelo_lenguaje.json`
- `datos/procesados/resultados_locales_vectorial.json`
- `datos/procesados/respuesta_sistema.json`
