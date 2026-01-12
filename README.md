# RAG System

System RAG (Retrieval-Augmented Generation) do analizy dokumentów przetargowych (PDF, DOCX) z OCR.

## Wymagania systemowe (macOS)

```bash
brew install poppler tesseract tesseract-lang
```

## Instalacja zależności Python

```bash
pip install -r requirements.txt
```

## Zmienne środowiskowe

Dla wersji OpenAI (`rag_openai.py`):
```bash
export OPENAI_API_KEY="sk-..."
```

## Struktura projektu

```
/data/       # Wstaw tutaj dokumenty (PDF, DOCX)
/storage/    # Automatycznie generowany indeks wektorowy
```

## Użycie

### Wersja lokalna (Ollama)
```bash
python rag_local.py
```

### Wersja OpenAI (hybrid search)
```bash
python rag_openai.py
```

### Starter (prosty przykład)
```bash
python starter.py
```

## Jak to działa

1. Dokumenty z folderu `data/` są przetwarzane przez OCR (PDF) lub czytnik DOCX
2. Teksty są konwertowane na wektory i zapisywane w `storage/`
3. Przy kolejnym uruchomieniu indeks jest ładowany z `storage/` (bez ponownego OCR)
4. Agent przeszukuje dokumenty i odpowiada na pytania
