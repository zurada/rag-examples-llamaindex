# brew install poppler
# brew install tesseract
# brew install tesseract-lang
# pip install llama-index-llms-openai rank_bm25 llama-index-retrievers-bm25

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import DocxReader
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- NOWE IMPORTY DLA HYBRID SEARCH ---
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

import asyncio
import pytesseract
from pdf2image import convert_from_path
import os
import logging
import sys
import time
import psutil

# Włącz szczegółowe logowanie
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 1. KONFIGURACJA LOKALNEGO OCR DLA PDF ---
from concurrent.futures import ThreadPoolExecutor

def process_single_page(page_data):
    """Przetwarza jedną stronę PDF (uruchamiane równolegle)"""
    page_num, image = page_data
    custom_config = r'--oem 1 --psm 3'
    page_text = pytesseract.image_to_string(image, lang='pol+eng', config=custom_config)
    return page_num, page_text

class LocalOCRReader(BaseReader):
    def load_data(self, file_path, extra_info=None):
        print(f"🔄 OCR PDF: {os.path.basename(file_path)}...")
        text = ""
        try:
            # DPI=150, 8 wątków dla M4 Pro
            images = convert_from_path(file_path, dpi=150, thread_count=8)
            print(f"   📄 Stron do OCR: {len(images)}")

            # 8 workerów dla OCR
            with ThreadPoolExecutor(max_workers=8) as executor:
                page_data = list(enumerate(images, 1))
                results = list(executor.map(process_single_page, page_data))

            results.sort(key=lambda x: x[0])
            for page_num, page_text in results:
                text += f"\n--- Strona {page_num} ---\n{page_text}"

            print(f"✅ Zakończono OCR: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Błąd OCR: {e}")
            return []
        return [Document(text=text, extra_info=extra_info or {})]

# --- 2. DEFINICJA OBSŁUGI PLIKÓW ---
file_extractor = {
    ".pdf": LocalOCRReader(),
    ".docx": DocxReader()
}

# --- 3. USTAWIENIA LLM (OpenAI) ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# OpenAI LLM - MUSISZ USTAWIĆ KLUCZ API:
# export OPENAI_API_KEY="sk-..."
Settings.llm = OpenAI(
    model="gpt-5.2",  # Zmieniono na gpt-4o (gpt-5.2 nie istnieje)
    temperature=0.1,
    max_tokens=4000,
)

print(f"🤖 Używam modelu: {Settings.llm.model}")

# --- 4. ŁADOWANIE LUB TWORZENIE INDEKSU ---
PERSIST_DIR = "./storage"

if os.path.exists(PERSIST_DIR):
    print("💾 Ładowanie zapisanego indeksu...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("✅ Załadowano indeks z dysku!")
else:
    print("📂 Skanowanie katalogu 'data' i podfolderów...")

    # Sprawdź liczbę rdzeni CPU
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"💻 Wykryto {cpu_count} rdzeni CPU")

    reader = SimpleDirectoryReader(
        "data",
        file_extractor=file_extractor,
        recursive=True
    )

    # Równoległe przetwarzanie plików
    from threading import Lock
    documents = []
    files = reader.input_files
    processed_count = [0]
    lock = Lock()

    def process_file(file_path):
        """Przetwarza pojedynczy plik"""
        print(f"▶️  Start: {os.path.basename(file_path)}")
        file_reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor
        )
        docs = file_reader.load_data()
        with lock:
            processed_count[0] += 1
            pct = (processed_count[0] / len(files)) * 100
            print(f"✅ [{processed_count[0]}/{len(files)} - {pct:.1f}%] Zakończono: {os.path.basename(file_path)}")
            return docs

    print("⚡ Przetwarzam pliki równolegle...\n")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_file, files)
        for docs in results:
            documents.extend(docs)

    print(f"\n📚 Załadowano łącznie {len(documents)} fragmentów dokumentów.")

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"💾 Zapisano indeks do {PERSIST_DIR}")

# --- 5. KONFIGURACJA HYBRID SEARCH (ZMIANA GŁÓWNA) ---
print("⚙️ Konfiguracja Hybrid Search (Vector + BM25)...")

# 1. Retriever Wektorowy (Semantyczny - rozumie znaczenie)
vector_retriever = index.as_retriever(similarity_top_k=20)

# 2. Retriever Słów Kluczowych (BM25 - precyzyjny dla nazw własnych i liczb)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, 
    similarity_top_k=20
)

# 3. Połączenie (Fusion) - Reciprocal Rank Fusion
retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=25,  # Zwróć top 15 najlepszych fragmentów z obu metod
    num_queries=1,        # Nie generuj dodatkowych pytań (oszczędność czasu/tokenów)
    mode="reciprocal_rerank",  # Poprawiona nazwa trybu
)

# 4. Budowa silnika zapytań
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=Settings.llm,
    response_mode="tree_summarize" # Tryb, który lepiej składa informacje z wielu kawałków
)

# --- 6. AGENT I NARZĘDZIA ---
def multiply(a: float, b: float) -> float:
    """Mnoży dwie liczby."""
    return a * b

def search_documents(query: str) -> str:
    """
    Wyszukuje informacje w dokumentach. Używa wyszukiwania hybrydowego (słowa kluczowe + kontekst).
    Użyj tego narzędzia gdy użytkownik pyta o zawartość plików, PDFów, SWZ lub Worda.
    """
    response = query_engine.query(query)
    return str(response)

# Konwertuj funkcje na FunctionTool
search_tool = FunctionTool.from_defaults(fn=search_documents)

# Callback handler do monitorowania
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

class VerboseCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.llm_call_count = 0
        self.start_time = time.time()
        self.token_count = 0

    def on_event_start(self, event_type, payload=None, event_id=None, **kwargs):
        if event_type == CBEventType.LLM:
            self.llm_call_count += 1
            elapsed = time.time() - self.start_time
            print(f"\n🤖 [Wywołanie OpenAI #{self.llm_call_count}] (czas: {elapsed:.1f}s)")
            if payload and EventPayload.MESSAGES in payload:
                messages = payload[EventPayload.MESSAGES]
                if messages:
                    last_msg = str(messages[-1])[:200]
                    print(f"   💬 {last_msg}...")

    def on_event_end(self, event_type, payload=None, event_id=None, **kwargs):
        if event_type == CBEventType.LLM:
            print(f"   ✅ Odpowiedź otrzymana")
            if payload and hasattr(payload.get(EventPayload.RESPONSE, None), 'raw'):
                raw = payload[EventPayload.RESPONSE].raw
                if hasattr(raw, 'usage') and raw.usage is not None:
                    usage = raw.usage
                    print(f"   📊 Tokeny: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")
                    self.token_count += usage.total_tokens

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass

verbose_handler = VerboseCallbackHandler()
callback_manager = CallbackManager([verbose_handler])
Settings.callback_manager = callback_manager

# Utwórz agenta
agent = ReActAgent(
    tools=[search_tool],
    llm=Settings.llm,
)

async def main():
    ctx = Context(agent)

    print("\n" + "="*60)
    print("PYTANIE:")
    print("="*60)
    question = """
Jesteś Bezwzględnym Audytorem Dokumentacji Przetargowej.
Twoim celem jest stworzenie "Checklisty Twardych Wymagań" (Must-Have) na podstawie analizy dokumentów (SWZ, OPZ, Wzór Umowy).

ZADANIE DLA AGENTA:
1. Przeszukaj dokumenty używając narzędzia `search_documents`.
   SZUKAJ SŁÓW KLUCZOWYCH SUGERUJĄCYCH WYMÓG:
   - "wymóg", "wymaga się", "wymagane"
   - "musi", "należy", "wykonawca jest zobowiązany"
   - "kryterium dopuszczające", "warunek graniczny"
   - oraz standardowe: "wadium", "kary", "doświadczenie", "ubezpieczenie", "gwarancja".

2. Ignoruj "lanie wody". Szukaj konkretów: kwot, dat, procentów, liczby osób, lat doświadczenia.

3. SFORMATUJ WYNIK JAKO JSON (DYNAMICZNA STRUKTURA):
   - Nie używaj sztywnych nazw kategorii ani numeracji (np. "1_terminy").
   - Stwórz kategorie na podstawie tego, co faktycznie znajdziesz w dokumencie (np. "UBEZPIECZENIE_OC", "KARY_UMOWNE", "KIEROWNIK_BUDOWY").
   - Jeśli dokument milczy na dany temat, NIE twórz pustej kategorii.

FORMAT WYJŚCIOWY (JSON):
{
  "meta_info": {
    "znalezione_dokumenty": ["lista plików"],
    "nazwa_postepowania": "..."
  },
  "WYKRYTE_WYMAGANIA": {
    "NAZWA_KATEGORII_WIELKIMI_LITERAMI (np. WADIUM)": [
      {
        "wymog": "Krótki opis czego dotyczy (np. 'Kwota wadium')",
        "szczegoly_wartosc": "Konkretna wartość (np. '50 000 PLN', '5 lat doświadczenia', 'Gwarancja bankowa')",
        "status": "WYMAGANE / OPCJONALNE / BRAK DANYCH",
        "zrodlo": "Nazwa pliku i przybliżona lokalizacja (np. Rozdział 4, pkt 2)"
      },
      {
        "wymog": "Forma wniesienia",
        "szczegoly_wartosc": "Pieniądz, gwarancja bankowa lub ubezpieczeniowa",
        "status": "WYMAGANE",
        "zrodlo": "..."
      }
    ],
    "NAZWA_INNEJ_ZNALEZIONEJ_KATEGORII (np. KARY_UMOWNE)": [
      {
        "wymog": "Limit kar umownych",
        "szczegoly_wartosc": "20% wartości umowy brutto",
        "status": "WYMAGANE",
        "zrodlo": "Wzór Umowy §15"
      }
    ]
  },
  "UWAGI_KRYTYCZNE": [
    "Tutaj wpisz ostrzeżenia, jeśli brakuje kluczowych elementów (np. brak informacji o terminie realizacji mimo znalezienia SWZ)."
  ]
}
"""
    print(question)
    print("="*60)

    # Monitor wydajności
    process = psutil.Process()
    ram_before = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    verbose_handler.start_time = start_time

    print(f"\n📊 RAM przed: {ram_before:.1f} MB")
    print("⏳ Rozpoczynam zapytanie do OpenAI...\n")

    result = await agent.run(ctx=ctx, user_msg=question)

    end_time = time.time()
    ram_after = process.memory_info().rss / 1024 / 1024
    duration = end_time - start_time

    print("\n" + "="*60)
    print("ODPOWIEDŹ AGENTA:")
    print("="*60)
    print(result)
    print("="*60)

    print("\n" + "="*60)
    print("📊 METRYKI WYDAJNOŚCI:")
    print(f"⏱️  Czas: {duration:.1f}s")
    print(f"🔢 Wywołania OpenAI: {verbose_handler.llm_call_count}")
    print(f"🎯 Tokeny: {verbose_handler.token_count}")
    estimated_cost = (verbose_handler.token_count / 1_000_000) * 2.50 # Przybliżony koszt mieszany gpt-4o
    print(f"💵 Szacunkowy koszt: ~${estimated_cost:.4f}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())