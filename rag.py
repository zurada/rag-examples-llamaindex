# brew install poppler
# brew install tesseract
# brew install tesseract-lang

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import DocxReader  # <--- IMPORT DLA WORD
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Włącz logi dla LlamaIndex workflow
logging.getLogger("llama_index.core.agent").setLevel(logging.DEBUG)
logging.getLogger("llama_index.core.workflow").setLevel(logging.DEBUG)

# --- 1. KONFIGURACJA LOKALNEGO OCR DLA PDF (Twoja wersja na Mac M4) ---
from concurrent.futures import ThreadPoolExecutor  # Zmiana z ProcessPoolExecutor na ThreadPoolExecutor

def process_single_page(page_data):
    """Przetwarza jedną stronę PDF (uruchamiane równolegle)"""
    page_num, image = page_data
    # Szybszy config tesseracta: --psm 3 (auto), --oem 1 (LSTM)
    custom_config = r'--oem 1 --psm 3'
    page_text = pytesseract.image_to_string(image, lang='pol+eng', config=custom_config)
    return page_num, page_text

class LocalOCRReader(BaseReader):
    def load_data(self, file_path, extra_info=None):
        print(f"🔄 OCR PDF: {os.path.basename(file_path)}...")
        text = ""
        try:
            # DPI=150 zamiast domyślnych 200 = znacznie szybciej, wciąż czytelne
            # thread_count=8 dla convert_from_path (M4 Pro ma 14-16 rdzeni)
            images = convert_from_path(file_path, dpi=150, thread_count=8)

            print(f"   📄 Stron do OCR: {len(images)}")

            # M4 Pro ma 14-16 rdzeni - używamy 8 workerów dla OCR
            # ThreadPoolExecutor działa świetnie bo Tesseract zwalnia GIL
            with ThreadPoolExecutor(max_workers=8) as executor:
                page_data = list(enumerate(images, 1))
                results = list(executor.map(process_single_page, page_data))

            # Sortuj po numerze strony i złącz tekst
            results.sort(key=lambda x: x[0])
            for page_num, page_text in results:
                text += f"\n--- Strona {page_num} ---\n{page_text}"

            print(f"✅ Zakończono OCR: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Błąd OCR: {e}")
            return []
        return [Document(text=text, extra_info=extra_info or {})]

# --- 2. DEFINICJA OBSŁUGI PLIKÓW ---

# Tworzymy mapę: rozszerzenie -> odpowiedni czytnik
file_extractor = {
    ".pdf": LocalOCRReader(),  # Nasz własny OCR dla PDF
    ".docx": DocxReader()      # Wbudowany czytnik Worda (wymaga pip install docx2txt)
}

# --- 3. USTAWIENIA LLM (Ollama) ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Klasa Ollama ze streamingiem tokenów
from llama_index.core.llms import ChatResponse, CompletionResponse
from llama_index.core.base.llms.types import ChatMessage, MessageRole

class StreamingOllama(Ollama):
    """Ollama z wydrukowaniem każdego tokena w czasie rzeczywistym"""

    def chat(self, messages, **kwargs):
        """Override chat żeby drukować tokeny"""
        full_response = ""

        # Użyj stream_chat do otrzymywania tokenów
        for chunk in self.stream_chat(messages, **kwargs):
            token = chunk.delta
            if token:
                print(token, end="", flush=True)
                full_response += token

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=full_response),
            raw={"content": full_response}
        )

    def complete(self, prompt, **kwargs):
        """Override complete żeby drukować tokeny"""
        full_response = ""

        for chunk in self.stream_complete(prompt, **kwargs):
            token = chunk.delta
            if token:
                print(token, end="", flush=True)
                full_response += token

        return CompletionResponse(text=full_response, raw={"content": full_response})

# Utwórz streaming LLM
Settings.llm = StreamingOllama(
    #model="SpeakLeash/bielik-11b-v3.0-instruct:bf16",
    model="qwen3:30b",
    request_timeout=360.0,
    context_window=8000,
)

# --- 4. ŁADOWANIE LUB TWORZENIE INDEKSU ---
PERSIST_DIR = "./storage"

if os.path.exists(PERSIST_DIR):
    print("💾 Ładowanie zapisanego indeksu...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("✅ Załadowano indeks z dysku (bez OCR)!")
else:
    print("📂 Skanowanie katalogu 'data' i podfolderów...")

    # Sprawdź liczbę rdzeni CPU
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"💻 Wykryto {cpu_count} rdzeni CPU")

    # Najpierw policz wszystkie pliki
    import glob
    pdf_files = glob.glob("data/**/*.pdf", recursive=True)
    docx_files = glob.glob("data/**/*.docx", recursive=True)
    total_files = len(pdf_files) + len(docx_files)

    print(f"🔍 Znaleziono {total_files} plików ({len(pdf_files)} PDF, {len(docx_files)} DOCX)")
    print("📂 Rozpoczynam ładowanie równoległe...\n")

    # Ładuj z progress tracking
    reader = SimpleDirectoryReader(
        "data",
        file_extractor=file_extractor,
        recursive=True
    )

    # Równoległe przetwarzanie plików (M4 Pro ma dużo rdzeni)
    from threading import Lock
    documents = []
    files = reader.input_files
    processed_count = [0]  # Licznik w liście żeby móc modyfikować w threadach
    lock = Lock()

    def process_file(file_path):
        """Przetwarza pojedynczy plik"""
        print(f"▶️  Start: {os.path.basename(file_path)}")

        # Ładuj pojedynczy plik
        file_reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor
        )
        docs = file_reader.load_data()

        # Thread-safe update licznika i listy
        with lock:
            processed_count[0] += 1
            pct = (processed_count[0] / len(files)) * 100
            print(f"✅ [{processed_count[0]}/{len(files)} - {pct:.1f}%] Zakończono: {os.path.basename(file_path)}")
            return docs

    # Przetwarzaj 4 pliki równocześnie (zostaw rdzenie dla OCR wewnątrz każdego pliku)
    print("⚡ Przetwarzam pliki równolegle...\n")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_file, files)
        for docs in results:
            documents.extend(docs)

    print(f"\n📚 Załadowano łącznie {len(documents)} fragmentów dokumentów.")

    index = VectorStoreIndex.from_documents(documents)

    # Zapisz indeks do dysku
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"💾 Zapisano indeks do {PERSIST_DIR}")

query_engine = index.as_query_engine(
    similarity_top_k=30, 
    response_mode="tree_summarize" # Tryb, który lepiej składa informacje z wielu kawałków
)
# --- 5. AGENT ---
def multiply(a: float, b: float) -> float:
    """Mnoży dwie liczby."""
    return a * b

def search_documents(query: str) -> str:
    """
    Wyszukuje informacje w załadowanych dokumentach (PDF i DOCX).
    Użyj tego narzędzia gdy użytkownik pyta o zawartość plików, dokumentów, PDFów lub Worda.

    Args:
        query: Zapytanie o informacje z dokumentów

    Returns:
        Odpowiedź zawierająca informacje znalezione w dokumentach
    """
    response = query_engine.query(query)
    return str(response)

# Custom callback handler do monitorowania LLM
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

class VerboseCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.llm_call_count = 0
        self.start_time = time.time()
        self.current_response = ""

    def on_event_start(self, event_type, payload=None, event_id=None, **kwargs):
        if event_type == CBEventType.LLM:
            self.llm_call_count += 1
            elapsed = time.time() - self.start_time
            print(f"\n🤖 [Wywołanie LLM #{self.llm_call_count}] (czas: {elapsed:.1f}s)")
            if payload and EventPayload.MESSAGES in payload:
                messages = payload[EventPayload.MESSAGES]
                if messages:
                    last_msg = str(messages[-1])[:200]
                    print(f"   💬 {last_msg}...")
            print("   🔄 Odpowiedź: ", end="", flush=True)
            self.current_response = ""

    def on_event_end(self, event_type, payload=None, event_id=None, **kwargs):
        if event_type == CBEventType.LLM:
            if self.current_response == "" and payload and EventPayload.RESPONSE in payload:
                # Jeśli nie było streamingu, wydrukuj całą odpowiedź
                response = str(payload[EventPayload.RESPONSE])
                print(response)
            print(f"\n   ✅ Odpowiedź zakończona")

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass

# Konwertuj funkcje na FunctionTool
multiply_tool = FunctionTool.from_defaults(fn=multiply)
search_tool = FunctionTool.from_defaults(fn=search_documents)

# Utwórz callback handler
verbose_handler = VerboseCallbackHandler()
callback_manager = CallbackManager([verbose_handler])

# Dodaj callback manager do Settings globalnie
Settings.callback_manager = callback_manager

# Utwórz agenta zgodnie z nową dokumentacją
agent = ReActAgent(
    tools=[search_tool],
    llm=Settings.llm,
)

async def main():
    # Utwórz kontekst dla sesji
    ctx = Context(agent)

    print("\n" + "="*60)
    print("PYTANIE:")
    print("="*60)
    question = question = """
Jesteś Bezwzględnym Audytorem Dokumentacji Przetargowej.
Twoim zadaniem jest znalezienie i wylistowanie twardych wymagań (Must-Have), nawet jeśli są ukryte głęboko w dokumentacji.

ZADANIE DLA AGENTA (Krok po kroku):
1. Twoim priorytetem jest znalezienie głównego dokumentu SWZ (Specyfikacja Warunków Zamówienia) lub SIWZ, PFU (Program Funkcjonalno-Użytkowy) oraz OPZ (Opis Przedmiotu Zamówienia).
2. Użyj narzędzia `search_documents` wielokrotnie, wpisując precyzyjne frazy kluczowe typowe dla polskich przetargów.
   
   Sugerowane zapytania do wyszukiwarki (wykonaj je):
   - "Rozdział Warunki Udziału w Postępowaniu wykształcenie doświadczenie"
   - "Wymagane wadium i zabezpieczenie należytego wykonania"
   - "Kary umowne i terminy realizacji zamówienia"
   - "Wymagany potencjał kadrowy i osoby skierowane do realizacji"
   - "Środki finansowe lub zdolność kredytowa wykonawcy"

3. Ignoruj aneksy środowiskowe, decyzje administracyjne i ogólne warunki, chyba że zawierają konkretne liczby/wymogi.

FORMAT WYJŚCIOWY (JSON):
Zwróć wynik jako JSON. Jeśli nie znajdziesz informacji dla danej kategorii, wpisz "BRAK DANYCH W POBRANYCH FRAGMENTACH".

{
  "critical_requirements": [
    {
      "category": "KADRA / FINANSE / DOŚWIADCZENIE / FORMALNE",
      "source_context": "Z jakiego dokumentu/rozdziału to pochodzi?",
      "requirement_raw": "Cytat z dokumentu",
      "value_to_check": "Konkretna wartość (np. 'Polisa OC 5 mln PLN', 'Kierownik z uprawnieniami mostowymi')"
    }
  ]
}
"""
    print(question)
    print("="*60)

    # Monitor wydajności
    process = psutil.Process()
    ram_before = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    verbose_handler.start_time = start_time  # Reset czasu w handlerze

    print(f"\n📊 RAM przed: {ram_before:.1f} MB")
    print("⏳ Rozpoczynam zapytanie...\n")

    # Użyj run z kontekstem
    result = await agent.run(ctx=ctx, user_msg=question)

    # Statystyki
    end_time = time.time()
    ram_after = process.memory_info().rss / 1024 / 1024  # MB
    duration = end_time - start_time

    print("\n" + "="*60)
    print("ODPOWIEDŹ AGENTA:")
    print("="*60)
    print(result)
    print("="*60)

    # Pokaż metryki
    print("\n" + "="*60)
    print("📊 METRYKI WYDAJNOŚCI:")
    print("="*60)
    print(f"⏱️  Czas całkowity: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"💾 RAM przed: {ram_before:.1f} MB")
    print(f"💾 RAM po: {ram_after:.1f} MB")
    print(f"💾 Różnica RAM: +{ram_after - ram_before:.1f} MB")

    # Użyj licznika z callback handlera
    total_llm_calls = verbose_handler.llm_call_count
    print(f"🔢 Liczba wywołań LLM: {total_llm_calls}")
    if total_llm_calls > 0:
        avg_time_per_call = duration / total_llm_calls
        print(f"⚡ Średni czas na wywołanie: {avg_time_per_call:.1f}s")
        # Szacujemy ~20-30 tokens/s dla Bielik na M4
        estimated_tokens = int(duration * 25)  # przybliżona wartość
        print(f"🎯 Szacowane tokeny wygenerowane: ~{estimated_tokens}")
        print(f"🚀 Szacowana prędkość: ~{estimated_tokens/duration:.1f} tokens/s")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())