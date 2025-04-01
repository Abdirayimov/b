import os
from openai import AsyncOpenAI
import pandas as pd
import json
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
import telegram.error
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

# Инициализация асинхронного клиента OpenAI
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Определение каталогов для файлов
FILES_DIR = "FILES"
DOCS_DIR = "DOCS"
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Класс для обработки DOC/PDF файлов
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def process_document(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла")
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    async def get_openai_response(self, prompt, chat_history):
        messages = [{"role": "system", "content": "Ты — помощник, который отвечает на вопросы на основе предоставленных документов. Используй только предоставленную информацию и отвечай в формате Markdown."}]
        messages.extend([{"role": "user" if msg.type == "human" else "assistant", "content": msg.content} for msg in chat_history])
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.9,
            max_tokens=5000
        )
        return response.choices[0].message.content

processor = DocumentProcessor()

# Функции для обработки Excel файлов
def find_header(df):
    for i in range(5):
        row = df.iloc[i]
        if any(keyword in str(row).lower() for keyword in ["kod", "номер", "код", "tavsif", "поставщик", "завод"]):
            return i
    return 0

def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        dataframes = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name, header=None)
            header_row = find_header(df)
            df = pd.read_excel(xls, sheet_name, header=header_row)
            df['Файл'] = file_path
            df['Лист'] = sheet_name
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Ошибка: {file_path} не загружено! {e}")
        return pd.DataFrame()

def clean_query(query):
    patterns = [
        r"Kod:\s*(\S+)", r"Terogo nomeri:\s*(\S+)", r"Maxsus kod:\s*(\S+)",
        r"код прв\s*(\S+)", r"\b([A-Za-z0-9\-]+)\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return query.split()[-1]

def search_data(query, df):
    query = clean_query(query)
    mask = df.applymap(str).stack().str.contains(query, case=False, na=False).unstack()
    result = df[mask.any(axis=1)]
    return result.head(5).to_string() if not result.empty else "Извините, по запросу ничего не найдено."

tools = [{
    "type": "function",
    "function": {
        "name": "search_data",
        "description": "Поиск информации в базе данных Excel.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

async def ask_gpt(question, df):
    messages = [
        {"role": "system", "content": """Ты — интеллектуальный помощник, который анализирует данные из загруженных Excel-таблиц.  
Твои основные задачи:

📊 **1️⃣ Поиск данных (показывай ВСЕ найденные совпадения)**  
— Отвечай на вопросы пользователей, находя **ВСЕ совпадения** в Excel-файлах.  
— Показывай **название файла** и **лист**, откуда получены данные.  
— Если данных нет, честно сообщай об этом, **не придумывай ответы**.  

📈 **2️⃣ Статистический анализ**  
— Если вопрос связан с подсчетами (сумма, среднее, процент, динамика и т. д.), анализируй данные в таблицах и делай расчет.  
— Используй математические формулы при необходимости.  

🎯 **3️⃣ Формат ответа (используй Markdown)**  
— Если найдено несколько значений, **показывай их ВСЕ в удобном виде (списком или таблицей)**.  
— Если найденные данные относятся к разным файлам, указывай их отдельно.  
— Если информации нет, сообщи: **"🚫 По вашему запросу данных не найдено."**  

⚠ **Важно:**  
- Ты **не должен** придумывать информацию или использовать внешние источники.  
- **Если данных много, выводи их в виде таблицы.**
"""},
        {"role": "user", "content": question}
    ]
    completion = await client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
    response_message = completion.choices[0].message
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        if tool_call.function.name == "search_data":
            query = json.loads(tool_call.function.arguments)["query"]
            search_result = search_data(query, df)
            messages.append(response_message)
            messages.append({"role": "tool", "content": search_result, "tool_call_id": tool_call.id})
            final_completion = await client.chat.completions.create(model="gpt-4o", messages=messages)
            return final_completion.choices[0].message.content
    return response_message.content

# Функция для проверки и добавления существующих файлов при запуске бота
def load_existing_files(context: ContextTypes.DEFAULT_TYPE):
    # Загрузка Excel файлов
    excel_files = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(('.xlsx', '.xls'))]
    if excel_files:
        if 'excel_files' not in context.user_data:
            context.user_data['excel_files'] = []
        if 'excel_data' not in context.user_data:
            context.user_data['excel_data'] = {}
        for file_path in excel_files:
            if file_path not in context.user_data['excel_files']:
                context.user_data['excel_files'].append(file_path)
                df = load_data(file_path)
                context.user_data['excel_data'][file_path] = df

    # Загрузка DOC/PDF файлов
    doc_files = [os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.docx', '.doc'))]
    if doc_files:
        if 'doc_files' not in context.user_data:
            context.user_data['doc_files'] = []
        if 'doc_indices' not in context.user_data:
            context.user_data['doc_indices'] = {}
        for file_path in doc_files:
            if file_path not in context.user_data['doc_files']:
                try:
                    chunks = processor.process_document(file_path)
                    db = FAISS.from_documents(chunks, processor.embeddings)
                    context.user_data['doc_files'].append(file_path)
                    context.user_data['doc_indices'][file_path] = db
                except Exception as e:
                    print(f"⚠️ Ошибка при загрузке файла {file_path}: {e}")

# Стандартная клавиатура
def get_default_keyboard():
    return [[{"text": "📤 Загрузить файл"}, {"text": "🔍 Поиск в Excel"}, {"text": "📄 Поиск в DOC/PDF"}]]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Загрузка существующих файлов при запуске
    load_existing_files(context)
    await update.message.reply_text(
        "Привет! Я помогу с данными в Excel, DOC и PDF. Выберите действие:",
        reply_markup={"keyboard": get_default_keyboard(), "resize_keyboard": True, "one_time_keyboard": True}
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message.text:
            user_message = update.message.text

            if 'selected_excel_file' in context.user_data:
                selected_file = context.user_data.pop('selected_excel_file')
                df = context.user_data['excel_data'][selected_file]
                temp = await update.message.reply_text('⌛')
                response = await ask_gpt(user_message, df)
                await temp.delete()
                await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)

            elif 'selected_doc_file' in context.user_data:
                selected_file = context.user_data.pop('selected_doc_file')
                db = context.user_data['doc_indices'][selected_file]
                retriever = db.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(user_message)
                context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
                prompt = (
                    "Ты — помощник, который отвечает на вопросы на основе предоставленных документов. "
                    "Используй только следующие фрагменты текста для ответа. "
                    "Если ответа нет в документах, сообщи об этом.\n\n"
                    "{context}\n\n"
                    "Вопрос: {question}\n\n"
                    "Ответ:"
                ).format(context=context_str, question=user_message)
                memory = context.user_data.get('memory', ConversationBufferMemory(memory_key="chat_history", return_messages=True))
                chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
                answer = await processor.get_openai_response(prompt, chat_history)
                memory.save_context({"question": user_message}, {"answer": answer})
                await update.message.reply_text(f"🔍 **Ответ:**\n{answer}", parse_mode="Markdown")

            elif user_message == "📤 Загрузить файл":
                await update.message.reply_text("Отправьте файл (Excel, DOC или PDF).")

            elif user_message == "🔍 Поиск в Excel":
                if 'excel_files' not in context.user_data or not context.user_data['excel_files']:
                    await update.message.reply_text("Excel файлы не загружены. Сначала загрузите файл.")
                else:
                    keyboard = [[InlineKeyboardButton(os.path.basename(file)[:50], callback_data=f"excel_file_{i}")] for i, file in enumerate(context.user_data['excel_files'])]
                    await update.message.reply_text("Выберите Excel файл:", reply_markup=InlineKeyboardMarkup(keyboard))

            elif user_message == "📄 Поиск в DOC/PDF":
                if 'doc_files' not in context.user_data or not context.user_data['doc_files']:
                    await update.message.reply_text("DOC/PDF файлы не загружены. Сначала загрузите файл.")
                else:
                    keyboard = [[InlineKeyboardButton(os.path.basename(file)[:50], callback_data=f"doc_file_{i}")] for i, file in enumerate(context.user_data['doc_files'])]
                    await update.message.reply_text("Выберите DOC/PDF файл:", reply_markup=InlineKeyboardMarkup(keyboard))

            else:
                await update.message.reply_text("Пожалуйста, сначала выберите тип поиска и файл.")

        elif update.message.document:
            file = update.message.document
            file_name = file.file_name
            # Проверка на дубликаты имен файлов
            base_name, ext = os.path.splitext(file_name)
            counter = 1
            new_file_name = file_name
            if file.mime_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                target_dir = FILES_DIR
            else:
                target_dir = DOCS_DIR
            while os.path.exists(os.path.join(target_dir, new_file_name)):
                new_file_name = f"{base_name}_{counter}{ext}"
                counter += 1
            file_path = os.path.join(target_dir, new_file_name)
            file_obj = await file.get_file()
            await file_obj.download_to_drive(file_path)

            if file.mime_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                if 'excel_files' not in context.user_data:
                    context.user_data['excel_files'] = []
                context.user_data['excel_files'].append(file_path)
                if 'excel_data' not in context.user_data:
                    context.user_data['excel_data'] = {}
                df = load_data(file_path)
                context.user_data['excel_data'][file_path] = df
                await update.message.reply_text(f"Excel файл '{new_file_name}' загружен!")
            elif file.mime_type in ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                chunks = processor.process_document(file_path)
                db = FAISS.from_documents(chunks, processor.embeddings)
                if 'doc_indices' not in context.user_data:
                    context.user_data['doc_indices'] = {}
                context.user_data['doc_indices'][file_path] = db
                if 'doc_files' not in context.user_data:
                    context.user_data['doc_files'] = []
                context.user_data['doc_files'].append(file_path)
                if 'memory' not in context.user_data:
                    context.user_data['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                await update.message.reply_text(f"DOC/PDF файл '{new_file_name}' загружен!")
            else:
                await update.message.reply_text("Загружайте только файлы в формате Excel, DOC или PDF.")
    except Exception as e:
        await update.message.reply_text(f"🚨 Ошибка: {str(e)}")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("excel_file_"):
        file_index = int(data.split("_")[2])
        context.user_data['selected_excel_file'] = context.user_data['excel_files'][file_index]
        selected_file = context.user_data['excel_files'][file_index]
        await query.edit_message_text(f"Выбран Excel файл: {os.path.basename(selected_file)[:50]}\nВведите запрос:")
    elif data.startswith("doc_file_"):
        file_index = int(data.split("_")[2])
        context.user_data['selected_doc_file'] = context.user_data['doc_files'][file_index]
        selected_file = context.user_data['doc_files'][file_index]
        await query.edit_message_text(f"Выбран DOC/PDF файл: {os.path.basename(selected_file)[:50]}\nВведите запрос:")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        raise context.error
    except telegram.error.TimedOut:
        await update.message.reply_text("🚫 Время ожидания истекло.")
    except Exception as e:
        await update.message.reply_text(f"🚨 Ошибка: {str(e)}")

def main():
    application = Application.builder().token(TELEGRAM_API_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_error_handler(error_handler)
    application.run_polling()

if __name__ == "__main__":
    main()