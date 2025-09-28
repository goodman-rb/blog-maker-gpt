# --- Импорт необходимых библиотек ---

import os
import openai
import currentsapi
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import httpx # Убедитесь, что httpx импортирован

# --- Инициализация и конфигурация ---

# Загружаем переменные окружения (API ключи) из файла .env
load_dotenv()

# Получаем ключи из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# Проверяем, что ключи действительно загружены. Если нет, приложение не запустится.
if not OPENAI_API_KEY:
    raise ValueError("Не найден ключ OPENAI_API_KEY. Убедитесь, что он есть в файле .env")
if not CURRENTS_API_KEY:
    raise ValueError("Не найден ключ CURRENTS_API_KEY. Убедитесь, что он есть в файле .env")

# Инициализируем асинхронный клиент OpenAI
# Использование async клиента предпочтительнее в асинхронных фреймворках, как FastAPI
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# Инициализируем клиент Currents API
currents_client = currentsapi.CurrentsApi(api_key=CURRENTS_API_KEY)

# Создаем экземпляр FastAPI приложения
app = FastAPI(
    title="API для генерации блог-постов",
    description="Это API использует Currents для получения новостей и OpenAI для создания статей на их основе.",
    version="1.0.0"
)


# --- Модели данных (Pydantic) ---

# Модель для валидации входящего запроса. 
# FastAPI будет автоматически проверять, что в теле запроса есть поле "topic".
class PostRequest(BaseModel):
    topic: str = Field(
        ..., 
        min_length=3, 
        max_length=100,
        description="Тема для генерации блог-поста, например, 'искусственный интеллект в медицине'."
    )

# Модель для ответа, чтобы структура ответа была стандартизирована.
class PostResponse(BaseModel):
    title: str
    meta_description: str
    post_content: str
    news_context_used: list[str]


# --- Вспомогательные функции ---

async def get_latest_news(topic: str) -> str:
    """
    Получает последние новости по заданной теме напрямую через Currents API,
    используя httpx. Форматирует результат в удобную строку для OpenAI.
    """
    # URL для запроса к API
    api_url = "https://api.currentsapi.services/v1/search"
    
    # Параметры запроса
    params = {
        "keywords": topic,
        "language": "ru",
        "apiKey": CURRENTS_API_KEY, # Ключ из .env
        "limit": 5
    }

    try:
        # Используем асинхронный клиент httpx, что идеально для FastAPI
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, params=params)
            
            # Вызовет ошибку, если API вернет статус 4xx или 5xx
            response.raise_for_status() 
            
            news_data = response.json()

        # Проверяем, нашлись ли новости
        if not news_data or not news_data.get('news'):
            return "Не удалось найти актуальные новости по данной теме."

        # Форматируем новости в пронумерованный список
        news_list = [f"{i+1}. {article['title']}: {article.get('description', 'Нет описания.')}" 
                     for i, article in enumerate(news_data['news'])]
        
        return "\n".join(news_list)
        
    except httpx.HTTPStatusError as e:
        # Ошибка ответа от API (например, 401 - неверный ключ, 429 - много запросов)
        print(f"Ошибка HTTP при запросе к Currents API: {e.response.status_code} - {e.response.text}")
        return f"API новостей вернуло ошибку {e.response.status_code}."
    except Exception as e:
        # Любая другая ошибка (проблемы с сетью, некорректный JSON и т.д.)
        print(f"Ошибка при запросе к Currents API: {e}")
        return "Произошла ошибка при получении новостей."

async def generate_blog_content(topic: str, news_context: str) -> dict:
    """
    Генерирует заголовок, мета-описание и текст поста с использованием OpenAI API,
    принимая в качестве контекста актуальные новости.
    """
    try:
        # --- 1. Генерация привлекательного заголовка ---
        prompt_title = f"Придумай один привлекательный и SEO-оптимизированный заголовок для поста в блог на тему: '{topic}'."
        response_title = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_title}],
            max_tokens=30,
            n=1,
            temperature=0.7,
        )
        title = response_title.choices[0].message.content.strip().strip('"')

        # --- 2. Генерация мета-описания на основе заголовка ---
        prompt_meta = f"Напиши краткое, но информативное мета-описание (150-160 символов) для поста с заголовком: '{title}'."
        response_meta = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_meta}],
            max_tokens=60,
            n=1,
            temperature=0.7,
        )
        meta_description = response_meta.choices[0].message.content.strip()

        # --- 3. Генерация основного контента поста с использованием новостного контекста ---
        prompt_post = (
            f"Напиши подробный, увлекательный и структурированный пост для блога на тему: '{topic}' с заголовком '{title}'.\n\n"
            f"**Обязательно используй следующие актуальные новости как контекст и основу для написания статьи. Ссылайся на них, развивай их идеи и анализируй их:**\n"
            f"--- НОВОСТНОЙ КОНТЕКСТ ---\n{news_context}\n--- КОНЕЦ КОНТЕКСТА ---\n\n"
            "Структура поста должна включать введение, несколько подзаголовков (используй Markdown для форматирования: ## Подзаголовок), короткие абзацы, примеры и заключение. "
            "Сделай текст живым и интересным для читателя."
        )
        response_post = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_post}],
            max_tokens=2048,
            n=1,
            temperature=0.7,
        )
        post_content = response_post.choices[0].message.content.strip()

        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }

    except openai.APIError as e:
        # Обработка ошибок, специфичных для OpenAI API
        print(f"Произошла ошибка OpenAI API: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Сервис OpenAI временно недоступен: {e}"
        )
    except Exception as e:
        # Обработка всех остальных непредвиденных ошибок
        print(f"Произошла непредвиденная ошибка при генерации контента: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера при генерации контента: {e}"
        )


# --- Эндпоинты API ---

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """
    Эндпоинт для проверки работоспособности сервиса.
    Возвращает статус 'ok', если приложение запущено.
    """
    return {"status": "ok"}

@app.post("/generate-post", response_model=PostResponse, tags=["Blog Generation"])
async def create_blog_post(request: PostRequest):
    """
    Основной эндпоинт для генерации блог-поста.
    
    Принимает тему, получает по ней новости, генерирует контент и возвращает его.
    """
    # Шаг 1: Получаем новости по теме из запроса
    news_context = await get_latest_news(request.topic)
    
    # Шаг 2: Генерируем контент, передавая тему и новостной контекст
    generated_content = await generate_blog_content(request.topic, news_context)

    # Шаг 3: Формируем и возвращаем финальный ответ
    return PostResponse(
        title=generated_content["title"],
        meta_description=generated_content["meta_description"],
        post_content=generated_content["post_content"],
        news_context_used=news_context.split('\n') # Возвращаем контекст в виде списка для наглядности
    )


# --- Запуск приложения ---

# Этот блок позволяет запускать приложение напрямую через 'python main.py'
# Но для продакшена рекомендуется использовать команду 'uvicorn main:app --host 0.0.0.0'
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
