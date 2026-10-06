# Redis → LlamaIndex → Qdrant MVP (Python 3.12)

Программа принимает задания только через `App(queue, pipeline)`. Worker читает
Redis Streams, загружает каждый указанный файл в Qdrant или выполняет поиск.
Поиск возвращает фрагменты, оценки и метаданные, без генерации ответа LLM.

```text
RedisQueue → App → RAGPipeline
                    ├─ ILoader → IDocumentSchema → LlamaIndex → Qdrant
                    └─ IRetriever → IReranker → SearchHit
```

## Запуск

```bash
uv sync --frozen --extra dev
mkdir -p data
docker compose up -d
printf 'Qdrant is a vector database.\n' > data/example.txt
uv run --frozen rag-worker
```

FastEmbed при первом запуске скачивает модель. По умолчанию используется
англоязычная `BAAI/bge-small-en-v1.5`, без API-ключей. Для русского языка задайте
поддерживаемую FastEmbed многоязычную модель через `EMBEDDING_MODEL`. При смене
модели используйте новую коллекцию и переиндексируйте все документы: одинаковая
размерность векторов не означает совместимость моделей.

Например: `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
В ограниченной облачной сети разрешите `huggingface.co`, `*.huggingface.co`
и `*.hf.co` для скачивания модели. Значения переменных задаются в окружении
процесса; `.env` автоматически не загружается worker.

В другом терминале отправьте задания (клиент лишь записывает/читает очередь;
вся обработка проходит через `App`):

```python
import json
import time
from redis import Redis

redis = Redis.from_url('redis://localhost:6379/0', decode_responses=True)

def submit(payload):
    message_id = redis.xadd('rag:jobs', {'payload': json.dumps(payload)})
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = redis.get(f'rag:jobs:result:{message_id}')
        if result is not None:
            return json.loads(result)
        time.sleep(0.1)
    raise TimeoutError(message_id)

print(submit({'job_id': 'load-1', 'operation': 'ingest',
              'file_path': 'example.txt'}))
print(submit({'job_id': 'search-1', 'operation': 'search',
              'query': 'vector database', 'top_k': 5}))
```

Путь файла задаётся относительно `DATA_ROOT` и должен быть доступен worker.
`TextLoader` поддерживает UTF-8 `.txt` и `.md`, запрещает выход за корень,
включая выход через символические ссылки. Для PDF, DOCX и прочих форматов
реализуйте отдельный `ILoader`.

## Компоненты

`RAGPipeline` принимает индекс и реализации `ILoader`, `IDocumentSchema`,
`IRetriever`, `IReranker`. `App` принимает только очередь `IQueue` и единый
pipeline `IPipeline`; передавать ему отдельные компоненты не требуется.
Схемы данных описаны Pydantic-моделями в `models.py`.

Чтобы добавить реранкер, наследуйте `IReranker`, реализуйте
`rerank(query, nodes)` и передайте экземпляр в `RAGPipeline` вместо
`IdentityReranker`. Например, адаптер cross-encoder может пересчитать оценки
и вернуть фрагменты в новом порядке. Ретривер аналогично заменяется через
`IRetriever.retrieve(query, top_k)`. Текущий реранкер сохраняет порядок Qdrant.

## Настройки

| Переменная | Значение по умолчанию |
| --- | --- |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `QDRANT_URL` | `http://localhost:6333` |
| `QDRANT_API_KEY` | не задан |
| `QDRANT_COLLECTION` | `documents` |
| `DATA_ROOT` | `data` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `MODEL_CACHE` | `.cache/fastembed` |

## Гарантии и границы MVP

- Redis 6.2+; клиент очереди должен использовать `decode_responses=True`.
- Результат и ACK записываются одной Redis-транзакцией; результаты хранятся сутки.
- Ошибочные задания возвращают `status=error` и подтверждаются. Для повторной
  попытки отправьте новое задание. Некорректный JSON не останавливает worker.
- Задания worker, погибшего до ACK, переуступаются через пять минут.
  Это доставка как минимум один раз. Обработка файла должна укладываться в этот
  интервал; MVP рассчитан на один worker, без heartbeat для долгих заданий.
- Повторная загрузка того же абсолютного пути удаляет его старые фрагменты
  и вставляет новые, в том числе после перезапуска. Замена не атомарна:
  при ошибке между удалением и вставкой нужно повторить задание.
- Перенос файла меняет идентификатор документа. Удаление исходного файла
  автоматически не удаляет документ из Qdrant.
- Redis Stream не очищается автоматически; для длительной эксплуатации нужна
  политика очистки с учётом pending-заданий, ограничения размеров файлов,
  авторизация и мониторинг. Compose открывает сервисы только на loopback.

## Проверка

```bash
uv run --frozen pytest -q
```

Тесты используют настоящий локальный Qdrant client (`:memory:`), LlamaIndex,
`MockEmbedding` и fakeredis. Проверяют загрузку, поиск, замену после создания
нового индекса, ACK и результаты, восстановление pending, ошибки, замену
реранкера и защиту от выхода за `DATA_ROOT`. Они не проверяют качество
семантического ранжирования и доступность серверов Docker или загрузку модели.
