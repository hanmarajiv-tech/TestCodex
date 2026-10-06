import json
from typing import Any

from redis import Redis
from redis.exceptions import ResponseError

from .interfaces import IQueue, QueueMessage


class RedisQueue(IQueue):
    """Streams with consumer-group recovery and atomic result + acknowledgement."""

    def __init__(self, client: Redis, consumer: str, stream: str = "rag:jobs",
                 group: str = "rag:workers", reclaim_ms: int = 300_000) -> None:
        self.client = client
        self.consumer = consumer
        self.stream = stream
        self.group = group
        self.reclaim_ms = reclaim_ms
        self.claim_cursor = "0-0"
        try:
            client.xgroup_create(stream, group, id="0-0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    def read(self, block_ms: int = 1000) -> QueueMessage | None:
        claimed = self.client.xautoclaim(self.stream, self.group, self.consumer,
                                        self.reclaim_ms, self.claim_cursor, count=1)
        self.claim_cursor = claimed[0]
        messages = claimed[1]
        if not messages:
            streams = self.client.xreadgroup(self.group, self.consumer,
                                            {self.stream: ">"}, count=1, block=block_ms)
            messages = streams[0][1] if streams else []
        if not messages:
            return None
        message_id, fields = messages[0]
        return QueueMessage(message_id, fields.get("payload", ""))

    def complete(self, message: QueueMessage, result: dict[str, Any]) -> None:
        # Use stream ID rather than untrusted job_id to address results.
        with self.client.pipeline(transaction=True) as transaction:
            transaction.set(f"{self.stream}:result:{message.message_id}",
                            json.dumps(result, ensure_ascii=False), ex=86400)
            transaction.xack(self.stream, self.group, message.message_id)
            transaction.execute()
