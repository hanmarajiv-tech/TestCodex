import logging
from threading import Event

from .interfaces import IPipeline, IQueue
from .models import Job

logger = logging.getLogger(__name__)


class App:
    def __init__(self, queue: IQueue, pipeline: IPipeline) -> None:
        self.queue = queue
        self.pipeline = pipeline

    def process_next(self, block_ms: int = 1000) -> bool:
        message = self.queue.read(block_ms)
        if message is None:
            return False
        job = None
        try:
            job = Job.model_validate_json(message.payload)
            if job.operation == "ingest":
                data = {"documents_loaded": self.pipeline.ingest(job.file_path)}
            else:
                data = {"hits": [hit.model_dump() for hit in
                                 self.pipeline.search(job.query, job.top_k)]}
            result = {"status": "ok", "job_id": job.job_id, **data}
        except Exception as error:
            logger.exception("Job failed: %s", message.message_id)
            result = {"status": "error", "job_id": job.job_id if job else None,
                      "error_type": type(error).__name__, "error": str(error)}
        # Redis failures here propagate: do not acknowledge a lost result.
        self.queue.complete(message, result)
        return True

    def run(self, stop_event: Event | None = None) -> None:
        stop_event = stop_event if stop_event is not None else Event()
        while not stop_event.is_set():
            self.process_next()
