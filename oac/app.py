import sys
from typing import Any, Callable
import threading
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
from datetime import datetime
from dataclasses import fields as get_dataclass_fields

from .session import ChatSession, ChatOption, SessionTrace

LOG = logging.getLogger("oac.app")

MainFunc = Callable[[ChatSession], None]


DEFAULT_CONFIG_DIR = Path("~/.config/oac/").expanduser()
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / "oaa.yaml"
OPENAI_OPTIONS = DEFAULT_CONFIG_DIR / "openai-chat-options.yaml"
DEFAULT_LOG_DIR = Path("~/.cache/oac/").expanduser()


class ChatApp:
    def __init__(self,
                 client: Any = None,
                 cli_args: list[str] | None = None,
                 trace_dir: Path | None = None,
                 trace_name: str | None = None):
        self.client = client
        self.trace_dir = trace_dir
        self.trace_name = trace_name
        self.cli_args = cli_args or sys.argv[1:]
        self.namespace = Namespace()
        self.setup_logging()


    def setup_logging(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


    def get_options(self) -> ChatOption:
        return ChatOption.from_namespace(self.namespace)

    def create_session(self):
        options = self.get_options()
        trace = SessionTrace(
            session_id=self.trace_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            options=options,
            log_dir=self.trace_dir or DEFAULT_LOG_DIR,
        )
        return ChatSession(
            options=options,
            trace=trace,
            openai_client=self.client
        )

    def start(self, main_func: MainFunc, block: bool = True) -> threading.Thread:
        thread = threading.Thread(target=main_func, args=[self.create_session()])
        thread.start()
        if block:
            thread.join()
        return thread

    def parse_args(self):
        parser = self.get_arg_parser()
        self.namespace = parser.parse_args(self.cli_args, self.namespace)
        return self.namespace

    def get_arg_parser() -> ArgumentParser:
        parser = ArgumentParser(description='')

        for field in get_dataclass_fields(ChatOption):
            parser.add_argument(
                f"--{field.name.replace('_', '-')}",
                default=field.default,
                dest=field.name,
                #type=field.type
            )

        return parser