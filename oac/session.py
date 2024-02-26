from abc import ABC, abstractmethod
import argparse
from datetime import datetime
from typing import Literal, TypedDict, Optional, Annotated, Any, Generator, TextIO
from dataclasses import dataclass, asdict, fields as get_fields
from pathlib import Path
from contextlib import contextmanager
import yaml
import logging
from textwrap import dedent
from copy import copy

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

LOG = logging.getLogger("oac.session")


MesgSourceType = Literal["generated", "provided"]


@dataclass
class Mesg:
    role: str
    content: str
    source: MesgSourceType = "provided"

    def __post_init__(self):
        self.content = dedent(self.content)


@dataclass
class ChatOption():

    @classmethod
    def from_namespace(cls, arg_ns: argparse.Namespace):
        options = {f.name: getattr(arg_ns, f.name, f.default) for f in get_fields(cls)}
        return cls(**options)

    frequency_penalty: Optional[Annotated[float, range(-2, 2)]] = None
    logit_bias: Optional[dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    stop: str | None = None
    temperature: Optional[float] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    user: str | None = None
    model: str = "gpt-3.5-turbo"


class SessionTrace:
    def __init__(self, project: str, session_id: str, options: ChatOption, log_dir: Path):
        self.project = project
        self.session_id = session_id
        self.dir = log_dir / project / session_id
        LOG.info("Writing trace logs to '%s'", str(self.dir))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index = 0
        self.history = []
        self.write_options(options)

    def __len__(self) -> int:
        return self._index

    def write_options(self, options: ChatOption):
        path = self.dir / f"{self._index:03}.options.yaml"
        path.write_text(yaml.safe_dump(asdict(options)))

    @contextmanager
    def stream_mesg(self, role: str, source: MesgSourceType = "generated"):
        content = ""
        filename = self._get_log_name(role, source)
        file = open(filename, "w")
        class _stream_writer:
            @staticmethod
            def write(data: str):
                nonlocal content
                content += data
                file.write(data)
        yield _stream_writer
        file.close()
        LOG.info("Message saved: %s", filename)
        self.history.append(Mesg(role, content, source))

    def write_mesg(self, mesg: Mesg):
        self.history.append(mesg)
        filename = self._get_log_name(mesg.role, mesg.source)
        filename.write_text(mesg.content)
        LOG.info("Message saved: %s", filename)


    def _get_log_name(self, role: str, source: MesgSourceType):
        self._index += 1  # if we increment the index we have to write it
        gen_tag = "_g" if source == "generated" else ""
        filename = self.dir / f"{self._index:03}_{role}{gen_tag}.md"
        return filename


class ChatSession:
    def __init__(self,
                 options: ChatOption,
                 trace: SessionTrace,
                 openai_client: Any = None,
                 output_stream = None):
        self.options = options
        self._option_stack = []
        self.trace = trace
        self.openai = openai_client or OpenAI()
        self.output_stream = output_stream

    def __enter__(self):
        self._option_stack.append(copy(self.options))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            self.run_prompt()
        self.options = self._option_stack.pop()
        return False

    def filter_messages(self, role) -> Generator[tuple[int, str], None, None]:
        for index, message in enumerate(self.trace.history):
            if message.role == role:
                yield (index, message.content)

    def run_prompt(self, prompt: str | None = None):
        if prompt is not None:
            self.user = prompt

        assert self.options is not None

        resp = self.openai.chat.completions.create(
            messages=self._make_openai_messages(),
            stream=True,
            **asdict(self.options)
        )

        with self.trace.stream_mesg("assistant", "generated") as out:
            for chunk in resp:
                content = chunk.choices[0].delta.content
                if content is not None:
                    out.write(content)
                    if self.output_stream is not None:
                        self.output_stream.write(content)

                if chunk.choices[0].finish_reason is not None:
                    print("\nGot finish reason:", chunk.choices[0].finish_reason)

        self.trace.write_options(self.options)

    def _make_openai_messages(self) -> Generator[ChatCompletionMessageParam, None, None]:
        for mesg in self.trace.history:
            yield {
                "role": mesg.role,
                "content": mesg.content,
            }

    @property
    def system(self) -> list[tuple[int, str]]:
        return list(self.filter_messages("system"))

    @system.setter
    def system(self, content: str):
        self.trace.write_mesg(Mesg("system", content))

    @property
    def assistant(self) -> list[tuple[int, str]]:
        return list(self.filter_messages("assistant"))

    @assistant.setter
    def assistant(self, content: str):
        self.trace.write_mesg(Mesg("assistant", content))

    @property
    def user(self) -> list[tuple[int, str]]:
        return list(self.filter_messages("user"))

    @user.setter
    def user(self, content: str):
        self.trace.write_mesg(Mesg("user", content))
    def last(self):
        try:
            return self.trace.history[-1].content
        except IndexError:
            return None
