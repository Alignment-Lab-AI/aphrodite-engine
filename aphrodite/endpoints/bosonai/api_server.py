# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import importlib
import inspect
import json
import os
import re
import signal
import socket
import ssl
import uuid
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from threading import Thread
from typing import Optional, get_args
from loguru import logger
import uvloop
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import State
from starlette.routing import Mount

import aphrodite.common.envs as envs
from aphrodite.common.config import ModelConfig
from aphrodite.engine.args_tools import AsyncEngineArgs, nullable_str
from aphrodite.engine.protocol import EngineClient
from aphrodite.endpoints.bosonai.serving_audio import (HiggsAudioServingAudio,
                                                    load_voice_presets)
# yapf: enable
from aphrodite.endpoints.bosonai.serving_chat import HiggsAudioServingChat
from aphrodite.endpoints.chat_utils import (load_chat_template,
                                         resolve_hf_chat_template,
                                         resolve_mistral_chat_template)
from aphrodite.server.launch import serve_http
from aphrodite.endpoints.openai.args import (ChatTemplateContentFormatOption,
                                              LoRAParserAction,
                                              PromptAdapterParserAction,
                                              validate_parsed_serve_args)
# yapf conflicts with isort for this block
# yapf: disable
from aphrodite.endpoints.openai.protocol import (AudioSpeechRequest,
                                              ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              ErrorResponse)
from aphrodite.endpoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from aphrodite.endpoints.openai.tool_parsers import ToolParserManager
from aphrodite.endpoints.utils import (cli_env_setup, load_aware_call,
                                    with_cancellation)
from aphrodite.modeling.models.higgs_audio_tokenizer import AudioTokenizer
from aphrodite.reasoning import ReasoningParserManager
from aphrodite.transformers_utils.tokenizer import MistralTokenizer
from aphrodite.usage.usage_lib import UsageContext
from aphrodite.utils import (FlexibleArgumentParser, is_valid_ipv6_address,
                        set_ulimit)
from aphrodite.version import __version__ as aphrodite_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        gc.collect()
        gc.freeze()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type: Only 'application/json' is allowed"
        )


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import (CollectorRegistry, make_asgi_app,
                                   multiprocess)
    from prometheus_fastapi_instrumentator import Instrumentator

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.debug("aphrodite to use %s as PROMETHEUS_MULTIPROC_DIR",
                     prometheus_multiproc_dir_path)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        Instrumentator(
            excluded_handlers=[
                "/metrics",
                "/health",
                "/load",
                "/ping",
                "/version",
            ],
            registry=registry,
        ).add().instrument(app).expose(app)

        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def chat(request: Request) -> Optional[HiggsAudioServingChat]:
    return request.app.state.openai_serving_chat


def audio(request: Request) -> Optional[HiggsAudioServingAudio]:
    return request.app.state.openai_serving_audio


def voice_presets(request: Request) -> Optional[dict]:
    return request.app.state.voice_presets


@asynccontextmanager
async def build_async_engine_client(
        args: Namespace) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)

    os.environ["HIGGS_AUDIO_TOKENIZER"] = args.audio_tokenizer_type
    if args.audio_tokenizer_path is not None:
        os.environ["HIGGS_AUDIO_TOKENIZER_PATH"] = args.audio_tokenizer_path
    audio_tokenizer = AudioTokenizer(
        model=args.audio_tokenizer_path or args.audio_tokenizer_type,
        device="cuda",
    )

    # HACK: This is a hack to allow more than one audio per request.
    if not hasattr(engine_args, "limit_mm_per_prompt") or \
        engine_args.limit_mm_per_prompt is None:
        engine_args.limit_mm_per_prompt = {"audio": 50}

    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:
        yield engine, audio_tokenizer


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """
    # Create the EngineConfig (determines if we can use V1).
    usage_context = UsageContext.OPENAI_API_SERVER

    aphrodite_config = engine_args.create_engine_config(usage_context=usage_context)

    # V1 AsyncLLM.
    if envs.aphrodite_USE_V1:
        if disable_frontend_multiprocessing:
            logger.warning(
                "V1 is enabled, but got --disable-frontend-multiprocessing. "
                "To disable frontend multiprocessing, set aphrodite_USE_V1=0.")

        from aphrodite.v1.engine.async_llm import AsyncLLM
        async_llm: Optional[AsyncLLM] = None
        try:
            async_llm = AsyncLLM.from_aphrodite_config(
                aphrodite_config=aphrodite_config,
                usage_context=usage_context,
                disable_log_requests=engine_args.disable_log_requests,
                disable_log_stats=engine_args.disable_log_stats)
            yield async_llm
        finally:
            if async_llm:
                async_llm.shutdown()
    else:
        raise ValueError("Only V1 is supported")


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = ErrorResponse(message=str(exc),
                            type="BadRequestError",
                            code=HTTPStatus.BAD_REQUEST)
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    # Ensure --api-key option from CLI takes precedence over aphrodite_API_KEY
    if token := args.api_key or envs.aphrodite_API_KEY:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path
            if app.root_path and url_path.startswith(app.root_path):
                url_path = url_path[len(app.root_path):]
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    if args.enable_request_id_headers:
        logger.warning(
            "CAUTION: Enabling X-Request-Id headers in the API Server. "
            "This can harm performance at high QPS.")

        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = request.headers.get(
                "X-Request-Id") or uuid.uuid4().hex
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response

    if envs.aphrodite_DEBUG_LOG_API_SERVER_RESPONSE:
        logger.warning("CAUTION: Enabling log response in the API Server. "
                       "This can include sensitive information and should be "
                       "avoided in production.")

        @app.middleware("http")
        async def log_response(request: Request, call_next):
            response = await call_next(request)
            response_body = [
                section async for section in response.body_iterator
            ]
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            logger.info("response_body={%s}", response_body[0].decode())
            return response

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


async def init_app_state(
    engine_client: EngineClient,
    audio_tokenizer: AudioTokenizer,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = await engine_client.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(
                chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer,
                chat_template=None,
                tools=None,
                trust_remote_code=model_config.trust_remote_code)

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: %s\n"
                    "It is different from official chat template '%s'. "
                    "This discrepancy may lead to performance degradation.",
                    resolved_chat_template, args.model)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = HiggsAudioServingChat(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        audio_tokenizer=audio_tokenizer,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_audio = HiggsAudioServingAudio(
        engine_client,
        model_config,
        state.openai_serving_models,
        voice_presets_dir=args.voice_presets_dir,
        request_logger=request_logger,
        chat_template_content_format=args.chat_template_content_format,
        audio_tokenizer=audio_tokenizer,
    ) if model_config.runner_type == "generate" else None

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0

    state.voice_presets = {}
    update_voice_presets_thd = Thread(
        target=load_voice_presets,
        args=(state, args.voice_presets_dir,
              args.voice_presets_refresh_interval),
        daemon=True,
    )
    update_voice_presets_thd.start()


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("aphrodite API server version %s", aphrodite_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valid_tool_parses)} }})")

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/aphrodite-project/aphrodite/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as (engine_client,
                                                   audio_tokenizer):
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, audio_tokenizer, model_config,
                             app.state, args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return '[' + a + ']'
            return a or "0.0.0.0"

        is_ssl = args.ssl_keyfile and args.ssl_certfile
        logger.info("Starting aphrodite API server on http%s://%s:%d",
                    "s" if is_ssl else "", _listen_addr(sock_addr[0]),
                    sock_addr[1])

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()

def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client

@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)

@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": aphrodite_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions",
             dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    handler = chat(raw_request)

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/audio/speech",
             dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_audio_speech(request: AudioSpeechRequest,
                              raw_request: Request):
    handler = audio(raw_request)

    generator = await handler.create_audio_speech_stream(
        request,
        voice_presets=voice_presets(raw_request),
        raw_request=raw_request,
    )

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    return StreamingResponse(content=generator, media_type="audio/mpeg")


@router.get("/v1/audio/voices")
async def get_audio_voices(raw_request: Request):
    voices = voice_presets(raw_request).keys()
    return list(voices)


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("--host",
                        type=nullable_str,
                        default=None,
                        help="Host name.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=['debug', 'info', 'warning', 'error', 'critical', 'trace'],
        help="Log level for uvicorn.")
    parser.add_argument("--disable-uvicorn-access-log",
                        action="store_true",
                        help="Disable uvicorn access log.")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="Allow credentials.")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="Allowed origins.")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="Allowed methods.")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="Allowed headers.")
    parser.add_argument("--api-key",
                        type=nullable_str,
                        default=None,
                        help="If provided, the server will require this key "
                        "to be presented in the header.")
    parser.add_argument(
        "--lora-modules",
        type=nullable_str,
        default=None,
        nargs='+',
        action=LoRAParserAction,
        help="LoRA module configurations in either 'name=path' format"
        "or JSON format. "
        "Example (old format): ``'name=path'`` "
        "Example (new format): "
        "``{\"name\": \"name\", \"path\": \"lora_path\", "
        "\"base_model_name\": \"id\"}``")
    parser.add_argument(
        "--prompt-adapters",
        type=nullable_str,
        default=None,
        nargs='+',
        action=PromptAdapterParserAction,
        help="Prompt adapter configurations in the format name=path. "
        "Multiple adapters can be specified.")
    parser.add_argument("--chat-template",
                        type=nullable_str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model.")
    parser.add_argument(
        '--chat-template-content-format',
        type=str,
        default="auto",
        choices=get_args(ChatTemplateContentFormatOption),
        help='The format to render message content within a chat template.'
        '\n\n'
        '* "string" will render the content as a string. '
        'Example: ``"Hello World"``\n'
        '* "openai" will render the content as a list of dictionaries, '
        'similar to OpenAI schema. '
        'Example: ``[{"type": "text", "text": "Hello world!"}]``')
    parser.add_argument("--response-role",
                        type=nullable_str,
                        default="assistant",
                        help="The role name to return if "
                        "``request.add_generation_prompt=true``.")
    parser.add_argument("--ssl-keyfile",
                        type=nullable_str,
                        default=None,
                        help="The file path to the SSL key file.")
    parser.add_argument("--ssl-certfile",
                        type=nullable_str,
                        default=None,
                        help="The file path to the SSL cert file.")
    parser.add_argument("--ssl-ca-certs",
                        type=nullable_str,
                        default=None,
                        help="The CA certificates file.")
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)."
    )
    parser.add_argument(
        "--root-path",
        type=nullable_str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy."
    )
    parser.add_argument(
        "--middleware",
        type=nullable_str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, aphrodite will add it to the server "
        "using ``@app.middleware('http')``. "
        "If a class is provided, aphrodite will add it to the server "
        "using ``app.add_middleware()``. ")
    parser.add_argument(
        "--return-tokens-as-token-ids",
        action="store_true",
        help="When ``--max-logprobs`` is specified, represents single tokens "
        " as strings of the form 'token_id:{token_id}' so that tokens "
        "that are not JSON-encodable can be identified.")
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="If specified, will run the OpenAI frontend server in the same "
        "process as the model serving engine.")
    parser.add_argument(
        "--enable-request-id-headers",
        action="store_true",
        help="If specified, API server will add X-Request-Id header to "
        "responses. Caution: this hurts performance at high QPS.")
    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=False,
        help="Enable auto tool choice for supported models. Use "
        "``--tool-call-parser`` to specify which parser to use.")

    valid_tool_parsers = ToolParserManager.tool_parsers.keys()
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        metavar="{" + ",".join(valid_tool_parsers) + "} or name registered in "
        "--tool-parser-plugin",
        default=None,
        help=
        "Select the tool call parser depending on the model that you're using."
        " This is used to parse the model-generated tool call into OpenAI API "
        "format. Required for ``--enable-auto-tool-choice``.")

    parser.add_argument(
        "--tool-parser-plugin",
        type=str,
        default="",
        help=
        "Special the tool parser plugin write to parse the model-generated tool"
        " into OpenAI API format, the name register in this plugin can be used "
        "in ``--tool-call-parser``.")

    parser.add_argument("--audio-tokenizer-type",
                        type=str,
                        default="xcodec_tps25_0215",
                        help="The type of audio tokenizer to use. ")
    parser.add_argument("--audio-tokenizer-path",
                        type=nullable_str,
                        default=None,
                        help="The path to the audio tokenizer to use. ")
    parser.add_argument("--voice-presets-dir",
                        type=nullable_str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "./voice_presets/"),
                        help="The path to the voice presets directory. ")
    parser.add_argument("--voice-presets-refresh-interval",
                        type=int,
                        default=10,
                        help="The interval to refresh the voice presets. ")

    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument('--max-log-len',
                        type=int,
                        default=None,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        ' The default of None means unlimited.')

    parser.add_argument(
        "--disable-fastapi-docs",
        action='store_true',
        default=False,
        help="Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint."
    )
    parser.add_argument(
        "--enable-prompt-tokens-details",
        action='store_true',
        default=False,
        help="If set to True, enable prompt_tokens_details in usage.")
    parser.add_argument(
        "--enable-server-load-tracking",
        action='store_true',
        default=False,
        help=
        "If set to True, enable tracking server_load_metrics in the app state."
    )

    return parser


if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="aphrodite OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
