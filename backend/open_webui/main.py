import asyncio
import inspect
import json
import logging
import mimetypes
import os
import shutil
import sys
import time
import random
import ipaddress

from contextlib import asynccontextmanager
from urllib.parse import urlencode, parse_qs, urlparse
from pydantic import BaseModel
from sqlalchemy import text

from typing import Optional, Set, Union
from aiocache import cached
import aiohttp
import requests


from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
    applications,
    BackgroundTasks,
)

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response, StreamingResponse


from open_webui.utils import logger
from open_webui.utils.audit import AuditLevel, AuditLoggingMiddleware
from open_webui.utils.logger import start_logger
from open_webui.socket.main import (
    app as socket_app,
    periodic_usage_pool_cleanup,
)
from open_webui.routers import (
    audio,
    images,
    ollama,
    openai,
    retrieval,
    pipelines,
    tasks,
    auths,
    auths_saml,
    channels,
    chats,
    folders,
    configs,
    groups,
    files,
    functions,
    memories,
    models,
    knowledge,
    prompts,
    evaluations,
    tools,
    users,
    utils,
)

from open_webui.routers.retrieval import (
    get_embedding_function,
    get_ef,
    get_rf,
)

from open_webui.internal.db import Session, engine

from open_webui.models.functions import Functions
from open_webui.models.models import Models
from open_webui.models.users import UserModel, Users
from open_webui.models.chats import Chats

from open_webui.config import (
    LICENSE_KEY,
    # Ollama
    ENABLE_OLLAMA_API,
    OLLAMA_BASE_URLS,
    OLLAMA_API_CONFIGS,
    # OpenAI
    ENABLE_OPENAI_API,
    ONEDRIVE_CLIENT_ID,
    OPENAI_API_BASE_URLS,
    OPENAI_API_KEYS,
    OPENAI_API_CONFIGS,
    # Direct Connections
    ENABLE_DIRECT_CONNECTIONS,
    # Tool Server Configs
    TOOL_SERVER_CONNECTIONS,
    # Code Execution
    ENABLE_CODE_EXECUTION,
    CODE_EXECUTION_ENGINE,
    CODE_EXECUTION_JUPYTER_URL,
    CODE_EXECUTION_JUPYTER_AUTH,
    CODE_EXECUTION_JUPYTER_AUTH_TOKEN,
    CODE_EXECUTION_JUPYTER_AUTH_PASSWORD,
    CODE_EXECUTION_JUPYTER_TIMEOUT,
    ENABLE_CODE_INTERPRETER,
    CODE_INTERPRETER_ENGINE,
    CODE_INTERPRETER_PROMPT_TEMPLATE,
    CODE_INTERPRETER_JUPYTER_URL,
    CODE_INTERPRETER_JUPYTER_AUTH,
    CODE_INTERPRETER_JUPYTER_AUTH_TOKEN,
    CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD,
    CODE_INTERPRETER_JUPYTER_TIMEOUT,
    # Image
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
    COMFYUI_BASE_URL,
    COMFYUI_API_KEY,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NODES,
    ENABLE_IMAGE_GENERATION,
    ENABLE_IMAGE_PROMPT_GENERATION,
    IMAGE_GENERATION_ENGINE,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    IMAGES_OPENAI_API_BASE_URL,
    IMAGES_OPENAI_API_KEY,
    IMAGES_GEMINI_API_BASE_URL,
    IMAGES_GEMINI_API_KEY,
    # Audio
    AUDIO_STT_ENGINE,
    AUDIO_STT_MODEL,
    AUDIO_STT_OPENAI_API_BASE_URL,
    AUDIO_STT_OPENAI_API_KEY,
    AUDIO_TTS_API_KEY,
    AUDIO_TTS_ENGINE,
    AUDIO_TTS_MODEL,
    AUDIO_TTS_OPENAI_API_BASE_URL,
    AUDIO_TTS_OPENAI_API_KEY,
    AUDIO_TTS_SPLIT_ON,
    AUDIO_TTS_VOICE,
    AUDIO_TTS_AZURE_SPEECH_REGION,
    AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT,
    PLAYWRIGHT_WS_URI,
    PLAYWRIGHT_TIMEOUT,
    FIRECRAWL_API_BASE_URL,
    FIRECRAWL_API_KEY,
    RAG_WEB_LOADER_ENGINE,
    WHISPER_MODEL,
    DEEPGRAM_API_KEY,
    WHISPER_MODEL_AUTO_UPDATE,
    WHISPER_MODEL_DIR,
    # Retrieval
    RAG_TEMPLATE,
    DEFAULT_RAG_TEMPLATE,
    RAG_FULL_CONTEXT,
    BYPASS_EMBEDDING_AND_RETRIEVAL,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_RERANKING_MODEL,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_RELEVANCE_THRESHOLD,
    RAG_FILE_MAX_COUNT,
    RAG_FILE_MAX_SIZE,
    RAG_OPENAI_API_BASE_URL,
    RAG_OPENAI_API_KEY,
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_API_KEY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTENT_EXTRACTION_ENGINE,
    TIKA_SERVER_URL,
    DOCLING_SERVER_URL,
    DOCUMENT_INTELLIGENCE_ENDPOINT,
    DOCUMENT_INTELLIGENCE_KEY,
    MISTRAL_OCR_API_KEY,
    RAG_TOP_K,
    RAG_TOP_K_RERANKER,
    RAG_TEXT_SPLITTER,
    TIKTOKEN_ENCODING_NAME,
    PDF_EXTRACT_IMAGES,
    YOUTUBE_LOADER_LANGUAGE,
    YOUTUBE_LOADER_PROXY_URL,
    # Retrieval (Web Search)
    RAG_WEB_SEARCH_ENGINE,
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
    RAG_WEB_SEARCH_RESULT_COUNT,
    RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
    RAG_WEB_SEARCH_TRUST_ENV,
    RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
    JINA_API_KEY,
    SEARCHAPI_API_KEY,
    SEARCHAPI_ENGINE,
    SERPAPI_API_KEY,
    SERPAPI_ENGINE,
    SEARXNG_QUERY_URL,
    SERPER_API_KEY,
    SERPLY_API_KEY,
    SERPSTACK_API_KEY,
    SERPSTACK_HTTPS,
    TAVILY_API_KEY,
    TAVILY_EXTRACT_DEPTH,
    BING_SEARCH_V7_ENDPOINT,
    BING_SEARCH_V7_SUBSCRIPTION_KEY,
    BRAVE_SEARCH_API_KEY,
    EXA_API_KEY,
    PERPLEXITY_API_KEY,
    KAGI_SEARCH_API_KEY,
    MOJEEK_SEARCH_API_KEY,
    BOCHA_SEARCH_API_KEY,
    GOOGLE_PSE_API_KEY,
    GOOGLE_PSE_ENGINE_ID,
    GOOGLE_DRIVE_CLIENT_ID,
    GOOGLE_DRIVE_API_KEY,
    ONEDRIVE_CLIENT_ID,
    ENABLE_RAG_HYBRID_SEARCH,
    ENABLE_RAG_LOCAL_WEB_FETCH,
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
    ENABLE_RAG_WEB_SEARCH,
    ENABLE_GOOGLE_DRIVE_INTEGRATION,
    ENABLE_ONEDRIVE_INTEGRATION,
    UPLOAD_DIR,
    # WebUI
    WEBUI_AUTH,
    WEBUI_NAME,
    WEBUI_BANNERS,
    WEBHOOK_URL,
    ADMIN_EMAIL,
    SHOW_ADMIN_DETAILS,
    JWT_EXPIRES_IN,
    ENABLE_SIGNUP,
    ENABLE_LOGIN_FORM,
    ENABLE_API_KEY,
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS,
    API_KEY_ALLOWED_ENDPOINTS,
    ENABLE_CHANNELS,
    ENABLE_COMMUNITY_SHARING,
    ENABLE_MESSAGE_RATING,
    ENABLE_USER_WEBHOOKS,
    ENABLE_EVALUATION_ARENA_MODELS,
    USER_PERMISSIONS,
    DEFAULT_USER_ROLE,
    DEFAULT_PROMPT_SUGGESTIONS,
    DEFAULT_MODELS,
    DEFAULT_ARENA_MODEL,
    MODEL_ORDER_LIST,
    EVALUATION_ARENA_MODELS,
    # WebUI (OAuth)
    ENABLE_OAUTH_ROLE_MANAGEMENT,
    OAUTH_ROLES_CLAIM,
    OAUTH_EMAIL_CLAIM,
    OAUTH_PICTURE_CLAIM,
    OAUTH_USERNAME_CLAIM,
    OAUTH_ALLOWED_ROLES,
    OAUTH_ADMIN_ROLES,
    # WebUI (LDAP)
    ENABLE_LDAP,
    LDAP_SERVER_LABEL,
    LDAP_SERVER_HOST,
    LDAP_SERVER_PORT,
    LDAP_ATTRIBUTE_FOR_MAIL,
    LDAP_ATTRIBUTE_FOR_USERNAME,
    LDAP_SEARCH_FILTERS,
    LDAP_SEARCH_BASE,
    LDAP_APP_DN,
    LDAP_APP_PASSWORD,
    LDAP_USE_TLS,
    LDAP_CA_CERT_FILE,
    LDAP_CIPHERS,
    # SAML
    ENABLE_SAML,
    SAML_IDP_ENTITY_ID,
    SAML_IDP_SSO_URL,
    SAML_IDP_SLO_URL,
    SAML_SP_ENTITY_ID,
    SAML_SP_ACS_URL,
    SAML_SP_SLO_URL,
    SAML_IDP_CERT,
    # Misc
    ENV,
    CACHE_DIR,
    STATIC_DIR,
    FRONTEND_BUILD_DIR,
    CORS_ALLOW_ORIGIN,
    DEFAULT_LOCALE,
    OAUTH_PROVIDERS,
    WEBUI_URL,
    # Admin
    ENABLE_ADMIN_CHAT_ACCESS,
    ENABLE_ADMIN_EXPORT,
    # Tasks
    TASK_MODEL,
    TASK_MODEL_EXTERNAL,
    ENABLE_TAGS_GENERATION,
    ENABLE_TITLE_GENERATION,
    ENABLE_SEARCH_QUERY_GENERATION,
    ENABLE_RETRIEVAL_QUERY_GENERATION,
    ENABLE_AUTOCOMPLETE_GENERATION,
    TITLE_GENERATION_PROMPT_TEMPLATE,
    TAGS_GENERATION_PROMPT_TEMPLATE,
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE,
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    QUERY_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH,
    AppConfig,
    reset_config,
)
from open_webui.env import (
    AUDIT_EXCLUDED_PATHS,
    AUDIT_LOG_LEVEL,
    CHANGELOG,
    REDIS_URL,
    REDIS_SENTINEL_HOSTS,
    REDIS_SENTINEL_PORT,
    GLOBAL_LOG_LEVEL,
    MAX_BODY_LOG_SIZE,
    SAFE_MODE,
    SRC_LOG_LEVELS,
    VERSION,
    WEBUI_BUILD_HASH,
    WEBUI_SECRET_KEY,
    WEBUI_SESSION_COOKIE_SAME_SITE,
    WEBUI_SESSION_COOKIE_SECURE,
    WEBUI_AUTH_TRUSTED_EMAIL_HEADER,
    WEBUI_AUTH_TRUSTED_NAME_HEADER,
    ENABLE_WEBSOCKET_SUPPORT,
    BYPASS_MODEL_ACCESS_CONTROL,
    RESET_CONFIG_ON_START,
    OFFLINE_MODE,
    ENABLE_OTEL,
)


from open_webui.utils.models import (
    get_all_models,
    get_all_base_models,
    check_model_access,
)
from open_webui.utils.chat import (
    generate_chat_completion as chat_completion_handler,
    chat_completed as chat_completed_handler,
    chat_action as chat_action_handler,
)
from open_webui.utils.middleware import process_chat_payload, process_chat_response
from open_webui.utils.access_control import has_access

from open_webui.utils.auth import (
    get_license_data,
    get_http_authorization_cred,
    decode_token,
    get_admin_user,
    get_verified_user,
)
from open_webui.utils.oauth import OAuthManager
from open_webui.utils.security_headers import SecurityHeadersMiddleware

from open_webui.tasks import stop_task, list_tasks  # Import from tasks.py

from open_webui.utils.redis import get_sentinels_from_env


if SAFE_MODE:
    print("SAFE MODE ENABLED")
    Functions.deactivate_all_functions()

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                if path.endswith(".js"):
                    # Return 404 for javascript files
                    raise ex
                else:
                    return await super().get_response("index.html", scope)
            else:
                raise ex


print(
    rf"""
 ██████╗ ██████╗ ███████╗███╗   ██╗    ██╗    ██╗███████╗██████╗ ██╗   ██╗██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██║    ██║██╔════╝██╔══██╗██║   ██║██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║    ██║ █╗ ██║█████╗  ██████╔╝██║   ██║██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ██║███╗██║██╔══╝  ██╔══██╗██║   ██║██║
╚██████╔╝██║     ███████╗██║ ╚████║    ╚███╔███╔╝███████╗██████╔╝╚██████╔╝██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝     ╚══╝╚══╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝


v{VERSION} - building the best open-source AI user interface.
{f"Commit: {WEBUI_BUILD_HASH}" if WEBUI_BUILD_HASH != "dev-build" else ""}
https://github.com/open-webui/open-webui
"""
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_logger()
    if RESET_CONFIG_ON_START:
        reset_config()

    if LICENSE_KEY:
        get_license_data(app, LICENSE_KEY)

    asyncio.create_task(periodic_usage_pool_cleanup())
    yield


app = FastAPI(
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
    lifespan=lifespan,
)

oauth_manager = OAuthManager(app)

app.state.config = AppConfig(
    redis_url=REDIS_URL,
    redis_sentinels=get_sentinels_from_env(REDIS_SENTINEL_HOSTS, REDIS_SENTINEL_PORT),
)

app.state.WEBUI_NAME = WEBUI_NAME
app.state.LICENSE_METADATA = None


########################################
#
# OPENTELEMETRY
#
########################################

if ENABLE_OTEL:
    from open_webui.utils.telemetry.setup import setup as setup_opentelemetry

    setup_opentelemetry(app=app, db_engine=engine)


########################################
#
# OLLAMA
#
########################################


app.state.config.ENABLE_OLLAMA_API = ENABLE_OLLAMA_API
app.state.config.OLLAMA_BASE_URLS = OLLAMA_BASE_URLS
app.state.config.OLLAMA_API_CONFIGS = OLLAMA_API_CONFIGS

app.state.OLLAMA_MODELS = {}

########################################
#
# OPENAI
#
########################################

app.state.config.ENABLE_OPENAI_API = ENABLE_OPENAI_API
app.state.config.OPENAI_API_BASE_URLS = OPENAI_API_BASE_URLS
app.state.config.OPENAI_API_KEYS = OPENAI_API_KEYS
app.state.config.OPENAI_API_CONFIGS = OPENAI_API_CONFIGS

app.state.OPENAI_MODELS = {}

########################################
#
# TOOL SERVERS
#
########################################

app.state.config.TOOL_SERVER_CONNECTIONS = TOOL_SERVER_CONNECTIONS
app.state.TOOL_SERVERS = []

########################################
#
# DIRECT CONNECTIONS
#
########################################

app.state.config.ENABLE_DIRECT_CONNECTIONS = ENABLE_DIRECT_CONNECTIONS

########################################
#
# WEBUI
#
########################################

app.state.config.WEBUI_URL = WEBUI_URL
app.state.config.ENABLE_SIGNUP = ENABLE_SIGNUP
app.state.config.ENABLE_LOGIN_FORM = ENABLE_LOGIN_FORM

app.state.config.ENABLE_API_KEY = ENABLE_API_KEY
app.state.config.ENABLE_API_KEY_ENDPOINT_RESTRICTIONS = (
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS
)
app.state.config.API_KEY_ALLOWED_ENDPOINTS = API_KEY_ALLOWED_ENDPOINTS

app.state.config.JWT_EXPIRES_IN = JWT_EXPIRES_IN

app.state.config.SHOW_ADMIN_DETAILS = SHOW_ADMIN_DETAILS
app.state.config.ADMIN_EMAIL = ADMIN_EMAIL


app.state.config.DEFAULT_MODELS = DEFAULT_MODELS
app.state.config.DEFAULT_PROMPT_SUGGESTIONS = DEFAULT_PROMPT_SUGGESTIONS
app.state.config.DEFAULT_USER_ROLE = DEFAULT_USER_ROLE

app.state.config.USER_PERMISSIONS = USER_PERMISSIONS
app.state.config.WEBHOOK_URL = WEBHOOK_URL
app.state.config.BANNERS = WEBUI_BANNERS
app.state.config.MODEL_ORDER_LIST = MODEL_ORDER_LIST


app.state.config.ENABLE_CHANNELS = ENABLE_CHANNELS
app.state.config.ENABLE_COMMUNITY_SHARING = ENABLE_COMMUNITY_SHARING
app.state.config.ENABLE_MESSAGE_RATING = ENABLE_MESSAGE_RATING
app.state.config.ENABLE_USER_WEBHOOKS = ENABLE_USER_WEBHOOKS

app.state.config.ENABLE_EVALUATION_ARENA_MODELS = ENABLE_EVALUATION_ARENA_MODELS
app.state.config.EVALUATION_ARENA_MODELS = EVALUATION_ARENA_MODELS

app.state.config.OAUTH_USERNAME_CLAIM = OAUTH_USERNAME_CLAIM
app.state.config.OAUTH_PICTURE_CLAIM = OAUTH_PICTURE_CLAIM
app.state.config.OAUTH_EMAIL_CLAIM = OAUTH_EMAIL_CLAIM

app.state.config.ENABLE_OAUTH_ROLE_MANAGEMENT = ENABLE_OAUTH_ROLE_MANAGEMENT
app.state.config.OAUTH_ROLES_CLAIM = OAUTH_ROLES_CLAIM
app.state.config.OAUTH_ALLOWED_ROLES = OAUTH_ALLOWED_ROLES
app.state.config.OAUTH_ADMIN_ROLES = OAUTH_ADMIN_ROLES

app.state.config.ENABLE_LDAP = ENABLE_LDAP
app.state.config.LDAP_SERVER_LABEL = LDAP_SERVER_LABEL
app.state.config.LDAP_SERVER_HOST = LDAP_SERVER_HOST
app.state.config.LDAP_SERVER_PORT = LDAP_SERVER_PORT
app.state.config.LDAP_ATTRIBUTE_FOR_MAIL = LDAP_ATTRIBUTE_FOR_MAIL
app.state.config.LDAP_ATTRIBUTE_FOR_USERNAME = LDAP_ATTRIBUTE_FOR_USERNAME
app.state.config.LDAP_APP_DN = LDAP_APP_DN
app.state.config.LDAP_APP_PASSWORD = LDAP_APP_PASSWORD
app.state.config.LDAP_SEARCH_BASE = LDAP_SEARCH_BASE
app.state.config.LDAP_SEARCH_FILTERS = LDAP_SEARCH_FILTERS
app.state.config.LDAP_USE_TLS = LDAP_USE_TLS
app.state.config.LDAP_CA_CERT_FILE = LDAP_CA_CERT_FILE
app.state.config.LDAP_CIPHERS = LDAP_CIPHERS

# SAML配置
app.state.config.ENABLE_SAML = ENABLE_SAML
app.state.config.SAML_IDP_ENTITY_ID = SAML_IDP_ENTITY_ID
app.state.config.SAML_IDP_SSO_URL = SAML_IDP_SSO_URL
app.state.config.SAML_IDP_SLO_URL = SAML_IDP_SLO_URL
app.state.config.SAML_SP_ENTITY_ID = SAML_SP_ENTITY_ID
app.state.config.SAML_SP_ACS_URL = SAML_SP_ACS_URL
app.state.config.SAML_SP_SLO_URL = SAML_SP_SLO_URL
app.state.config.SAML_IDP_CERT = SAML_IDP_CERT

app.state.AUTH_TRUSTED_EMAIL_HEADER = WEBUI_AUTH_TRUSTED_EMAIL_HEADER
app.state.AUTH_TRUSTED_NAME_HEADER = WEBUI_AUTH_TRUSTED_NAME_HEADER

app.state.USER_COUNT = None
app.state.TOOLS = {}
app.state.FUNCTIONS = {}

########################################
#
# RETRIEVAL
#
########################################


app.state.config.TOP_K = RAG_TOP_K
app.state.config.TOP_K_RERANKER = RAG_TOP_K_RERANKER
app.state.config.RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD
app.state.config.FILE_MAX_SIZE = RAG_FILE_MAX_SIZE
app.state.config.FILE_MAX_COUNT = RAG_FILE_MAX_COUNT


app.state.config.RAG_FULL_CONTEXT = RAG_FULL_CONTEXT
app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL = BYPASS_EMBEDDING_AND_RETRIEVAL
app.state.config.ENABLE_RAG_HYBRID_SEARCH = ENABLE_RAG_HYBRID_SEARCH
app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION
)

app.state.config.CONTENT_EXTRACTION_ENGINE = CONTENT_EXTRACTION_ENGINE
app.state.config.TIKA_SERVER_URL = TIKA_SERVER_URL
app.state.config.DOCLING_SERVER_URL = DOCLING_SERVER_URL
app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT = DOCUMENT_INTELLIGENCE_ENDPOINT
app.state.config.DOCUMENT_INTELLIGENCE_KEY = DOCUMENT_INTELLIGENCE_KEY
app.state.config.MISTRAL_OCR_API_KEY = MISTRAL_OCR_API_KEY

app.state.config.TEXT_SPLITTER = RAG_TEXT_SPLITTER
app.state.config.TIKTOKEN_ENCODING_NAME = TIKTOKEN_ENCODING_NAME

app.state.config.CHUNK_SIZE = CHUNK_SIZE
app.state.config.CHUNK_OVERLAP = CHUNK_OVERLAP

app.state.config.RAG_EMBEDDING_ENGINE = RAG_EMBEDDING_ENGINE
app.state.config.RAG_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
app.state.config.RAG_EMBEDDING_BATCH_SIZE = RAG_EMBEDDING_BATCH_SIZE
app.state.config.RAG_RERANKING_MODEL = RAG_RERANKING_MODEL
app.state.config.RAG_TEMPLATE = RAG_TEMPLATE

app.state.config.RAG_OPENAI_API_BASE_URL = RAG_OPENAI_API_BASE_URL
app.state.config.RAG_OPENAI_API_KEY = RAG_OPENAI_API_KEY

app.state.config.RAG_OLLAMA_BASE_URL = RAG_OLLAMA_BASE_URL
app.state.config.RAG_OLLAMA_API_KEY = RAG_OLLAMA_API_KEY

app.state.config.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

app.state.config.YOUTUBE_LOADER_LANGUAGE = YOUTUBE_LOADER_LANGUAGE
app.state.config.YOUTUBE_LOADER_PROXY_URL = YOUTUBE_LOADER_PROXY_URL


app.state.config.ENABLE_RAG_WEB_SEARCH = ENABLE_RAG_WEB_SEARCH
app.state.config.RAG_WEB_SEARCH_ENGINE = RAG_WEB_SEARCH_ENGINE
app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL = (
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
)
app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST = RAG_WEB_SEARCH_DOMAIN_FILTER_LIST

app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION = ENABLE_GOOGLE_DRIVE_INTEGRATION
app.state.config.ENABLE_ONEDRIVE_INTEGRATION = ENABLE_ONEDRIVE_INTEGRATION
app.state.config.SEARXNG_QUERY_URL = SEARXNG_QUERY_URL
app.state.config.GOOGLE_PSE_API_KEY = GOOGLE_PSE_API_KEY
app.state.config.GOOGLE_PSE_ENGINE_ID = GOOGLE_PSE_ENGINE_ID
app.state.config.BRAVE_SEARCH_API_KEY = BRAVE_SEARCH_API_KEY
app.state.config.KAGI_SEARCH_API_KEY = KAGI_SEARCH_API_KEY
app.state.config.MOJEEK_SEARCH_API_KEY = MOJEEK_SEARCH_API_KEY
app.state.config.BOCHA_SEARCH_API_KEY = BOCHA_SEARCH_API_KEY
app.state.config.SERPSTACK_API_KEY = SERPSTACK_API_KEY
app.state.config.SERPSTACK_HTTPS = SERPSTACK_HTTPS
app.state.config.SERPER_API_KEY = SERPER_API_KEY
app.state.config.SERPLY_API_KEY = SERPLY_API_KEY
app.state.config.TAVILY_API_KEY = TAVILY_API_KEY
app.state.config.SEARCHAPI_API_KEY = SEARCHAPI_API_KEY
app.state.config.SEARCHAPI_ENGINE = SEARCHAPI_ENGINE
app.state.config.SERPAPI_API_KEY = SERPAPI_API_KEY
app.state.config.SERPAPI_ENGINE = SERPAPI_ENGINE
app.state.config.JINA_API_KEY = JINA_API_KEY
app.state.config.BING_SEARCH_V7_ENDPOINT = BING_SEARCH_V7_ENDPOINT
app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = BING_SEARCH_V7_SUBSCRIPTION_KEY
app.state.config.EXA_API_KEY = EXA_API_KEY
app.state.config.PERPLEXITY_API_KEY = PERPLEXITY_API_KEY

app.state.config.RAG_WEB_SEARCH_RESULT_COUNT = RAG_WEB_SEARCH_RESULT_COUNT
app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS = RAG_WEB_SEARCH_CONCURRENT_REQUESTS
app.state.config.RAG_WEB_LOADER_ENGINE = RAG_WEB_LOADER_ENGINE
app.state.config.RAG_WEB_SEARCH_TRUST_ENV = RAG_WEB_SEARCH_TRUST_ENV
app.state.config.PLAYWRIGHT_WS_URI = PLAYWRIGHT_WS_URI
app.state.config.PLAYWRIGHT_TIMEOUT = PLAYWRIGHT_TIMEOUT
app.state.config.FIRECRAWL_API_BASE_URL = FIRECRAWL_API_BASE_URL
app.state.config.FIRECRAWL_API_KEY = FIRECRAWL_API_KEY
app.state.config.TAVILY_EXTRACT_DEPTH = TAVILY_EXTRACT_DEPTH

app.state.EMBEDDING_FUNCTION = None
app.state.ef = None
app.state.rf = None

app.state.YOUTUBE_LOADER_TRANSLATION = None


try:
    app.state.ef = get_ef(
        app.state.config.RAG_EMBEDDING_ENGINE,
        app.state.config.RAG_EMBEDDING_MODEL,
        RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    )

    app.state.rf = get_rf(
        app.state.config.RAG_RERANKING_MODEL,
        RAG_RERANKING_MODEL_AUTO_UPDATE,
    )
except Exception as e:
    log.error(f"Error updating models: {e}")
    pass


app.state.EMBEDDING_FUNCTION = get_embedding_function(
    app.state.config.RAG_EMBEDDING_ENGINE,
    app.state.config.RAG_EMBEDDING_MODEL,
    app.state.ef,
    (
        app.state.config.RAG_OPENAI_API_BASE_URL
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_BASE_URL
    ),
    (
        app.state.config.RAG_OPENAI_API_KEY
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_API_KEY
    ),
    app.state.config.RAG_EMBEDDING_BATCH_SIZE,
)

########################################
#
# CODE EXECUTION
#
########################################

app.state.config.ENABLE_CODE_EXECUTION = ENABLE_CODE_EXECUTION
app.state.config.CODE_EXECUTION_ENGINE = CODE_EXECUTION_ENGINE
app.state.config.CODE_EXECUTION_JUPYTER_URL = CODE_EXECUTION_JUPYTER_URL
app.state.config.CODE_EXECUTION_JUPYTER_AUTH = CODE_EXECUTION_JUPYTER_AUTH
app.state.config.CODE_EXECUTION_JUPYTER_AUTH_TOKEN = CODE_EXECUTION_JUPYTER_AUTH_TOKEN
app.state.config.CODE_EXECUTION_JUPYTER_AUTH_PASSWORD = (
    CODE_EXECUTION_JUPYTER_AUTH_PASSWORD
)
app.state.config.CODE_EXECUTION_JUPYTER_TIMEOUT = CODE_EXECUTION_JUPYTER_TIMEOUT

app.state.config.ENABLE_CODE_INTERPRETER = ENABLE_CODE_INTERPRETER
app.state.config.CODE_INTERPRETER_ENGINE = CODE_INTERPRETER_ENGINE
app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE = CODE_INTERPRETER_PROMPT_TEMPLATE

app.state.config.CODE_INTERPRETER_JUPYTER_URL = CODE_INTERPRETER_JUPYTER_URL
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH = CODE_INTERPRETER_JUPYTER_AUTH
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_TOKEN = (
    CODE_INTERPRETER_JUPYTER_AUTH_TOKEN
)
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD = (
    CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD
)
app.state.config.CODE_INTERPRETER_JUPYTER_TIMEOUT = CODE_INTERPRETER_JUPYTER_TIMEOUT

########################################
#
# IMAGES
#
########################################

app.state.config.IMAGE_GENERATION_ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLE_IMAGE_GENERATION = ENABLE_IMAGE_GENERATION
app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = ENABLE_IMAGE_PROMPT_GENERATION

app.state.config.IMAGES_OPENAI_API_BASE_URL = IMAGES_OPENAI_API_BASE_URL
app.state.config.IMAGES_OPENAI_API_KEY = IMAGES_OPENAI_API_KEY

app.state.config.IMAGES_GEMINI_API_BASE_URL = IMAGES_GEMINI_API_BASE_URL
app.state.config.IMAGES_GEMINI_API_KEY = IMAGES_GEMINI_API_KEY

app.state.config.IMAGE_GENERATION_MODEL = IMAGE_GENERATION_MODEL

app.state.config.AUTOMATIC1111_BASE_URL = AUTOMATIC1111_BASE_URL
app.state.config.AUTOMATIC1111_API_AUTH = AUTOMATIC1111_API_AUTH
app.state.config.AUTOMATIC1111_CFG_SCALE = AUTOMATIC1111_CFG_SCALE
app.state.config.AUTOMATIC1111_SAMPLER = AUTOMATIC1111_SAMPLER
app.state.config.AUTOMATIC1111_SCHEDULER = AUTOMATIC1111_SCHEDULER
app.state.config.COMFYUI_BASE_URL = COMFYUI_BASE_URL
app.state.config.COMFYUI_API_KEY = COMFYUI_API_KEY
app.state.config.COMFYUI_WORKFLOW = COMFYUI_WORKFLOW
app.state.config.COMFYUI_WORKFLOW_NODES = COMFYUI_WORKFLOW_NODES

app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS


########################################
#
# AUDIO
#
########################################

app.state.config.STT_OPENAI_API_BASE_URL = AUDIO_STT_OPENAI_API_BASE_URL
app.state.config.STT_OPENAI_API_KEY = AUDIO_STT_OPENAI_API_KEY
app.state.config.STT_ENGINE = AUDIO_STT_ENGINE
app.state.config.STT_MODEL = AUDIO_STT_MODEL

app.state.config.WHISPER_MODEL = WHISPER_MODEL
app.state.config.DEEPGRAM_API_KEY = DEEPGRAM_API_KEY

app.state.config.TTS_OPENAI_API_BASE_URL = AUDIO_TTS_OPENAI_API_BASE_URL
app.state.config.TTS_OPENAI_API_KEY = AUDIO_TTS_OPENAI_API_KEY
app.state.config.TTS_ENGINE = AUDIO_TTS_ENGINE
app.state.config.TTS_MODEL = AUDIO_TTS_MODEL
app.state.config.TTS_VOICE = AUDIO_TTS_VOICE
app.state.config.TTS_API_KEY = AUDIO_TTS_API_KEY
app.state.config.TTS_SPLIT_ON = AUDIO_TTS_SPLIT_ON


app.state.config.TTS_AZURE_SPEECH_REGION = AUDIO_TTS_AZURE_SPEECH_REGION
app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT = AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT


app.state.faster_whisper_model = None
app.state.speech_synthesiser = None
app.state.speech_speaker_embeddings_dataset = None


########################################
#
# TASKS
#
########################################


app.state.config.TASK_MODEL = TASK_MODEL
app.state.config.TASK_MODEL_EXTERNAL = TASK_MODEL_EXTERNAL


app.state.config.ENABLE_SEARCH_QUERY_GENERATION = ENABLE_SEARCH_QUERY_GENERATION
app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION = ENABLE_RETRIEVAL_QUERY_GENERATION
app.state.config.ENABLE_AUTOCOMPLETE_GENERATION = ENABLE_AUTOCOMPLETE_GENERATION
app.state.config.ENABLE_TAGS_GENERATION = ENABLE_TAGS_GENERATION
app.state.config.ENABLE_TITLE_GENERATION = ENABLE_TITLE_GENERATION


app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE = TITLE_GENERATION_PROMPT_TEMPLATE
app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE = TAGS_GENERATION_PROMPT_TEMPLATE
app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE = (
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE
)

app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE = (
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
)
app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE = QUERY_GENERATION_PROMPT_TEMPLATE
app.state.config.AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE = (
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE
)
app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH = (
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH
)


########################################
#
# WEBUI
#
########################################

app.state.MODELS = {}

print(f"%%%%%%%%%%\n DEFAULT_PROMPT_SUGGESTIONS: {DEFAULT_PROMPT_SUGGESTIONS} \n%%%%%%%%%%")

class IPBlockingMiddleware(BaseHTTPMiddleware):
    """
    IP地址拦截中间件
    
    用于拦截恶意IP地址的请求，防止攻击和恶意扫描。
    支持单个IP地址和IP地址段的拦截。
    """
    
    def __init__(self, app, blocked_ips: Set[Union[str, ipaddress.IPv4Network, ipaddress.IPv6Network]] = None):
        """
        初始化IP拦截中间件
        
        Args:
            app: FastAPI应用实例
            blocked_ips: 被拦截的IP地址集合，支持单个IP和CIDR格式的网段
        """
        super().__init__(app)
        self.blocked_ips = blocked_ips or set()
        self.blocked_networks = set()
        self.blocked_single_ips = set()
        
        # 分类处理IP地址和网段
        for ip_item in self.blocked_ips:
            try:
                if isinstance(ip_item, str):
                    if '/' in ip_item:
                        # CIDR格式的网段
                        network = ipaddress.ip_network(ip_item, strict=False)
                        self.blocked_networks.add(network)
                    else:
                        # 单个IP地址
                        ip_addr = ipaddress.ip_address(ip_item)
                        self.blocked_single_ips.add(ip_addr)
                else:
                    # 已经是ipaddress对象
                    if hasattr(ip_item, 'network_address'):
                        self.blocked_networks.add(ip_item)
                    else:
                        self.blocked_single_ips.add(ip_item)
            except ValueError as e:
                log.warning(f"无效的IP地址格式: {ip_item}, 错误: {e}")
    
    def is_ip_blocked(self, client_ip: str) -> bool:
        """
        检查IP地址是否被拦截
        
        Args:
            client_ip: 客户端IP地址字符串
            
        Returns:
            bool: 如果IP被拦截返回True，否则返回False
        """
        try:
            ip_addr = ipaddress.ip_address(client_ip)
            
            # 检查单个IP地址
            if ip_addr in self.blocked_single_ips:
                return True
            
            # 检查IP网段
            for network in self.blocked_networks:
                if ip_addr in network:
                    return True
                    
            return False
        except ValueError:
            # 无效的IP地址格式，记录警告但不拦截
            log.warning(f"无效的客户端IP地址格式: {client_ip}")
            return False
    
    async def dispatch(self, request: Request, call_next):
        """
        中间件主要处理逻辑
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理函数
            
        Returns:
            Response: HTTP响应对象
        """
        # 获取客户端IP地址
        client_ip = self.get_client_ip(request)
        
        # 检查IP是否被拦截
        if self.is_ip_blocked(client_ip):
            log.warning(f"拦截恶意IP请求: {client_ip} - {request.method} {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Just a moment... We're brewing some coffee!"}
            )
        
        # IP未被拦截，继续处理请求
        response = await call_next(request)
        return response
    
    def get_client_ip(self, request: Request) -> str:
        """
        获取客户端真实IP地址
        
        优先从代理头部获取真实IP，如果没有则使用直连IP
        
        Args:
            request: FastAPI请求对象
            
        Returns:
            str: 客户端IP地址
        """
        # 优先从X-Forwarded-For头部获取真实IP（适用于代理/负载均衡器场景）
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For可能包含多个IP，取第一个（客户端IP）
            client_ip = forwarded_for.split(",")[0].strip()
            return client_ip
        
        # 从X-Real-IP头部获取（Nginx等代理常用）
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # 从CF-Connecting-IP头部获取（Cloudflare）
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip.strip()
        
        # 最后使用直连IP地址
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return "unknown"


class RedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if the request is a GET request
        if request.method == "GET":
            path = request.url.path
            query_params = dict(parse_qs(urlparse(str(request.url)).query))

            # Check for the specific watch path and the presence of 'v' parameter
            if path.endswith("/watch") and "v" in query_params:
                video_id = query_params["v"][0]  # Extract the first 'v' parameter
                encoded_video_id = urlencode({"youtube": video_id})
                redirect_url = f"/?{encoded_video_id}"
                return RedirectResponse(url=redirect_url)

        # Proceed with the normal flow of other requests
        response = await call_next(request)
        return response


# 配置需要拦截的恶意IP地址
BLOCKED_IPS = {
    "64.39.106.89",  # 当前攻击IP
    # 可以添加更多恶意IP地址或IP段，例如：
    # "192.168.1.0/24",  # 拦截整个网段
    # "10.0.0.1",       # 拦截单个IP
}

# Add the middleware to the app
app.add_middleware(IPBlockingMiddleware, blocked_ips=BLOCKED_IPS)
app.add_middleware(RedirectMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def commit_session_after_request(request: Request, call_next):
    response = await call_next(request)
    # log.debug("Commit session after request")
    Session.commit()
    return response


@app.middleware("http")
async def check_url(request: Request, call_next):
    start_time = int(time.time())
    request.state.token = get_http_authorization_cred(
        request.headers.get("Authorization")
    )

    request.state.enable_api_key = app.state.config.ENABLE_API_KEY
    response = await call_next(request)
    process_time = int(time.time()) - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def inspect_websocket(request: Request, call_next):
    if (
        "/ws/socket.io" in request.url.path
        and request.query_params.get("transport") == "websocket"
    ):
        upgrade = (request.headers.get("Upgrade") or "").lower()
        connection = (request.headers.get("Connection") or "").lower().split(",")
        # Check that there's the correct headers for an upgrade, else reject the connection
        # This is to work around this upstream issue: https://github.com/miguelgrinberg/python-engineio/issues/367
        if upgrade != "websocket" or "upgrade" not in connection:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid WebSocket upgrade request"},
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/ws", socket_app)


app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
app.include_router(openai.router, prefix="/openai", tags=["openai"])


app.include_router(pipelines.router, prefix="/api/v1/pipelines", tags=["pipelines"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])

app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["retrieval"])

app.include_router(configs.router, prefix="/api/v1/configs", tags=["configs"])

app.include_router(auths.router, prefix="/api/v1/auths", tags=["auths"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])


# 仅在SAML启用时注册SAML路由
if app.state.config.ENABLE_SAML:
    app.include_router(auths_saml.router, prefix="/api/v1/auths/saml", tags=["auths", "saml"])


app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(chats.router, prefix="/api/v1/chats", tags=["chats"])

app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])

app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(folders.router, prefix="/api/v1/folders", tags=["folders"])
app.include_router(groups.router, prefix="/api/v1/groups", tags=["groups"])
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(functions.router, prefix="/api/v1/functions", tags=["functions"])
app.include_router(
    evaluations.router, prefix="/api/v1/evaluations", tags=["evaluations"]
)
app.include_router(utils.router, prefix="/api/v1/utils", tags=["utils"])


try:
    audit_level = AuditLevel(AUDIT_LOG_LEVEL)
except ValueError as e:
    logger.error(f"Invalid audit level: {AUDIT_LOG_LEVEL}. Error: {e}")
    audit_level = AuditLevel.NONE

if audit_level != AuditLevel.NONE:
    app.add_middleware(
        AuditLoggingMiddleware,
        audit_level=audit_level,
        excluded_paths=AUDIT_EXCLUDED_PATHS,
        max_body_size=MAX_BODY_LOG_SIZE,
    )
##################################
#
# Chat Endpoints
#
##################################


@app.get("/api/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    def get_filtered_models(models, user):
        filtered_models = []
        for model in models:
            if model.get("arena"):
                if has_access(
                    user.id,
                    type="read",
                    access_control=model.get("info", {})
                    .get("meta", {})
                    .get("access_control", {}),
                ):
                    filtered_models.append(model)
                continue

            model_info = Models.get_model_by_id(model["id"])
            if model_info:
                if user.id == model_info.user_id or has_access(
                    user.id, type="read", access_control=model_info.access_control
                ):
                    filtered_models.append(model)

        return filtered_models

    all_models = await get_all_models(request, user=user)

    models = []
    for model in all_models:
        # Filter out filter pipelines
        if "pipeline" in model and model["pipeline"].get("type", None) == "filter":
            continue

        model_tags = [
            tag.get("name")
            for tag in model.get("info", {}).get("meta", {}).get("tags", [])
        ]
        tags = [tag.get("name") for tag in model.get("tags", [])]

        tags = list(set(model_tags + tags))
        model["tags"] = [{"name": tag} for tag in tags]

        models.append(model)

    model_order_list = request.app.state.config.MODEL_ORDER_LIST
    if model_order_list:
        model_order_dict = {model_id: i for i, model_id in enumerate(model_order_list)}
        # Sort models by order list priority, with fallback for those not in the list
        models.sort(
            key=lambda x: (model_order_dict.get(x["id"], float("inf")), x["name"])
        )

    # Filter out models that the user does not have access to
    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models = get_filtered_models(models, user)

    log.debug(
        f"/api/models returned filtered models accessible to the user: {json.dumps([model['id'] for model in models])}"
    )
    return {"data": models}


@app.get("/api/models/base")
async def get_base_models(request: Request, user=Depends(get_admin_user)):
    models = await get_all_base_models(request, user=user)
    return {"data": models}


@app.post("/api/chat/completions")
async def chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    # 详细记录请求体内容
    log.info(f"接收到 /api/chat/completions 请求 - 用户ID: {user.id}")
    log.debug(f"请求体详情: {json.dumps(form_data, ensure_ascii=False, default=str)}")
    
    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    model_item = form_data.pop("model_item", {})
    tasks = form_data.pop("background_tasks", None)

    try:
        if not model_item.get("direct", False):
            model_id = form_data.get("model", None)
            if model_id not in request.app.state.MODELS:
                log.error(f"模型未找到: {model_id}")
                raise Exception("Model not found")

            model = request.app.state.MODELS[model_id]
            log.info(f"使用模型: {model_id}")
            log.debug(f"模型详情: {json.dumps(model, ensure_ascii=False, default=str)}")
            model_info = Models.get_model_by_id(model_id)

            # Check if user has access to the model
            if not BYPASS_MODEL_ACCESS_CONTROL and user.role == "user":
                try:
                    check_model_access(user, model)
                except Exception as e:
                    raise e
        else:
            model = model_item
            model_info = None

            request.state.direct = True
            request.state.model = model

        metadata = {
            "user_id": user.id,
            "chat_id": form_data.pop("chat_id", None),
            "message_id": form_data.pop("id", None),
            "session_id": form_data.pop("session_id", None),
            "tool_ids": form_data.get("tool_ids", None),
            "tool_servers": form_data.pop("tool_servers", None),
            "files": form_data.get("files", None),
            "features": form_data.get("features", None),
            "variables": form_data.get("variables", None),
            "model": model,
            "direct": model_item.get("direct", False),
            **(
                {"function_calling": "native"}
                if form_data.get("params", {}).get("function_calling") == "native"
                or (
                    model_info
                    and model_info.params.model_dump().get("function_calling")
                    == "native"
                )
                else {}
            ),
        }

        request.state.metadata = metadata
        form_data["metadata"] = metadata

        log.info(f"处理聊天载荷 - 聊天ID: {metadata.get('chat_id')}, 消息ID: {metadata.get('message_id')}")
        form_data, metadata, events = await process_chat_payload(
            request, form_data, user, metadata, model
        )
        log.debug(f"处理后的载荷: {json.dumps(form_data, ensure_ascii=False, default=str)}")
        log.debug(f"元数据: {json.dumps(metadata, ensure_ascii=False, default=str)}")

    except Exception as e:
        log.debug(f"Error processing chat payload: {e}")
        Chats.upsert_message_to_chat_by_id_and_message_id(
            metadata["chat_id"],
            metadata["message_id"],
            {
                "error": {"content": str(e)},
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    try:
        log.info(f"调用聊天完成处理器 - 模型: {model.get('id', '未知')}, 流式: {form_data.get('stream', False)}")
        response = await chat_completion_handler(request, form_data, user)
        log.debug(f"聊天完成处理器响应类型: {type(response).__name__}")

        log.info(f"处理聊天响应 - 聊天ID: {metadata.get('chat_id')}")
        return await process_chat_response(
            request, response, form_data, user, metadata, model, events, tasks
        )
    except Exception as e:
        log.error(f"聊天完成处理异常: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# Alias for chat_completion (Legacy)
generate_chat_completions = chat_completion
generate_chat_completion = chat_completion


@app.post("/api/chat/completed")
async def chat_completed(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):
    # 详细记录请求体内容
    log.info(f"接收到 /api/chat/completed 请求 - 用户ID: {user.id}")
    log.debug(f"请求体详情: {json.dumps(form_data, ensure_ascii=False, default=str)}")
    
    try:
        model_item = form_data.pop("model_item", {})

        if model_item.get("direct", False):
            request.state.direct = True
            request.state.model = model_item
        # return await chat_completed_handler(request, form_data, user)
        log.info(f"调用聊天完成处理器 - 聊天ID: {form_data.get('chat_id')}")
        result = await chat_completed_handler(request, form_data, user)
        log.debug(f"聊天完成处理器结果: {json.dumps(result, ensure_ascii=False, default=str)}")
        return result
    except Exception as e:
        log.error(f"聊天完成处理异常: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/api/chat/actions/{action_id}")
async def chat_action(
    request: Request, action_id: str, form_data: dict, user=Depends(get_verified_user)
):
    try:
        model_item = form_data.pop("model_item", {})

        if model_item.get("direct", False):
            request.state.direct = True
            request.state.model = model_item

        return await chat_action_handler(request, action_id, form_data, user)
    except Exception as e:
        log.error(f"聊天完成处理异常: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/api/tasks/stop/{task_id}")
async def stop_task_endpoint(task_id: str, user=Depends(get_verified_user)):
    try:
        result = await stop_task(task_id)  # Use the function from tasks.py
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.get("/api/tasks")
async def list_tasks_endpoint(user=Depends(get_verified_user)):
    return {"tasks": list_tasks()}  # Use the function from tasks.py


##################################
#
# Config Endpoints
#
##################################


@app.get("/api/config")
async def get_app_config(request: Request):
    user = None
    if "token" in request.cookies:
        token = request.cookies.get("token")
        try:
            data = decode_token(token)
        except Exception as e:
            log.debug(e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        if data is not None and "id" in data:
            user = Users.get_user_by_id(data["id"])

    user_count = Users.get_num_users()
    onboarding = False

    if user is None:
        onboarding = user_count == 0

    return {
        **({"onboarding": True} if onboarding else {}),
        "status": True,
        "name": app.state.WEBUI_NAME,
        "version": VERSION,
        "default_locale": str(DEFAULT_LOCALE),
        "oauth": {
            "providers": {
                name: config.get("name", name)
                for name, config in OAUTH_PROVIDERS.items()
            }
        },
        "features": {
            "auth": WEBUI_AUTH,
            "auth_trusted_header": bool(app.state.AUTH_TRUSTED_EMAIL_HEADER),
            "enable_ldap": app.state.config.ENABLE_LDAP,
            "enable_api_key": app.state.config.ENABLE_API_KEY,
            "enable_signup": app.state.config.ENABLE_SIGNUP,
            "enable_login_form": app.state.config.ENABLE_LOGIN_FORM,
            "enable_websocket": ENABLE_WEBSOCKET_SUPPORT,
            **(
                {
                    "enable_direct_connections": app.state.config.ENABLE_DIRECT_CONNECTIONS,
                    "enable_channels": app.state.config.ENABLE_CHANNELS,
                    "enable_web_search": app.state.config.ENABLE_RAG_WEB_SEARCH,
                    "enable_code_execution": app.state.config.ENABLE_CODE_EXECUTION,
                    "enable_code_interpreter": app.state.config.ENABLE_CODE_INTERPRETER,
                    "enable_image_generation": app.state.config.ENABLE_IMAGE_GENERATION,
                    "enable_autocomplete_generation": app.state.config.ENABLE_AUTOCOMPLETE_GENERATION,
                    "enable_community_sharing": app.state.config.ENABLE_COMMUNITY_SHARING,
                    "enable_message_rating": app.state.config.ENABLE_MESSAGE_RATING,
                    "enable_user_webhooks": app.state.config.ENABLE_USER_WEBHOOKS,
                    "enable_admin_export": ENABLE_ADMIN_EXPORT,
                    "enable_admin_chat_access": ENABLE_ADMIN_CHAT_ACCESS,
                    "enable_google_drive_integration": app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
                    "enable_onedrive_integration": app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
                }
                if user is not None
                else {}
            ),
        },
        **(
            {
                "default_models": app.state.config.DEFAULT_MODELS,
                "default_prompt_suggestions": app.state.config.DEFAULT_PROMPT_SUGGESTIONS,
                "user_count": user_count,
                "code": {
                    "engine": app.state.config.CODE_EXECUTION_ENGINE,
                },
                "audio": {
                    "tts": {
                        "engine": app.state.config.TTS_ENGINE,
                        "voice": app.state.config.TTS_VOICE,
                        "split_on": app.state.config.TTS_SPLIT_ON,
                    },
                    "stt": {
                        "engine": app.state.config.STT_ENGINE,
                    },
                },
                "file": {
                    "max_size": app.state.config.FILE_MAX_SIZE,
                    "max_count": app.state.config.FILE_MAX_COUNT,
                },
                "permissions": {**app.state.config.USER_PERMISSIONS},
                "google_drive": {
                    "client_id": GOOGLE_DRIVE_CLIENT_ID.value,
                    "api_key": GOOGLE_DRIVE_API_KEY.value,
                },
                "onedrive": {"client_id": ONEDRIVE_CLIENT_ID.value},
                "license_metadata": app.state.LICENSE_METADATA,
                **(
                    {
                        "active_entries": app.state.USER_COUNT,
                    }
                    if user.role == "admin"
                    else {}
                ),
            }
            if user is not None
            else {}
        ),
    }


class UrlForm(BaseModel):
    url: str


@app.get("/api/webhook")
async def get_webhook_url(user=Depends(get_admin_user)):
    return {
        "url": app.state.config.WEBHOOK_URL,
    }


@app.post("/api/webhook")
async def update_webhook_url(form_data: UrlForm, user=Depends(get_admin_user)):
    app.state.config.WEBHOOK_URL = form_data.url
    app.state.WEBHOOK_URL = app.state.config.WEBHOOK_URL
    return {"url": app.state.config.WEBHOOK_URL}


class IPBlockForm(BaseModel):
    ip: str
    action: str  # "add" 或 "remove"


@app.get("/api/security/blocked-ips")
async def get_blocked_ips(user=Depends(get_admin_user)):
    """
    获取当前被拦截的IP地址列表
    
    Returns:
        dict: 包含被拦截IP地址列表的字典
    """
    # 从中间件中获取当前的拦截IP列表
    ip_middleware = None
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, IPBlockingMiddleware):
            ip_middleware = middleware
            break
    
    blocked_ips = []
    if ip_middleware and hasattr(ip_middleware, 'kwargs') and 'blocked_ips' in ip_middleware.kwargs:
        blocked_ips = list(ip_middleware.kwargs['blocked_ips'])
    
    return {
        "blocked_ips": blocked_ips,
        "count": len(blocked_ips)
    }


@app.post("/api/security/blocked-ips")
async def manage_blocked_ip(form_data: IPBlockForm, user=Depends(get_admin_user)):
    """
    动态添加或移除被拦截的IP地址
    
    Args:
        form_data: 包含IP地址和操作类型的表单数据
        user: 管理员用户对象
        
    Returns:
        dict: 操作结果
    """
    ip = form_data.ip.strip()
    action = form_data.action.lower()
    
    if action not in ["add", "remove"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="操作类型必须是 'add' 或 'remove'"
        )
    
    # 验证IP地址格式
    try:
        if '/' in ip:
            ipaddress.ip_network(ip, strict=False)
        else:
            ipaddress.ip_address(ip)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的IP地址格式: {ip}"
        )
    
    global BLOCKED_IPS
    
    if action == "add":
        if ip not in BLOCKED_IPS:
            BLOCKED_IPS.add(ip)
            log.info(f"管理员 {user.email} 添加IP到拦截列表: {ip}")
            message = f"成功添加IP地址 {ip} 到拦截列表"
        else:
            message = f"IP地址 {ip} 已在拦截列表中"
    else:  # remove
        if ip in BLOCKED_IPS:
            BLOCKED_IPS.remove(ip)
            log.info(f"管理员 {user.email} 从拦截列表移除IP: {ip}")
            message = f"成功从拦截列表移除IP地址 {ip}"
        else:
            message = f"IP地址 {ip} 不在拦截列表中"
    
    # 重新初始化IP拦截中间件（注意：这种方法在生产环境中需要重启服务才能完全生效）
    # 建议使用Redis或数据库来存储拦截IP列表以实现真正的动态更新
    
    return {
        "message": message,
        "ip": ip,
        "action": action,
        "current_blocked_ips": list(BLOCKED_IPS)
    }


@app.get("/api/version")
async def get_app_version():
    return {
        "version": VERSION,
    }


@app.get("/api/version/updates")
async def get_app_latest_release_version(user=Depends(get_verified_user)):
    if OFFLINE_MODE:
        log.debug(
            f"Offline mode is enabled, returning current version as latest version"
        )
        return {"current": VERSION, "latest": VERSION}
    try:
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(
                "https://api.github.com/repos/open-webui/open-webui/releases/latest"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                latest_version = data["tag_name"]

                return {"current": VERSION, "latest": latest_version[1:]}
    except Exception as e:
        log.debug(e)
        return {"current": VERSION, "latest": VERSION}


@app.get("/api/changelog")
async def get_app_changelog():
    return {key: CHANGELOG[key] for idx, key in enumerate(CHANGELOG) if idx < 5}


############################
# OAuth Login & Callback
############################

# SessionMiddleware is used by authlib for oauth
if len(OAUTH_PROVIDERS) > 0:
    app.add_middleware(
        SessionMiddleware,
        secret_key=WEBUI_SECRET_KEY,
        session_cookie="oui-session",
        same_site=WEBUI_SESSION_COOKIE_SAME_SITE,
        https_only=WEBUI_SESSION_COOKIE_SECURE,
    )


@app.get("/oauth/{provider}/login")
async def oauth_login(provider: str, request: Request):
    return await oauth_manager.handle_login(request, provider)


# OAuth login logic is as follows:
# 1. Attempt to find a user with matching subject ID, tied to the provider
# 2. If OAUTH_MERGE_ACCOUNTS_BY_EMAIL is true, find a user with the email address provided via OAuth
#    - This is considered insecure in general, as OAuth providers do not always verify email addresses
# 3. If there is no user, and ENABLE_OAUTH_SIGNUP is true, create a user
#    - Email addresses are considered unique, so we fail registration if the email address is already taken
@app.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request, response: Response):
    return await oauth_manager.handle_callback(request, provider, response)


@app.get("/manifest.json")
async def get_manifest_json():
    return {
        "name": app.state.WEBUI_NAME,
        "short_name": app.state.WEBUI_NAME,
        "description": "Open WebUI is an open, extensible, user-friendly interface for AI that adapts to your workflow.",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#343541",
        "orientation": "natural",
        "icons": [
            {
                "src": "/static/logo.png",
                "type": "image/png",
                "sizes": "500x500",
                "purpose": "any",
            },
            {
                "src": "/static/logo.png",
                "type": "image/png",
                "sizes": "500x500",
                "purpose": "maskable",
            },
        ],
    }


@app.get("/opensearch.xml")
async def get_opensearch_xml():
    xml_content = rf"""
    <OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/" xmlns:moz="http://www.mozilla.org/2006/browser/search/">
    <ShortName>{app.state.WEBUI_NAME}</ShortName>
    <Description>Search {app.state.WEBUI_NAME}</Description>
    <InputEncoding>UTF-8</InputEncoding>
    <Image width="16" height="16" type="image/x-icon">{app.state.config.WEBUI_URL}/static/favicon.png</Image>
    <Url type="text/html" method="get" template="{app.state.config.WEBUI_URL}/?q={"{searchTerms}"}"/>
    <moz:SearchForm>{app.state.config.WEBUI_URL}</moz:SearchForm>
    </OpenSearchDescription>
    """
    return Response(content=xml_content, media_type="application/xml")


@app.get("/health")
async def healthcheck():
    return {"status": True}


@app.get("/health/db")
async def healthcheck_with_db():
    Session.execute(text("SELECT 1;")).all()
    return {"status": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/cache", StaticFiles(directory=CACHE_DIR), name="cache")


def swagger_ui_html(*args, **kwargs):
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon.png",
    )


applications.get_swagger_ui_html = swagger_ui_html

if os.path.exists(FRONTEND_BUILD_DIR):
    mimetypes.add_type("text/javascript", ".js")
    app.mount(
        "/",
        SPAStaticFiles(directory=FRONTEND_BUILD_DIR, html=True),
        name="spa-static-files",
    )
else:
    log.warning(
        f"Frontend build directory not found at '{FRONTEND_BUILD_DIR}'. Serving API only."
    )
