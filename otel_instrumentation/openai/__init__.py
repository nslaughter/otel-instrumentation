import json
from functools import wraps
import os

from typing import Optional

import openai

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from opentelemetry.metrics import get_meter_provider, set_meter_provider, Counter
from opentelemetry.sdk.metrics import Counter, MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
#     AggregationTemporality,
)
# from opentelemetry.sdk.metrics.view import LastValueAggregation
from opentelemetry.metrics import get_meter

from .utils import calculate_cost, get_rate_limits
from .version import __version__
# from package import _instruments


# from opentelemetry.instrumentation.requests import RequestsInstrumentor
INSTRUMENTION_NAME = "io.opentelemetry.contrib.openai"
INSTRUMENTATION_VERSION = "0.0.0-experimental"

OPENAI_KIND_CHAT = "openai.chatcompletion"
OPENAI_KIND_COMPLETION = "openai.completion"
OPENAI_KIND_EMBEDDING = "openai.embedding"

OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'localhost:4317')

# There are two ways to instrument the OpenAI API:
# 1. Use the OpenTelemetry API to create spans and add attributes to them
# 2. Use the OpenTelemetry API to create metrics and add values to them
# 3. Use the OpenTelemetry API to create logs and add attributes to them

# Service name is required for most backends

chat_resource = Resource(attributes={
    SERVICE_NAME: INSTRUMENTION_NAME,
    "openai.version": INSTRUMENTATION_VERSION,
    "openai.api_resource": "ChatCompletions",
})

span_exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT)
span_provider = TracerProvider(resource=chat_resource)
span_processor = BatchSpanProcessor(span_exporter)

span_provider.add_span_processor(span_processor)
trace.set_tracer_provider(span_provider)

tracer = trace.get_tracer(INSTRUMENTION_NAME, INSTRUMENTATION_VERSION)

class OpenAIInstrumentor(BaseInstrumentor):

    @staticmethod
    def instrument_embedding(client=None, tracer_provider=None):
        """Instrument the OpenAI Embedding API client instance to trace requests and responses."""
        if not client:
            client = openai.Embedding()
        if not tracer_provider:
            tracer_provider = trace.get_tracer_provider()
        instrument_client_method(openai.Embedding, "create", embeddings_request_tracing_hook, embeddings_response_tracing_hook)
        instrument_async_client_method(openai.Embedding, "acreate", embeddings_request_tracing_hook, embeddings_response_tracing_hook)
        instrument_client_method(openai.Embedding, "create", embeddings_request_metrics_hook, embeddings_response_metrics_hook)
        instrument_async_client_method(openai.Embedding, "acreate", embeddings_request_metrics_hook, embeddings_response_metrics_hook)


    @staticmethod
    def instrument_chat(client=None, tracer_provider=None):
        """Instrument the OpenAI Chat API client instance to trace requests and responses."""
        if not client:
            client = openai.ChatCompletion()
        if not tracer_provider:
            tracer_provider = trace.get_tracer_provider()
        # openai.ChatCompletion = _instrument_chat(client=client, tracer_provider=tracer_provider)
        instrument_client_method(openai.ChatCompletion, "create", chat_request_tracing_hook, chat_response_tracing_hook)
        instrument_async_client_method(openai.ChatCompletion, "areate", chat_request_tracing_hook, chat_response_tracing_hook)
        instrument_client_method(openai.ChatCompletion, "create", chat_request_metrics_hook, chat_response_metrics_hook)
        instrument_async_client_method(openai.ChatCompletion, "areate", chat_request_metrics_hook, chat_response_metrics_hook)


# instruments client methods with request/response hooks
# NOTE: the particulars of how this wrapper works are specific to underlying
def instrument_client_method(cls, method_name, request_hook, response_hook):
    original_method = getattr(cls, method_name)

    @classmethod
    def new_method(cls, *args, **kwargs):
        with tracer.start_as_current_span(method_name) as span:
            if request_hook:
                request_hook(span, kwargs)
            response = original_method(*args, **kwargs)
            if response_hook:
                response_hook(span, response)
            return response

    setattr(cls, method_name, new_method)

def instrument_async_client_method(cls, method_name, request_hook, response_hook):
    original_method = getattr(cls, method_name)

    @classmethod
    async def new_method(cls, *args, **kwargs):
        with tracer.start_as_current_span(method_name) as span:
            if request_hook:
                request_hook(span, kwargs)
            response = await original_method(*args, **kwargs)
            if response_hook:
                response_hook(span, response)
            return response

    setattr(cls, method_name, new_method)

def chat_request_tracing_hook(span, scope):
    if span and span.is_recording():
        span.set_attributes({
            "model": scope.get('model', ''),
            "temperature": scope.get('temperature', ''),
            "max_tokens": scope.get('max_tokens', ''),
            "stop": scope.get('stop', ''),
        })

def chat_response_tracing_hook(span, response):
    if span and span.is_recording():
        cost = calculate_cost(response)
        strResponse = str(response)
        span.set_attributes({
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "cost": cost,
            "response": strResponse,
        })

# NOTE: defaulting temporality for now

metrics_exporter = OTLPMetricExporter(
    endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
)
metrics_reader = PeriodicExportingMetricReader(exporter=metrics_exporter)

chat_provider = MeterProvider(metric_readers=[metrics_reader], resource=chat_resource)
set_meter_provider(chat_provider)

chat_meter = get_meter(__name__, __version__, chat_provider)

counter_completion_count = chat_meter.create_counter("openai.chatcompletion.completion_count", "{{unit}}", "Number of completion requests")
counter_completion_tokens = chat_meter.create_counter("openai.chatcompletion.completion_tokens", "{{unit}}", "Number of completion tokens")
counter_prompt_tokens = chat_meter.create_counter("openai.chatcompletion.prompt_tokens", "{{unit}}", "Number of prompt tokens")
counter_total_tokens = chat_meter.create_counter("openai.chatcompletion.total_tokens", "{{unit}}", "Number of total tokens")
# gauge_ratelimit_requests = chat_meter.create_observable_gauge("openai.chatcompletion.ratelimit_rpm", "{{unit}}", "Rate limit in requests per minute")
# gauge_rate_limit_tokens = chat_meter.create_gauge("openai.chatcompletion.ratelimit_tpm", "{{unit}}", "Rate limit in tokens per minute")

def chat_request_metrics_hook(span, scope):
    model = scope.get('model', '')
    if model == '':
        model = 'gpt-3.5-turbo'
    rpm_limit, tpm_limit = get_rate_limits(model)
    attributes = {"model": model, "rpm_limit": rpm_limit, "tpm_limit": tpm_limit}
    counter_completion_count.add(1, attributes=attributes)

def chat_response_metrics_hook(span, response):
    if response:
        counter_completion_tokens.add(response.usage.completion_tokens)
        counter_prompt_tokens.add(response.usage.prompt_tokens)
        counter_total_tokens.add(response.usage.total_tokens)

embeddings_resource = Resource(attributes={
    SERVICE_NAME: INSTRUMENTION_NAME,
    "openai.version": INSTRUMENTATION_VERSION,
    "openai.api_resource": "Embedding",
})

def embeddings_request_tracing_hook(span, scope):
    if span and span.is_recording():
        span.set_attributes({
            "model": scope.get('model', ''),
            "temperature": scope.get('temperature', ''),
            "user": scope.get('user', ''),
        })

def embeddings_response_tracing_hook(span, response):
    if span and span.is_recording():
        cost = calculate_cost(response)
        # strResponse = str(response)
        span.set_attributes({
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "cost": cost,
        })


embeddings_metrics_reader = PeriodicExportingMetricReader(exporter=metrics_exporter)

embeddings_provider = MeterProvider(metric_readers=[embeddings_metrics_reader], resource=embeddings_resource)
set_meter_provider(embeddings_provider)

embeddings_meter = get_meter(__name__, __version__, embeddings_provider)

embeddings_counter_completion_count = embeddings_meter.create_counter("openai.embeddings.requests", "{{unit}}", "Number of completion requests")
embeddings_counter_total_tokens = embeddings_meter.create_counter("openai.embeddings.total_tokens", "{{unit}}", "Number of total tokens")


def embeddings_request_metrics_hook(span, scope):
    model = scope.get('model', '')
    if model == '':
        model = 'text-embedding-ada-002'
    rpm_limit, tpm_limit = get_rate_limits(model)
    attributes = {"model": model, "rpm_limit": rpm_limit, "tpm_limit": tpm_limit}
    embeddings_counter_completion_count.add(1, attributes=attributes)

def embeddings_response_metrics_hook(span, response):
    if response:
        embeddings_counter_prompt_tokens.add(response.usage.prompt_tokens)
        embeddings_counter_total_tokens.add(response.usage.total_tokens)

# Instrument chat for tracing
instrument_client_method(openai.ChatCompletion, 'create', chat_request_tracing_hook, chat_response_tracing_hook)
instrument_async_client_method(openai.ChatCompletion, 'acreate', chat_request_tracing_hook, chat_response_tracing_hook)
instrument_client_method(openai.ChatCompletion, 'create', chat_request_metrics_hook, chat_response_metrics_hook)
instrument_async_client_method(openai.ChatCompletion, 'acreate', chat_request_metrics_hook, chat_response_metrics_hook)

# Instrument embeddings for tracing
instrument_client_method(openai.Embedding, 'create', embeddings_request_tracing_hook, embeddings_response_tracing_hook)
instrument_async_client_method(openai.Embedding, 'acreate', embeddings_request_tracing_hook, embeddings_response_tracing_hook)
instrument_client_method(openai.Embedding, 'create', embeddings_request_metrics_hook, embeddings_response_metrics_hook)
instrument_async_client_method(openai.Embedding, 'acreate', embeddings_request_metrics_hook, embeddings_response_metrics_hook)
