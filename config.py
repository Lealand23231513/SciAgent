DEFAULT_CACHE_DIR = ".cache"
SUPPORT_TOOLS = ['websearch', 'retrieval']
SUPPORT_LLMS = [
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-0125-preview',
    'glm-3-turbo',
    'glm-4',
    'chatglm3-6b',
    'qwen1.5-0.5b-chat',
    'qwen1.5-1.8b-chat',
    'qwen1.5-4b-chat',
    'qwen1.5-7b-chat',
]
SUPPORT_EMBS = [
    "text-embedding-ada-002",
    'bge-m3',
    'bce-embbedding-base_v1'
]
EMB_MODEL_MAP = {
    "text-embedding-ada-002": {
        "api_key": None,
        "base_url": None,
    },
    "bge-m3": {"api_key": "EMPTY", "base_url": ""},
}
DEFAULT_EMB_MODEL_NAME="text-embedding-ada-002"
SUPPORT_MLLMS = [
    "gpt-4o"
]


DEFAULT_CACHE_NAMESPACE="default"