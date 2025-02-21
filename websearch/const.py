class BingSearchConst:
    NAME = "bing"
    DEFAULT_MAX_RESULT = 20
    MINIMUM_MAX_RESULT = 1
    MAXIMUN_MAX_RESULT = 10
    DEFAULT_TOP_K_RESULTS = 3
    MINIMUM_TOP_K_RESULTS = 1
    MAXIMUN_TOP_K_RESULTS = 5
    DEFAULT_CHUNK_SIZE = 500
    MAXIMUN_CHUNK_SIZE = 1000
    MINIMUN_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    MAX_CHUNK_OVERLAP = 200
    MIN_CHUNK_OVERLAP = 0
    DEFAULT_MAX_RETRIES = 5
    MAX_TIME_SLEEP = 0.5
    bing_search_url = "https://api.bing.microsoft.com/v7.0/search"


class ArxivConst:
    NAME = "arxiv"


class GoogleScholarConst:
    NAME = "google_scholar"


class WebSearchStateConst:
    DEFAULT_DOWNLOAD = False
    DEFAULT_TOP_K_RESULTS = 3
    DEFAULT_LOAD_ALL_AVAILABLE_META = False
    DEFAULT_LOAD_MAX_DOCS = 100
    PAPER_SEARCH_CHOICES = [ArxivConst.NAME, GoogleScholarConst.NAME]
    DEFAULT_PAPER_SEARCH_SELECT = PAPER_SEARCH_CHOICES[0]
    SE_CHOICES = [BingSearchConst.NAME]
    DEFAULT_SE_SELECT = SE_CHOICES[0]
    DEFAULT_ENABLE_SE_SEARCH = False
    DEFAULT_TOP_K_RESULTS = 3
    MAX_TOP_K_RESULTS = 10
    MIN_TOP_K_RESULTS = 1
    PAPER_SEARCH_KWARGS = ["download", "top_k_results"]