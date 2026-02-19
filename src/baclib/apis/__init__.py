import urllib.request
import urllib.error
import urllib.parse
import json
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional, Union, Any, Dict, Callable, Iterable, List


# Classes --------------------------------------------------------------------------------------------------------------
class ApiClient:
    """
    Base class for REST API clients using standard library.
    Handles authentication, retries, rate limiting, and threading.
    """
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                 requests_per_second: float = 3.0, 
                 max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self._api_key = api_key
        self._rate_limit_delay = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self._last_request_time = 0.0
        self._max_retries = max_retries
        self._lock = threading.Lock()

    def _get_headers(self) -> Dict[str, str]:
        headers = {'User-Agent': 'baclib-client/0.1'}
        if self._api_key:
            headers['api-key'] = self._api_key
        return headers

    def _wait_for_rate_limit(self):
        if self._rate_limit_delay <= 0: return
        
        with self._lock:
            now = time.time()
            # Calculate when we can next request
            # If last request was long ago, next_allowed is in the past, wait is negative
            next_allowed = self._last_request_time + self._rate_limit_delay
            wait = next_allowed - now
            
            if wait > 0:
                time.sleep(wait)
                # Update time to now + wait (which is next_allowed) to ensure spacing
                self._last_request_time = time.time()
            else:
                self._last_request_time = now

    def run_parallel(self, func: Callable, items: Iterable, max_workers: int = 4) -> List[Any]:
        """
        executes `func` for each item in `items` in parallel using threads.
        `func` should accept a single argument (item).
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in items}
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    results.append(future.result())
                except Exception as e:
                    # In a real app we might want to capture errors instead of raising
                    raise e
        return results

    def request(self, method: str, path: str, params: dict = None, json_data: dict = None, stream: bool = False) -> Any:
        """
        Performs an HTTP request with rate limiting and error handling.
        Returns the response object (file-like) if stream=True, else parsed JSON.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        if params:
            # removing None values
            params = {k: v for k, v in params.items() if v is not None}
            # handle lists in params (e.g. ?id=1&id=2)
            # urllib.parse.urlencode handles lists if doseq=True
            query_string = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}?{query_string}"

        headers = self._get_headers()
        data = None
        if json_data:
            headers['Content-Type'] = 'application/json'
            data = json.dumps(json_data).encode('utf-8')

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        for attempt in range(self._max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = urllib.request.urlopen(req)
                if stream:
                    return response
                
                content = response.read()
                if not content: return None
                return json.loads(content)

            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    wait_time = (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise

            except urllib.error.URLError as e:
                 if attempt < self._max_retries:
                    wait_time = (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                 raise e

    def get(self, path: str, params: dict = None, **kwargs) -> Any:
        return self.request('GET', path, params=params, **kwargs)

    def post(self, path: str, json: dict = None, **kwargs) -> Any:
        return self.request('POST', path, json_data=json, **kwargs)

    def download(self, path: str, destination: Union[str, Path], params: dict = None, method: str = 'GET', json: dict = None):
        """
        Downloads a file to the specified destination.
        """
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        response = self.request(method, path, params=params, json_data=json, stream=True)
        try:
             with open(dest_path, 'wb') as f:
                 shutil.copyfileobj(response, f)
        finally:
             response.close()
        return dest_path

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def close(self): pass
