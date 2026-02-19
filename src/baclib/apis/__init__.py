"""REST API client base class with rate limiting, retries, and parallel request support."""
import urllib.request
import urllib.error
import urllib.parse
import json
import time
import threading
import concurrent.futures
import http.client
import ssl
from pathlib import Path
from typing import Optional, Union, Any, Dict, Callable, Iterable, List


# Classes --------------------------------------------------------------------------------------------------------------
class Token(str):
    """Base class for API tokens/accessions.
    
    Inherits from str to allow direct usage in f-strings and URLs 
    while providing type safety.
    """
    def __repr__(self):
        return f"{self.__class__.__name__}('{self}')"


class ApiClient:
    """
    Base class for REST API clients.
    Handles authentication, retries, rate limiting, and threading.
    Supports persistent connections (Keep-Alive) when used as a context manager.
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
        
        # Persistence state
        self._keep_alive_active = False
        self._thread_local = threading.local()

        # Parse base URL for http.client
        parsed = urllib.parse.urlparse(self.base_url)
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        self._scheme = parsed.scheme
        self._api_root = parsed.path # e.g. /api

    def _get_headers(self) -> Dict[str, str]:
        headers = {'User-Agent': 'baclib-client/0.1'}
        if self._api_key:
            headers['api-key'] = self._api_key
        return headers

    def _wait_for_rate_limit(self):
        if self._rate_limit_delay <= 0: return
        
        with self._lock:
            now = time.time()
            next_allowed = self._last_request_time + self._rate_limit_delay
            wait = next_allowed - now
            
            if wait > 0:
                time.sleep(wait)
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

    def _get_connection(self) -> Union[http.client.HTTPSConnection, http.client.HTTPConnection]:
        """Returns a thread-local connection, creating it if necessary."""
        if not hasattr(self._thread_local, 'conn'):
            context = ssl.create_default_context() if self._scheme == 'https' else None
            if self._scheme == 'https':
                self._thread_local.conn = http.client.HTTPSConnection(self._host, self._port, context=context)
            else:
                self._thread_local.conn = http.client.HTTPConnection(self._host, self._port)
        return self._thread_local.conn

    def _close_conn_for_thread(self):
        """Closes the connection for the current thread if it exists."""
        if hasattr(self._thread_local, 'conn'):
            self._thread_local.conn.close()
            del self._thread_local.conn

    def request(self, method: str, path: str, params: dict = None, json_data: dict = None, stream: bool = False) -> Any:
        """
        Performs an HTTP request with rate limiting and error handling.
        Uses persistent connection if available (inside context manager), otherwise urllib.
        """
        if self._keep_alive_active:
            return self._request_persistent(method, path, params, json_data, stream)
        else:
            return self._request_urllib(method, path, params, json_data, stream)

    def _request_persistent(self, method: str, path: str, params: dict, json_data: dict, stream: bool) -> Any:
        # Construct path
        full_path = f"{self._api_root}/{path.lstrip('/')}"
        if params:
            params = {k: v for k, v in params.items() if v is not None}
            query_string = urllib.parse.urlencode(params, doseq=True)
            full_path = f"{full_path}?{query_string}"

        headers = self._get_headers()
        body = None
        if json_data:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(json_data).encode('utf-8')

        for attempt in range(self._max_retries + 1):
            self._wait_for_rate_limit()
            conn = self._get_connection()
            
            try:
                conn.request(method, full_path, body=body, headers=headers)
                response = conn.getresponse()
                
                # Check for errors status codes
                if response.status >= 400:
                    # Raise urllib.error.HTTPError to match existing interface
                    # We need: url, code, msg, hdrs, fp
                    raise urllib.error.HTTPError(
                        url=self.base_url + full_path,
                        code=response.status,
                        msg=response.reason,
                        hdrs=response.headers,
                        fp=None
                    )

                if stream:
                    # For stream, we return the http.client.HTTPResponse object
                    # It behaves like a file object (read, etc.)
                    # IMPORTANT: The connection cannot be reused until this is fully read or closed.
                    return response

                data = response.read()
                if not data: return None
                return json.loads(data)

            except (http.client.CannotSendRequest, http.client.ResponseNotReady, BrokenPipeError, ConnectionError):
                # Connection might have dropped
                self._close_conn_for_thread()
                if attempt < self._max_retries:
                    # Retry with new connection
                    continue
                raise
            except urllib.error.HTTPError as e:
                # Handle retries for server errors
                if e.code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise

    def _request_urllib(self, method: str, path: str, params: dict, json_data: dict, stream: bool) -> Any:
        """Original implementation using urllib."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        if params:
            params = {k: v for k, v in params.items() if v is not None}
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
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise

            except urllib.error.URLError as e:
                 if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
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
             # If using persistent connection, we must ensure response is fully consumed/closed
             if hasattr(response, 'close'):
                 response.close()
        return dest_path

    def __enter__(self): 
        self._keep_alive_active = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self._keep_alive_active = False
        # Clean up the main thread's connection
        self._close_conn_for_thread()
    
    def close(self): 
        """Explicit close."""
        self._close_conn_for_thread()

