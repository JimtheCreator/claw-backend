from prometheus_client import start_http_server, Counter

API_REQUESTS = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

start_http_server(8001)  # Metrics on port 8001