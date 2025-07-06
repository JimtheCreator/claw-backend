# Background Analysis Processing

## Overview

The market analysis system has been refactored to use background processing to prevent blocking WebSocket connections for price and pattern alerts. Heavy analysis tasks are now queued and processed by dedicated background workers.

## Architecture

### Components

1. **Analysis Service** (`src/infrastructure/notifications/analysis_service.py`)
   - Queues analysis jobs to Redis streams
   - Manages job status and cancellation
   - Provides queue statistics

2. **Analysis Worker** (`src/infrastructure/notifications/analysis_worker.py`)
   - Background process that consumes analysis jobs from Redis streams
   - Performs pattern analysis, chart generation, and LLM summarization
   - Updates job status in the database

3. **API Endpoints** (`src/presentation/api/routes/analysis.py`)
   - `/analyze/{symbol}/{interval}` - Queue background analysis
   - `/analyze-immediate/{symbol}/{interval}` - Immediate analysis for small datasets
   - `/analyze-stream/{symbol}/{interval}` - Queue and stream progress via SSE
   - `/analysis/status/{analysis_id}` - Check job status
   - `/analysis/cancel/{analysis_id}` - Cancel pending jobs
   - `/analysis/queue/stats` - Get queue statistics

## Usage

### Starting the Analysis Worker

```bash
# Start the analysis worker
python scripts/start_analysis_worker.py

# Or with custom configuration
ANALYSIS_WORKER_BATCH_SIZE=10 python scripts/start_analysis_worker.py
```

### API Usage

#### 1. Queue Background Analysis

```bash
curl -X POST "http://localhost:8000/analyze/BTCUSDT/1h" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "timeframe": "7d"
  }'
```

Response:
```json
{
  "analysis_id": "analysis_123",
  "status": "queued",
  "message": "Analysis job queued for background processing",
  "check_status_url": "/analysis/status/analysis_123"
}
```

#### 2. Check Analysis Status

```bash
curl "http://localhost:8000/analysis/status/analysis_123?user_id=user123"
```

Response:
```json
{
  "analysis_id": "analysis_123",
  "status": "completed",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:05:00Z",
  "image_url": "https://...",
  "llm_summary": "Market analysis summary...",
  "analysis_data": { ... }
}
```

#### 3. Stream Progress Updates

```bash
curl "http://localhost:8000/analyze-stream/BTCUSDT/1h?user_id=user123&timeframe=7d"
```

This returns Server-Sent Events (SSE) with progress updates:
```
data: {"status": "queued", "message": "Analysis job queued for background processing.", "analysis_id": "analysis_123"}

data: {"status": "processing", "message": "Analysis started in background."}

data: {"status": "completed", "message": "Analysis completed successfully.", "data": {...}}
```

#### 4. Cancel Analysis Job

```bash
curl -X DELETE "http://localhost:8000/analysis/cancel/analysis_123?user_id=user123"
```

#### 5. Get Queue Statistics

```bash
curl "http://localhost:8000/analysis/queue/stats"
```

Response:
```json
{
  "stream_name": "market-analysis-jobs",
  "total_jobs_in_stream": 15,
  "pending_jobs": 3,
  "active_consumers": 2,
  "timestamp": 1704067200.0
}
```

### Immediate Analysis (Small Datasets)

For datasets with less than 1000 candles, you can use immediate processing:

```bash
curl -X POST "http://localhost:8000/analyze-immediate/BTCUSDT/5m" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "timeframe": "1h"
  }'
```

This returns the analysis result immediately without queuing.

## Configuration

### Environment Variables

- `ANALYSIS_WORKER_BATCH_SIZE` - Number of jobs to process in each batch (default: 5)
- `ANALYSIS_HEALTH_CHECK_INTERVAL` - Health check interval in seconds (default: 30)
- `ANALYSIS_METRICS_PUBLISH_INTERVAL` - Metrics publishing interval in seconds (default: 60)
- `BACKGROUND_ANALYSIS_THRESHOLD` - Maximum candles for immediate analysis (default: 1000)

### Redis Configuration

The system uses Redis streams for job queuing. Ensure Redis is running and accessible.

## Job Statuses

- `processing` - Job is currently being processed by a worker
- `completed` - Job completed successfully
- `failed` - Job failed with an error
- `cancelled` - Job was cancelled by the user

## Monitoring

### Worker Metrics

Workers publish metrics to Redis under the key pattern:
```
worker_metrics:analysis-workers:worker-{pid}
```

Metrics include:
- `analyses_processed` - Total number of analyses processed
- `errors_total` - Total number of errors
- `avg_processing_time` - Average processing time per job
- `uptime` - Worker uptime in seconds
- `circuit_breaker_state` - Current circuit breaker state

### Health Checks

Workers perform periodic health checks and register themselves in Redis. You can monitor active workers by checking:
```
workers:analysis-workers:*
```

## Error Handling

### Circuit Breaker

The analysis worker implements a circuit breaker pattern to handle failures:
- After 5 consecutive failures, the circuit opens
- After 60 seconds, the circuit moves to half-open state
- Successful operations close the circuit

### Dead Letter Queue

Failed jobs are automatically retried with exponential backoff. After 3 retries, jobs are moved to a dead letter queue for manual inspection.

## Scaling

### Multiple Workers

You can run multiple analysis workers to increase throughput:

```bash
# Start multiple workers
python scripts/start_analysis_worker.py &
python scripts/start_analysis_worker.py &
python scripts/start_analysis_worker.py &
```

### Horizontal Scaling

For production deployments, you can run workers on different machines. They will automatically coordinate through Redis streams and consumer groups.

## Troubleshooting

### Common Issues

1. **Jobs not being processed**
   - Check if analysis workers are running
   - Verify Redis connection
   - Check worker logs for errors

2. **High memory usage**
   - Reduce `ANALYSIS_WORKER_BATCH_SIZE`
   - Monitor worker metrics
   - Consider running fewer workers

3. **Slow processing**
   - Increase number of workers
   - Check database performance
   - Monitor Redis performance

### Logs

Workers log important events including:
- Job processing start/completion
- Errors and retries
- Health check failures
- Circuit breaker state changes

Check the logs for detailed debugging information. 