# File: test_redis_pubsub.py
# Simple utility to test Redis pub/sub for analysis updates
import redis
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

def test_redis_listener(analysis_id: str):
    """
    Test utility to listen for analysis updates via Redis pub/sub.
    Run this in a separate terminal to see real-time updates.
    
    Usage: python test_redis_pubsub.py
    """
    redis_client = redis.from_url(REDIS_URL)
    pubsub = redis_client.pubsub()
    
    # Subscribe to the analysis channel
    channel = f"analysis:{analysis_id}"
    pubsub.subscribe(channel)
    
    print(f"🔄 Listening for updates on channel: {channel}")
    print("Press Ctrl+C to stop\n")
    
    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'].decode('utf-8'))
                    timestamp = time.strftime('%H:%M:%S', time.localtime(data.get('timestamp', time.time())))
                    
                    print(f"[{timestamp}] Status: {data.get('status', 'unknown')}")
                    print(f"         Progress: {data.get('progress', 'No progress info')}")
                    
                    if data.get('step') and data.get('total_steps'):
                        print(f"         Step: {data['step']}/{data['total_steps']}")
                    
                    if data.get('status') == 'completed':
                        print(f"✅ Analysis completed!")
                        if data.get('summary'):
                            print(f"   Summary: {json.dumps(data['summary'], indent=2)}")
                        break
                    elif data.get('status') == 'failed':
                        print(f"❌ Analysis failed: {data.get('error_message', 'Unknown error')}")
                        break
                    
                    print()  # Empty line for readability
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse message: {e}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
    except KeyboardInterrupt:
        print("\n👋 Stopping listener...")
    finally:
        pubsub.close()
        redis_client.close()

if __name__ == "__main__":
    # Example usage - replace with actual analysis_id
    test_analysis_id = input("Enter analysis_id to monitor (or press Enter for test): ").strip()
    if not test_analysis_id:
        test_analysis_id = "test-analysis-123"
    
    test_redis_listener(test_analysis_id)