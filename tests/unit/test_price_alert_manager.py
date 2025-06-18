import asyncio
import time
from unittest.mock import AsyncMock, patch, Mock
from infrastructure.notifications.alerts.price_alerts.PriceAlertManager import AlertManager


# Constants
NUM_USERS = 20000
PROVIDED_FCM_TOKEN = "cLqhZ-V7S4Cp39SMrallJ9:APA91bFqOL37MGqqVDcv8n4V5k4qAEItzbcWqK5bP9dOnmh_VlmjIIDZtpOH4KWXUgEuRWInEuiNmJkHCP5jDlbBjdeeGipzyCS4WOBYyXtmxRSCiKNZ8bw"

# Generate 1000 identical alerts (different users, same message content)
alerts = []
symbol = "BTCUSDT"
condition_type = "price_above"
condition_value = 50000

for i in range(1, NUM_USERS + 1):
    alerts.append({
        'id': f"alert_{i}",
        'symbol': symbol,
        'user_id': str(i),
        'condition_type': condition_type,
        'condition_value': condition_value
    })

# Use same FCM token for all users
fcm_tokens = {str(i): PROVIDED_FCM_TOKEN for i in range(1, NUM_USERS + 1)}

async def test_handle_triggered_alerts():
    # Mock Supabase repo
    mock_repo = AsyncMock()
    mock_repo.get_fcm_tokens_for_users.return_value = fcm_tokens
    mock_repo.deactivate_triggered_price_alerts.return_value = None

    # Init AlertManager and inject mock repo
    alert_manager = AlertManager()
    alert_manager.supabase_repo = mock_repo

    # Sync mock for multicast send (matches firebase_admin.messaging.send_each_for_multicast)
    def mock_send_each_for_multicast(message):
        num_tokens = len(message.tokens)
        responses = [Mock(success=True, message_id=f"mock_id_{i}") for i in range(num_tokens)]
        return Mock(
            responses=responses,
            success_count=num_tokens,
            failure_count=0
        )

    # Patch Firebase messaging methods
    with patch('firebase_admin.messaging.send_each_for_multicast', side_effect=mock_send_each_for_multicast) as mock_send_multicast:
        with patch('firebase_admin.messaging.send') as mock_send:
            current_price = 60000.0  # Will trigger all alerts

            start_time = time.time()
            await alert_manager._handle_triggered_alerts(alerts, current_price)
            end_time = time.time()

            print(f"Processed {NUM_USERS} alerts in {end_time - start_time:.2f} seconds")
            print(f"send_each_for_multicast called {mock_send_multicast.call_count} times")
            print(f"send called {mock_send.call_count} times")

if __name__ == "__main__":
    asyncio.run(test_handle_triggered_alerts())
