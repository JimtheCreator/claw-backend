import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from collections import defaultdict
import json

# Fixed import - import the class directly from the module
from infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager import PatternAlertManager
from mock_data.simulated_patterns import (
    get_mock_bearish_engulfing_klines,
    get_mock_bullish_engulfing_klines,
    get_mock_hammer_klines,
    get_mock_shooting_star_klines,
    get_mock_doji_klines,
    get_mock_morning_star_klines,
    get_mock_evening_star_klines,
    get_mock_head_and_shoulders_klines,
    get_mock_double_top_klines,
    get_mock_triangle_klines,
    get_mock_cup_and_handle_klines,
    get_pattern_klines
)


class TestPatternDetectionWorkflow:
    """Test suite for pattern detection workflow using simulated patterns"""

    @pytest.fixture
    def mock_components(self):
        """Fixture to set up common mock components"""
        mock_binance_client = MagicMock()
        mock_notification_service = MagicMock()
        mock_notification_service.send_batch_fcm_notifications = AsyncMock()
        
        mock_repo = MagicMock()
        mock_repo.get_fcm_tokens_for_users = AsyncMock(return_value={"user-test-1": "fcm-token-for-user-1"})
        mock_repo.deactivate_pattern_alerts_by_criteria = AsyncMock(return_value=["alert-id-123"])
        mock_repo.create_pattern_match_history = AsyncMock()
        
        # Mock Redis cache
        mock_redis_cache = MagicMock()
        mock_redis_cache.initialize = AsyncMock()
        mock_redis_cache.get_cached_data = AsyncMock(return_value=None)
        mock_redis_cache.set_cached_data = AsyncMock()
        mock_redis_cache.hset_data = AsyncMock()
        mock_redis_cache.hdel_data = AsyncMock()
        mock_redis_cache.hgetall_data = AsyncMock(return_value={})

        mock_pattern_detector = MagicMock()
        
        return {
            'binance_client': mock_binance_client,
            'notification_service': mock_notification_service,
            'repo': mock_repo,
            'redis_cache': mock_redis_cache,
            'pattern_detector': mock_pattern_detector
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pattern_name,mock_data_func", [
        ('bearish_engulfing', get_mock_bearish_engulfing_klines),
        ('bullish_engulfing', get_mock_bullish_engulfing_klines),
        ('hammer', get_mock_hammer_klines),
        ('bearish_shooting_star', get_mock_shooting_star_klines),
        ('bullish_shooting_star', get_mock_shooting_star_klines),
        ('standard_doji', get_mock_doji_klines),
        ('dragonfly_doji', get_mock_doji_klines),
        ('gravestone_doji', get_mock_doji_klines),
        ('morning_star', get_mock_morning_star_klines),
        ('evening_star', get_mock_evening_star_klines),
        ('bearish_head_and_shoulders', get_mock_head_and_shoulders_klines),
        ('double_top', get_mock_double_top_klines),
        ('symmetrical_triangle', get_mock_triangle_klines),
        ('cup_and_handle', get_mock_cup_and_handle_klines),
    ])
    async def test_pattern_detection_triggers_notification(self, mock_components, pattern_name, mock_data_func):
        """Test that each pattern detection triggers appropriate notifications"""
        # ARRANGE
        mock_components['binance_client'].get_klines = AsyncMock(return_value=mock_data_func())
        
        with patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.SupabaseCryptoRepository', 
                  return_value=mock_components['repo']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.NotificationService', 
                  return_value=mock_components['notification_service']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.redis_cache', 
                  mock_components['redis_cache']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.PatternDetector', 
                  return_value=mock_components['pattern_detector']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.initialized_pattern_registry', {
                 pattern_name: {
                     'function': AsyncMock(return_value=(True, 0.95, pattern_name))
                 }
             }):

            # Instantiate manager and set up
            manager = PatternAlertManager()
            manager.binance_client = mock_components['binance_client']
            manager._is_running = True

            symbol = 'BTCUSDT'
            interval = '1h'
            user_id = 'user-test-1'

            # Initialize subscription map
            manager._subscription_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            manager._subscription_map[symbol][interval][pattern_name].add(user_id)

            # ACT
            await manager._handle_closed_candle(
                symbol=symbol,
                interval=interval,
                ohlcv={
                    'e': 'kline',
                    's': symbol,
                    'k': {
                        'i': interval,
                        'x': True,
                        't': 123400000,
                        'o': '100.0',
                        'h': '105.0',
                        'l': '95.0',
                        'c': '102.0'
                    }
                }
            )

            # ASSERT
            # Verify klines were fetched with correct pattern data
            mock_components['binance_client'].get_klines.assert_called_once_with(
                symbol=symbol, interval=interval, limit=100
            )
            
            # Verify notification was sent
            mock_components['notification_service'].send_batch_fcm_notifications.assert_called_once()
            
            # Check notification content
            call_args, _ = mock_components['notification_service'].send_batch_fcm_notifications.call_args
            assert call_args[0] == ['fcm-token-for-user-1']
            assert call_args[1] == f"📈 Pattern Alert: {symbol}"
            expected_pattern_display = pattern_name.replace('_', ' ').title()
            assert f"'{expected_pattern_display}' pattern has been detected" in call_args[2]
            
            # Verify database operations
            mock_components['repo'].deactivate_pattern_alerts_by_criteria.assert_called_once_with(
                user_ids=[user_id],
                symbol=symbol,
                pattern_name=pattern_name,
                time_interval=interval
            )
            mock_components['repo'].create_pattern_match_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_bearish_engulfing_specific_data_validation(self, mock_components):
        """Test bearish engulfing pattern with specific data validation"""
        # ARRANGE
        bearish_klines = get_mock_bearish_engulfing_klines()
        mock_components['binance_client'].get_klines = AsyncMock(return_value=bearish_klines)
        
        with patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.SupabaseCryptoRepository', 
                  return_value=mock_components['repo']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.NotificationService', 
                  return_value=mock_components['notification_service']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.redis_cache', 
                  mock_components['redis_cache']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.PatternDetector', 
                  return_value=mock_components['pattern_detector']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.initialized_pattern_registry', {
                 'bearish_engulfing': {
                     'function': AsyncMock(return_value=(True, 0.85, 'bearish_engulfing'))
                 }
             }):

            manager = PatternAlertManager()
            manager.binance_client = mock_components['binance_client']
            manager._is_running = True

            symbol = 'BTCUSDT'
            interval = '1h'
            pattern_name = 'bearish_engulfing'
            user_id = 'user-test-1'

            manager._subscription_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            manager._subscription_map[symbol][interval][pattern_name].add(user_id)

            # ACT
            await manager._handle_closed_candle(symbol, interval, {'test': 'data'})

            # ASSERT
            # Verify the pattern-specific klines were used
            fetched_klines = mock_components['binance_client'].get_klines.call_args[1]
            assert len(bearish_klines) == 100
            
            # Verify last two candles form bearish engulfing
            last_candle = bearish_klines[-1]
            prev_candle = bearish_klines[-2]
            
            # Last candle should be red (close < open) and engulf previous
            assert last_candle[4] < last_candle[1]  # close < open (red candle)
            assert prev_candle[4] > prev_candle[1]  # previous was green (close > open)
            assert last_candle[1] > prev_candle[4]  # last open > prev close
            assert last_candle[4] < prev_candle[1]  # last close < prev open

    @pytest.mark.asyncio
    async def test_morning_star_three_candle_pattern(self, mock_components):
        """Test morning star pattern with three-candle validation"""
        # ARRANGE
        morning_star_klines = get_mock_morning_star_klines()
        mock_components['binance_client'].get_klines = AsyncMock(return_value=morning_star_klines)
        
        with patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.SupabaseCryptoRepository', 
                  return_value=mock_components['repo']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.NotificationService', 
                  return_value=mock_components['notification_service']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.redis_cache', 
                  mock_components['redis_cache']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.PatternDetector', 
                  return_value=mock_components['pattern_detector']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.initialized_pattern_registry', {
                 'morning_star': {
                     'function': AsyncMock(return_value=(True, 0.90, 'morning_star'))
                 }
             }):

            manager = PatternAlertManager()
            manager.binance_client = mock_components['binance_client']
            manager._is_running = True

            symbol = 'BTCUSDT'
            interval = '4h'
            pattern_name = 'morning_star'
            user_id = 'user-test-1'

            manager._subscription_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            manager._subscription_map[symbol][interval][pattern_name].add(user_id)

            # ACT
            await manager._handle_closed_candle(symbol, interval, {'test': 'data'})

            # ASSERT
            # Verify three-candle pattern structure
            last_three = morning_star_klines[-3:]
            
            # First candle: large red
            assert last_three[0][4] < last_three[0][1]  # close < open
            # Second candle: small body (doji-like)
            small_body = abs(last_three[1][4] - last_three[1][1])
            assert small_body < 2.0  # Small body
            # Third candle: large green
            assert last_three[2][4] > last_three[2][1]  # close > open
            
            # Verify notification was sent
            mock_components['notification_service'].send_batch_fcm_notifications.assert_called_once()

    @pytest.mark.asyncio
    async def test_complex_patterns_with_extended_data(self, mock_components):
        """Test complex patterns that require more historical data"""
        complex_patterns = [
            ('bearish_head_and_shoulders', get_mock_head_and_shoulders_klines),
            ('double_top', get_mock_double_top_klines),
            ('symmetrical_triangle', get_mock_triangle_klines),
            ('cup_and_handle', get_mock_cup_and_handle_klines)
        ]
        
        for pattern_name, mock_func in complex_patterns:
            # ARRANGE
            pattern_klines = mock_func()
            mock_components['binance_client'].get_klines = AsyncMock(return_value=pattern_klines)
            
            with patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.SupabaseCryptoRepository', 
                      return_value=mock_components['repo']), \
                 patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.NotificationService', 
                      return_value=mock_components['notification_service']), \
                 patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.redis_cache', 
                      mock_components['redis_cache']), \
                 patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.PatternDetector', 
                      return_value=mock_components['pattern_detector']), \
                 patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.initialized_pattern_registry', {
                     pattern_name: {
                         'function': AsyncMock(return_value=(True, 0.88, pattern_name))
                     }
                 }):

                manager = PatternAlertManager()
                manager.binance_client = mock_components['binance_client']
                manager._is_running = True

                symbol = 'ETHUSDT'
                interval = '1d'
                user_id = 'user-test-1'

                manager._subscription_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
                manager._subscription_map[symbol][interval][pattern_name].add(user_id)

                # ACT
                await manager._handle_closed_candle(symbol, interval, {'test': 'data'})

                # ASSERT
                assert len(pattern_klines) == 100
                mock_components['notification_service'].send_batch_fcm_notifications.assert_called()
                
                # Reset mocks for next iteration
                mock_components['notification_service'].reset_mock()
                mock_components['binance_client'].reset_mock()

    @pytest.mark.asyncio
    async def test_pattern_not_detected_no_notification(self, mock_components):
        """Test that no notification is sent when pattern is not detected"""
        # ARRANGE
        mock_components['binance_client'].get_klines = AsyncMock(return_value=get_mock_bearish_engulfing_klines())
        
        with patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.SupabaseCryptoRepository', 
                  return_value=mock_components['repo']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.NotificationService', 
                  return_value=mock_components['notification_service']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.redis_cache', 
                  mock_components['redis_cache']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.PatternDetector', 
                  return_value=mock_components['pattern_detector']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.initialized_pattern_registry', {
                 'bearish_engulfing': {
                     'function': AsyncMock(return_value=(False, 0.1, None))  # Pattern not detected
                 }
             }):

            manager = PatternAlertManager()
            manager.binance_client = mock_components['binance_client']
            manager._is_running = True

            symbol = 'BTCUSDT'
            interval = '1h'
            pattern_name = 'bearish_engulfing'
            user_id = 'user-test-1'

            manager._subscription_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            manager._subscription_map[symbol][interval][pattern_name].add(user_id)

            # ACT
            await manager._handle_closed_candle(symbol, interval, {'test': 'data'})

            # ASSERT
            mock_components['binance_client'].get_klines.assert_called_once()
            mock_components['notification_service'].send_batch_fcm_notifications.assert_not_called()
            mock_components['repo'].deactivate_pattern_alerts_by_criteria.assert_not_called()
            mock_components['repo'].create_pattern_match_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_pattern_results_used(self, mock_components):
        """Test that cached pattern results are used when available"""
        # ARRANGE
        cached_result = json.dumps({
            'found': True,
            'confidence': 0.85,
            'specific_type': 'hammer'
        })
        
        mock_components['redis_cache'].get_cached_data = AsyncMock(return_value=cached_result)
        mock_components['binance_client'].get_klines = AsyncMock(return_value=get_mock_hammer_klines())
        
        mock_detector_function = AsyncMock()
        
        with patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.SupabaseCryptoRepository', 
                  return_value=mock_components['repo']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.NotificationService', 
                  return_value=mock_components['notification_service']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.redis_cache', 
                  mock_components['redis_cache']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.PatternDetector', 
                  return_value=mock_components['pattern_detector']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.initialized_pattern_registry', {
                 'hammer': {
                     'function': mock_detector_function
                 }
             }):

            manager = PatternAlertManager()
            manager.binance_client = mock_components['binance_client']
            manager._is_running = True

            symbol = 'BTCUSDT'
            interval = '1h'
            pattern_name = 'hammer'
            user_id = 'user-test-1'

            manager._subscription_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            manager._subscription_map[symbol][interval][pattern_name].add(user_id)

            # ACT
            await manager._handle_closed_candle(symbol, interval, {'test': 'data'})

            # ASSERT
            # Should use cached result and not call detector function
            mock_detector_function.assert_not_called()
            
            # Should still send notification based on cached result
            mock_components['notification_service'].send_batch_fcm_notifications.assert_called_once()
            
            # Should retrieve from cache
            cache_key = f"pattern_cache:{symbol}:{interval}:{pattern_name}"
            mock_components['redis_cache'].get_cached_data.assert_called_with(cache_key)

    @pytest.mark.asyncio
    async def test_multiple_users_multiple_patterns(self, mock_components):
        """Test handling multiple users subscribed to different patterns"""
        # ARRANGE
        mock_components['binance_client'].get_klines = AsyncMock(return_value=get_mock_doji_klines())
        mock_components['repo'].get_fcm_tokens_for_users = AsyncMock(return_value={
            "user-1": "token-1",
            "user-2": "token-2",
            "user-3": "token-3"
        })
        
        with patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.SupabaseCryptoRepository', 
                  return_value=mock_components['repo']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.NotificationService', 
                  return_value=mock_components['notification_service']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.redis_cache', 
                  mock_components['redis_cache']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.PatternDetector', 
                  return_value=mock_components['pattern_detector']), \
             patch('infrastructure.notifications.alerts.pattern_alerts.PatternAlertManager.initialized_pattern_registry', {
                 'standard_doji': {
                     'function': AsyncMock(return_value=(True, 0.92, 'standard_doji'))
                 }
             }):

            manager = PatternAlertManager()
            manager.binance_client = mock_components['binance_client']
            manager._is_running = True

            symbol = 'ADAUSDT'
            interval = '15m'
            pattern_name = 'standard_doji'

            # Set up multiple users
            manager._subscription_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            manager._subscription_map[symbol][interval][pattern_name].update(['user-1', 'user-2', 'user-3'])

            # ACT
            await manager._handle_closed_candle(symbol, interval, {'test': 'data'})

            # ASSERT
            # Should send notification to all users
            mock_components['notification_service'].send_batch_fcm_notifications.assert_called_once()
            call_args, _ = mock_components['notification_service'].send_batch_fcm_notifications.call_args
            
            # Check all tokens are included
            sent_tokens = call_args[0]
            assert 'token-1' in sent_tokens
            assert 'token-2' in sent_tokens  
            assert 'token-3' in sent_tokens

    @pytest.mark.asyncio
    async def test_pattern_data_structure_validation(self):
        """Test that all mock pattern functions return valid data structures"""
        pattern_functions = [
            ('bearish_engulfing', get_mock_bearish_engulfing_klines),
            ('bullish_engulfing', get_mock_bullish_engulfing_klines),
            ('hammer', get_mock_hammer_klines),
            ('shooting_star', get_mock_shooting_star_klines),
            ('doji', get_mock_doji_klines),
            ('morning_star', get_mock_morning_star_klines),
            ('evening_star', get_mock_evening_star_klines),
            ('head_and_shoulders', get_mock_head_and_shoulders_klines),
            ('double_top', get_mock_double_top_klines),
            ('triangle', get_mock_triangle_klines),
            ('cup_and_handle', get_mock_cup_and_handle_klines),
        ]
        
        for pattern_name, func in pattern_functions:
            klines = func()
            
            # Validate structure
            assert len(klines) == 100, f"{pattern_name} should have 100 klines"
            
            for i, kline in enumerate(klines):
                assert len(kline) == 6, f"{pattern_name} kline {i} should have 6 elements [timestamp, open, high, low, close, volume]"
                timestamp, open_price, high, low, close, volume = kline
                
                # Validate OHLC relationships
                assert high >= max(open_price, close), f"{pattern_name} kline {i}: high should be >= max(open, close)"
                assert low <= min(open_price, close), f"{pattern_name} kline {i}: low should be <= min(open, close)"
                assert volume > 0, f"{pattern_name} kline {i}: volume should be positive"
                
                # Validate data types (should be numeric)
                assert isinstance(timestamp, (int, float)), f"{pattern_name} kline {i}: timestamp should be numeric"
                assert isinstance(high, (int, float)), f"{pattern_name} kline {i}: high should be numeric"
                assert isinstance(low, (int, float)), f"{pattern_name} kline {i}: low should be numeric"
                assert isinstance(volume, (int, float)), f"{pattern_name} kline {i}: volume should be numeric"

    def test_get_pattern_klines_utility_function(self):
        """Test the utility function for getting patterns by name"""
        # Test valid patterns
        valid_patterns = [
            'bearish_engulfing', 'bullish_engulfing', 'hammer', 
            'bearish_shooting_star', 'standard_doji', 'morning_star'
        ]
        
        for pattern in valid_patterns:
            klines = get_pattern_klines(pattern)
            assert len(klines) == 100
            assert all(len(kline) == 6 for kline in klines)
        
        # Test invalid pattern (should return default)
        invalid_klines = get_pattern_klines('invalid_pattern')
        assert len(invalid_klines) == 100
        # Should be same as bearish_engulfing default
        bearish_klines = get_pattern_klines('bearish_engulfing')
        assert invalid_klines == bearish_klines