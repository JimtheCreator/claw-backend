"""
Universal pattern notification body formatter for all pattern types.
All notifications will include only the pattern name and the start/end time window.
"""

from typing import Dict, Any
from datetime import datetime

def format_pattern_notification_body(event_data: Dict[str, Any]) -> str:
    pattern_name = (event_data.get('pattern_name') or '').replace('_', ' ').title()
    start_time = event_data.get('start_time', 'N/A')
    end_time = event_data.get('end_time', 'N/A')
    price = event_data.get('price')
    
    # Convert Unix timestamps to readable format
    def format_timestamp(timestamp):
        if timestamp is None or timestamp == 'N/A':
            return 'N/A'
        try:
            # Handle both string and int timestamps
            if isinstance(timestamp, str):
                timestamp = int(timestamp)
            # Convert milliseconds to seconds if needed
            if timestamp > 1e10:  # Likely milliseconds
                timestamp = timestamp / 1000
            dt = datetime.fromtimestamp(timestamp)
            # Format as "Monday 28th June at 04:00AM"
            day_name = dt.strftime('%A')  # Monday
            day = dt.strftime('%d')  # 28
            # Add ordinal suffix
            if day.endswith('1') and day != '11':
                suffix = 'st'
            elif day.endswith('2') and day != '12':
                suffix = 'nd'
            elif day.endswith('3') and day != '13':
                suffix = 'rd'
            else:
                suffix = 'th'
            month_name = dt.strftime('%B')  # June
            time_12hr = dt.strftime('%I:%M%p')  # 04:00AM
            return f"{day_name} {day}{suffix} {month_name} at {time_12hr}"
        except (ValueError, TypeError, OSError):
            return str(timestamp)
    
    formatted_start = format_timestamp(start_time)
    formatted_end = format_timestamp(end_time)
    
    body = f"{pattern_name} detected"
    
    if price:
        body += f" at US${price}"
    
    body += f". Pattern window: {formatted_start} to {formatted_end}."
    
    return body 