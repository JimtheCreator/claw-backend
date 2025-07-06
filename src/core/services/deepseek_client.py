import os
import requests
from typing import Dict, Any
from common.logger import logger

class DeepSeekClient:
    """
    A universal client for interacting with the DeepSeek LLM via the OpenRouter API.
    """
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_R1_OPENROUTER_API_KEY")
        self.api_url = os.getenv("DEEPSEEK_R1_OPENROUTER_API_URL")
        self.model = "deepseek/deepseek-r1-0528:free"

        if not self.api_key or not self.api_url:
            raise ValueError("DeepSeek API Key and URL must be set in environment variables.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _format_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """
        Formats the raw analysis data into a detailed prompt for the LLM.
        """
        # Example: Extracting key information to build a structured prompt
        # This should be tailored to get the best summary from your data structure.
        context = analysis_data.get("market_context", {})
        patterns = analysis_data.get("patterns", [])
        
        # Safely format numeric values
        volatility = context.get('volatility')
        trend_strength = context.get('trend_strength')
        
        volatility_str = f"{volatility:.4f}" if isinstance(volatility, (int, float)) else str(volatility or 'N/A')
        trend_strength_str = f"{trend_strength:.4f}" if isinstance(trend_strength, (int, float)) else str(trend_strength or 'N/A')
        
        prompt = (
            "You are an expert financial analyst. Based on the following market analysis data, "
            "provide a clear and concise summary for a retail trader. "
            "Explain the current market structure, key patterns identified, and potential future scenarios. "
            "Keep the tone objective and informative.\n\n"
            f"**Market Context:**\n"
            f"- Scenario: {context.get('scenario', 'N/A')}\n"
            f"- Volatility: {volatility_str}\n"
            f"- Trend Strength: {trend_strength_str}\n"
            f"- Identified Patterns: {len(patterns)}\n\n"
            "**Key Findings:**\n"
            f"{analysis_data.get('context', {}).get('demand_supply_summary', 'No summary available.')}\n\n"
            "Please provide your summary."
        )
        return prompt

    def generate_summary(self, analysis_data: Dict[str, Any]) -> str:
        """
        Sends analysis data to the DeepSeek LLM and returns a market summary.

        Args:
            analysis_data: The raw JSON output from the market analysis.

        Returns:
            A string containing the LLM-generated summary, or an error message.
        """
        prompt = self._format_prompt(analysis_data)
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            logger.info("Sending analysis to DeepSeek LLM for summary.")
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            summary = result['choices'][0]['message']['content']
            logger.info("Successfully received summary from DeepSeek LLM.")
            return summary.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return "Market analysis completed. AI summary unavailable due to service issues."
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing DeepSeek API response: {e}")
            return "Market analysis completed. AI summary unavailable due to parsing issues."
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek client: {e}")
            return "Market analysis completed. AI summary unavailable."