# src/presentation/api/routes/market_data.py
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.domain.entities.MarketDataEntity import MarketDataEntity
from core.domain.entities.MarketDataEntity import MarketDataResponse, DeleteResponse
from core.use_cases.market.market_data import fetch_crypto_data_paginated
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
import json
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.services.crypto_list import search_cryptos, downsample_sparkline
from common.logger import logger
from fastapi.responses import StreamingResponse

from core.domain.entities.MarketDataEntity import MarketDataResponse, DeleteResponse
from core.use_cases.market.market_data import delete_market_data
from datetime import datetime, timezone
from typing import Optional
import websockets

router = APIRouter(tags=["Plan Usage Limits"])

