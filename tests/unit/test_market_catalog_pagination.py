import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock
from infrastructure.database.supabase.markets_repo import MarketRepository
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity


def item(symbol='EURUSD', **extra):
    return dict(symbol=symbol,source='massive',market_type='forex',base_asset='EUR',quote_asset='USD',display_name='EUR / USD', **extra)


def test_active_catalog_reads_beyond_postgrest_page_limit():
    repo=object.__new__(MarketRepository)
    repo.market_instruments='market_instruments'
    query=Mock()
    for method in ('select','eq','order','range'):
        getattr(query,method).return_value=query
    query.execute.side_effect=[NS(data=[item(str(n)) for n in range(1000)]), NS(data=[item('XAUUSD')])]
    repo.client=NS(table=Mock(return_value=query))
    rows=asyncio.run(repo.get_active_instruments())
    assert len(rows)==1001 and rows[-1].symbol=='XAUUSD'
    assert [c.args for c in query.range.call_args_list]==[(0,999),(1000,1999)]


def test_catalog_upserts_do_not_replace_primary_keys_or_send_mixed_null_ids():
    repo=object.__new__(MarketRepository); repo.market_instruments='market_instruments'
    query=Mock();query.upsert.return_value=query
    repo.client=NS(table=Mock(return_value=query))
    rows=[MarketInstrumentEntity(**item()),MarketInstrumentEntity(**item('XAUUSD',id='existing-id'))]
    assert asyncio.run(repo.upsert_instruments(rows))
    sent=query.upsert.call_args.args[0]
    assert all('id' not in row and 'price' not in row for row in sent)
