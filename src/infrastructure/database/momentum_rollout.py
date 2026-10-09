"""Reversible momentum-history mirror. SQLite remains the default and rollback."""
from contextlib import contextmanager
import fcntl
import logging
import os
import time

from .questdb.momentum import QuestMomentum

log = logging.getLogger(__name__)


def same_frame(left, right):
    if list(left.columns)!=list(right.columns) or len(left)!=len(right):
        return False
    return bool(((left.reset_index(drop=True)==right.reset_index(drop=True)) |
                 (left.reset_index(drop=True).isna() & right.reset_index(drop=True).isna())).all().all())


@contextmanager
def mirror_lock(path, timeout=10):
    """Serialize mirror writers across Celery processes to preserve corrections."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.quest-mirror.lock').open('a') as lock:
        deadline=time.monotonic()+timeout
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>=deadline:
                    raise TimeoutError('Momentum mirror writer busy') from None
                time.sleep(.05)
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


class MomentumRollout:
    def __init__(self, legacy, quest, mode):
        if mode not in {'dual','shadow','quest'}:
            raise ValueError('Invalid MOMENTUM_CANDLE_STORE mode')
        self.legacy,self.quest,self.mode=legacy,quest,mode

    def put(self,symbol,interval,frame):
        with mirror_lock(self.legacy.path):
            saved=self.legacy.put_and_snapshot(symbol,interval,frame)
            # Commit primary first. A target failure propagates; replay takes
            # another authoritative snapshot rather than guessing taker values.
            self.quest.put(symbol,interval,saved)

    def get(self,symbol,interval,end,count):
        if self.mode=='quest':
            return self.quest.get(symbol,interval,end,count)
        result=self.legacy.get(symbol,interval,end,count)
        if self.mode=='shadow':
            try:
                if not same_frame(result,self.quest.get(symbol,interval,end,count)):
                    log.warning('Momentum history shadow mismatch')
            except Exception:
                log.warning('Momentum history shadow unavailable')
        return result


def momentum_store(legacy_factory, **kwargs):
    mode=os.getenv('MOMENTUM_CANDLE_STORE','sqlite')
    if mode == 'quest_only':
        from .questdb.momentum import QuestMomentumCache
        return QuestMomentumCache('binance', 'spot')
    if mode not in {'sqlite','dual','shadow','quest'}:
        raise ValueError('Invalid MOMENTUM_CANDLE_STORE mode')
    legacy=legacy_factory(**kwargs)
    if mode=='sqlite':
        return legacy
    return MomentumRollout(legacy,QuestMomentum('binance','spot'),mode)
