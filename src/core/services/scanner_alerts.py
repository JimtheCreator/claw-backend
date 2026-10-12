"""Shared event inbox, set-based fan-out and bounded notification delivery."""
import asyncio
import logging
import time
from datetime import datetime, timezone
from core.scanner.watches import EventConflict
from infrastructure.database.redis.scanner_events import MalformedBatch

log = logging.getLogger(__name__)


class PermanentDeliveryError(Exception):
    pass


class PushConfigurationError(RuntimeError):
    """Provider credentials need operator attention; the device token is valid."""
    pass


async def consume_events(stream, repository, consumer, *, count=10):
    accepted = 0
    for identifier, batch in await stream.read(consumer, count=count):
        try:
            if isinstance(batch, MalformedBatch):
                raise ValueError("Malformed event JSON")
            started = time.monotonic()
            await repository.accept_batch(batch, stream_id=identifier)
            cutoff = batch.get('data_as_of')
            age = ((datetime.now(timezone.utc) - datetime.fromisoformat(cutoff)).total_seconds()
                   if cutoff else -1)
            log.info('Pattern inbox timing: batch=%s database=%.3fs since_close=%.3fs events=%s',
                     batch.get('batch_id'), time.monotonic()-started, age, len(batch.get('events', [])))
        except (ValueError, KeyError, TypeError, EventConflict) as exc:
            await stream.quarantine(identifier, batch, type(exc).__name__)
            log.error("Scanner event quarantined (%s); operator review required", type(exc).__name__)
            continue
        # A failed DB transaction or a process death before this point leaves the
        # stream entry pending. Replaying the committed batch is idempotent.
        await stream.acknowledge(identifier)
        accepted += 1
    return accepted


async def deliver_one(repository, sender):
    started = time.monotonic()
    delivery = await repository.claim_delivery()
    claimed = time.monotonic()
    if delivery is None:
        return None
    if delivery['status'] != 'sending':
        return delivery['status']
    if not await repository.delivery_allowed(delivery):
        await repository.finish_delivery(delivery, cancelled=True)
        return 'cancelled'
    checked = time.monotonic()
    try:
        # The lease is 120s. Sender adapters must bound their I/O below this.
        provider_id = await asyncio.wait_for(sender.send(delivery), timeout=30)
    except PermanentDeliveryError:
        await repository.finish_delivery(delivery, error='invalid_destination', permanent=True)
        return 'failed'
    except PushConfigurationError:
        log.error('Scanner push credentials rejected: check Firebase APNs key, Apple team, '
                  'bundle ID and development/production environment. Device registration retained.')
        await repository.finish_delivery(delivery, error='push_credentials_invalid')
        return 'retry'
    except Exception as exc:
        # Exception messages can contain tokens, request payloads or credentials.
        log.warning('Scanner delivery attempt failed (%s); retry scheduled', type(exc).__name__)
        await repository.finish_delivery(delivery, error='delivery_unavailable')
        return 'retry'
    sent = time.monotonic()
    accepted = await repository.finish_delivery(delivery, provider_id=provider_id)
    created = delivery.get('created_at')
    queue_age = (datetime.now(timezone.utc) - created).total_seconds() if created else -1
    log.info('Pattern delivery timing: notification=%s claim=%.3fs consent=%.3fs '
             'send_and_receipts=%.3fs finish=%.3fs outbox_age=%.3fs accepted=%s',
             delivery.get('id'), claimed-started, checked-claimed, sent-checked,
             time.monotonic()-sent, queue_age, accepted)
    return 'delivered' if accepted else 'lease_lost'
