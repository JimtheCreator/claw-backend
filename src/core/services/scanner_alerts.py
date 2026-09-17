"""Shared event inbox, set-based fan-out and bounded notification delivery."""
import asyncio


class PermanentDeliveryError(Exception):
    pass


async def consume_events(stream, repository, consumer, *, count=10):
    accepted = 0
    for identifier, batch in await stream.read(consumer, count=count):
        await repository.accept_batch(batch, stream_id=identifier)
        # A failed DB transaction or a process death before this point leaves the
        # stream entry pending. Replaying the committed batch is idempotent.
        await stream.acknowledge(identifier)
        accepted += 1
    return accepted


async def deliver_one(repository, sender):
    delivery = await repository.claim_delivery()
    if delivery is None:
        return None
    if delivery['status'] != 'sending':
        return delivery['status']
    if not await repository.delivery_allowed(delivery):
        await repository.finish_delivery(delivery, cancelled=True)
        return 'cancelled'
    try:
        # The lease is 120s. Sender adapters must bound their I/O below this.
        provider_id = await asyncio.wait_for(sender.send(delivery), timeout=30)
    except PermanentDeliveryError:
        await repository.finish_delivery(delivery, error='invalid_destination', permanent=True)
        return 'failed'
    except Exception:
        await repository.finish_delivery(delivery, error='delivery_unavailable')
        return 'retry'
    accepted = await repository.finish_delivery(delivery, provider_id=provider_id)
    return 'delivered' if accepted else 'lease_lost'
