# Notification latency plan — 11 October 2026

Status: first latency fixes implemented and validated on 11 October 2026; see the implementation record below. Existing Redis Streams, PostgreSQL inbox/outbox and Firebase remain in use. No synthetic pushes were sent to real devices.

The objective is to remove avoidable delay between a received market trigger and a push submission. Price targets are evaluated from quotes; pattern alerts are evaluated from finalized candles. Waiting for a 1h candle to close is part of the pattern contract, not push latency.

## Evidence

Read-only audit: `logs/notification-latency-audit-20261011.json`. Latest 500 pattern outbox records span 7–9 October; price records span 26 September–9 October. These include earlier code, interruptions and recovery. They do not isolate today's steady-state performance. Acceptance timings below do not measure phone display.

| Observation | Evidence and implication |
|---|---|
| Explicit pattern scheduling waits | `SCHEDULE_GRACE=30`; initial and repeating partial-publication checks use 30 seconds. Complete batches may publish earlier. This creates avoidable delay before any push is attempted. |
| Slow database round trips | Three read-only `SELECT 1` measurements were 0.616/0.684/0.748 seconds. Pattern claim, permission check, device lookup, receipts and completion use separate transactions. Multiple round trips amplify latency. |
| Push calls are not the whole delay | Ten existing price log samples: median Firebase call 1.821 seconds, median claim 1.756 seconds, median finish bookkeeping 1.795 seconds. Small historical sample, not an SLA. |
| Pattern delays occur upstream | Among delivered historical records, median candle-close→inbox was 436.734 seconds for crypto and 299.760 seconds for Forex; inbox→outbox 24.872/5.604 seconds; outbox→recorded acceptance 13.172/12.815 seconds. Different stage medians must not be added as an end-to-end percentile. |
| Price samples are limited | One accepted Forex price alert took 8.894 seconds from quote timestamp to outbox and 4.458 seconds to recorded completion. Two further Forex price rows remain pending in the audit. Check expiry and worker availability rather than interpreting these as healthy delivery. |
| Limited send parallelism | Local launcher runs two pattern send lanes and two price send lanes. Pattern device sends are sequential within each alert. This constrains burst throughput. |
| Existing iPhone priority | Both senders already specify APNs alert priority 10. Price messages additionally carry time-sensitive interruption level; pattern messages currently do not. |

## Implementation order

1. **Establish one timing trail.** Reuse event/notification IDs; record source time, backend receipt, detection completion, publication, inbox acceptance, outbox creation, send start, provider acceptance and bookkeeping completion. Report p50/p95 and expiry/failure counts by market and alert type. Measure controlled phone receipt separately. Do not rename provider acceptance as device delivery.

2. **Remove the largest pattern waits.** Schedule a symbol when its finalized candle is persisted, using the existing queue and dispatch fences. Prioritize symbols with active watches while preserving fair progress for the rest. Coalesce publication of completed outcomes over a short bounded interval instead of waiting for the next 30-second watchdog; retain the watchdog for recovery. Never emit an unconfirmed candle, replay a recovered baseline as a new signal, or let history backfill hold ready symbols up. If this requires extensive lifecycle changes, first reduce the publication delay and measure that smaller change independently. The 30-second scheduling grace should only shrink where finalized-data readiness is proven.

3. **Shorten the database path and drain work promptly.** Bring pattern device lookup and receipt handling in line with the price path's existing combined operations. Keep latest consent checks, leases and per-device receipts. Drain available inbox/fan-out work in bounded batches without a mandatory one-second sleep after every successful pass. Use the existing queue as a wake-up hint if needed; the durable outbox remains authoritative and a timed fallback prevents missed wakeups.

4. **Increase throughput conservatively.** Make send concurrency configurable and benchmark two versus four/eight lanes. Size the connection pool with the chosen concurrency. Group claims/receipt writes only where measurements justify it; process per-device results individually. Separate fresh alerts from slow retries, respect Retry-After, add jitter, and keep freshness expiry. Do not wait for a batch to fill before sending a lone alert. Do not bypass delivery consent or drop brief Forex price crossings.

5. **Run the launch service continuously near its database.** Local Mac sleep and Internet/WAN latency cannot be fixed by queue code. Confirm deployment and database regions, benchmark round trips from a proposed small hosting instance, and present its cost before provisioning. Keep resource caps. This is an operational deployment step, not a new notification architecture.

6. **Verify phone behavior and decide whether more is needed.** Retain APNs priority 10. Consider time-sensitive presentation for explicitly requested pattern alerts, consistent with price alerts and user settings. Test foreground, background and locked iPhone behavior, including Focus. FCM-to-direct-APNs migration is a later controlled comparison only if provider-stage measurements justify it.

## Acceptance and stop conditions

Initial engineering targets under the agreed healthy launch load: p95 backend-received price quote→FCM acceptance under 5 seconds; p95 finalized ready candle received→FCM acceptance under 10 seconds. Also report source timestamp/candle-close→receipt delay so provider lag cannot be hidden. These are targets to validate, not present capabilities or guaranteed phone display times.

Replay recorded Forex quotes/candles locally while markets are closed. Verify crossing-and-retreat, next-close pattern detection, one-time/repeat watches, cancellation, market-scope changes, duplicate delivery attempts and restart recovery. Exercise 1, 100 and 1,000-recipient bursts and report the observed completion curve and sustainable rate; larger bursts may require additional bounded workers. Controlled phone tests are separate from load tests, which use a fake push adapter. Recheck natural open-market triggers afterward. Stop increasing complexity once the measured launch targets are met.

No Kafka, Kubernetes, second broker, database migration, mass test pushes or unbounded concurrency is proposed.

## Research informing the choices

- [YouTube notification guidance](https://support.google.com/youtube/answer/7389684?hl=en): YouTube says sending all notifications typically takes 10–20 minutes. A fast notification to one subscriber does not prove simultaneous delivery to millions; its proprietary architecture cannot be inferred from that observation.
- [Firebase topic messaging](https://firebase.google.com/docs/cloud-messaging/topic-messaging): topics favor throughput; Firebase recommends registration-token targeting for fast delivery to individual devices or small groups. Keep direct targeting for personal trading alerts.
- [Firebase scaling guidance](https://firebase.google.com/docs/cloud-messaging/scale-fcm): bounded traffic, gradual ramp-up, backoff and jitter prevent retry congestion. Candle-close alerts naturally coincide with busy quarter-hour boundaries; do not intentionally postpone urgent alerts by minutes to smooth them.
- [FCM architecture](https://firebase.google.com/docs/cloud-messaging/fcm-architecture): iPhone delivery already travels through APNs. FCM is not a substitute for Apple's final transport.
- [APNs request headers](https://developer.apple.com/library/archive/documentation/NetworkingInternet/Conceptual/RemoteNotificationsPG/CommunicatingwithAPNs.html): priority 10 requests immediate alert delivery and is already configured.
- [Firebase delivery metrics](https://firebase.google.com/docs/cloud-messaging/understand-delivery): message acceptance and device delivery are different stages. iOS instrumentation has platform limitations; a push-service response alone is not evidence of a visible notification.
- [Apple time-sensitive notifications](https://developer.apple.com/documentation/usernotifications/unnotificationinterruptionlevel/timesensitive): presentation can bypass some Focus/summary controls when permitted by the user; it does not accelerate our backend.


## Implementation record — 11 October 2026

- Scheduling grace now matches the engine's 5-second closed-candle guard (previously 30 seconds). Missing or unfinalized windows still enter recovery; they cannot produce a signal.
- Partial publication checks run every 5 seconds, previously 30 seconds (180 seconds in the small-universe dispatch path). Completed symbols remain independent of missing-history repair.
- The local launcher reserves one worker for `scanner_control`. Preparation goes to `scanner_ingestion`; previously queued preparation jobs reroute before doing I/O. Candle persistence and history preparation cannot occupy the publication worker.
- A later completed symbol at the same cutoff can now emit its first new detection if it has a consecutive prior observation. Already observed symbols, corrections and replay remain silent. New version-2 batch envelopes identify each publication fragment while event IDs remain stable by scope/cutoff/pattern. Version-1 queued batches are still accepted. No database migration is needed. Restart inbox consumers with the updated producers; old consumers do not understand version 2. The detector fingerprint includes lifecycle code, so the first observation after this update intentionally establishes a fresh baseline.
- The inbox prepares notifications after the first committed stream batch rather than after up to 32 batches, preserves round-robin fairness, handles up to 100 events per fan-out transaction instead of 16, and sleeps only when idle or after errors.
- The final consent/lease check also fetches device destinations and existing receipts. This removes the separate destination-read transaction. Transaction settings share one round trip. Immediate per-device receipts, expiry, cancellation and lease fences remain intact.
- Local pattern delivery defaults to four lanes instead of two; `SCANNER_DELIVERY_CONCURRENCY` overrides this within the existing CLI's 1–16 bound. The connection pool follows concurrency.
- Logs separate publication age, inbox database time, claim, consent, Firebase acceptance and receipt/completion time. Acceptance is not phone display.

Validation: 116 focused unit checks pass. The disposable Redis/Postgres/worker suite validates Forex and crypto late same-close detections, replay, once/repeat watches, cancellation, lease recovery and ownership. The 2,000-recipient fan-out result is recorded in `logs/pattern-latency-runtime-20261011.json`; it measures local enqueueing, not 2,000 phone deliveries. No new message broker or historical database copy was added.

Remaining acceptance work: measure new natural triggers with the backend continuously running, including Forex when it resumes, and correlate FCM acceptance with phone appearance. This update does not certify the proposed p95 targets. Cold history, provider lag, downtime, and hosting/database network latency remain separately visible causes; an unavailable candle is not treated as ready merely to send faster. Symbol-coverage completion is unchanged.
