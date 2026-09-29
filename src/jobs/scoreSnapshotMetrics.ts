/**
 * Prometheus metrics for the score snapshot job
 * (src/jobs/scoreSnapshot.ts).
 *
 * Counters and a histogram only — increments/observations are safe against
 * concurrent job runs and never carry per-identity state.
 *
 * Labels deliberately exclude identity addresses so that scraping the
 * metrics endpoint can never leak identity data.
 */

import { Counter, Histogram } from 'prom-client'
import { register } from '../middleware/metrics.js'

export const scoreSnapshotRunsTotal = new Counter({
  name: 'score_snapshot_runs_total',
  help: 'Score snapshot job runs by terminal status (completed | failed)',
  labelNames: ['status'],
  registers: [register],
})

export const scoreSnapshotBatchesSavedTotal = new Counter({
  name: 'score_snapshot_batches_saved_total',
  help: 'Score snapshot batches persisted successfully',
  registers: [register],
})

export const scoreSnapshotBatchRetriesTotal = new Counter({
  name: 'score_snapshot_batch_retries_total',
  help: 'Retries performed after a transient score snapshot saveBatch failure',
  registers: [register],
})

export const scoreSnapshotBatchFailuresTotal = new Counter({
  name: 'score_snapshot_batch_failures_total',
  help: 'Score snapshot batch failures that gave up (after retries), by stage',
  labelNames: ['stage'],
  registers: [register],
})

export const scoreSnapshotInvalidationFailuresTotal = new Counter({
  name: 'score_snapshot_invalidation_failures_total',
  help: 'Post-save trust-score cache invalidation failures (stale-cache risk signal)',
  registers: [register],
})

export const scoreSnapshotBatchSize = new Histogram({
  name: 'score_snapshot_batch_size',
  help: 'Number of snapshots per successfully persisted batch',
  buckets: [1, 5, 10, 25, 50, 100, 250, 500, 1000],
  registers: [register],
})

/**
 * Record one successfully persisted batch.
 *
 * Kept as a helper so call sites cannot observe the size histogram without
 * also incrementing the saved counter (the two are always consistent).
 */
export function recordScoreSnapshotBatch(size: number): void {
  scoreSnapshotBatchesSavedTotal.inc()
  scoreSnapshotBatchSize.observe(size)
}
