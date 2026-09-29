/**
 * Unit coverage for the score snapshot Prometheus instruments
 * (src/jobs/scoreSnapshotMetrics.ts).
 *
 * These are the counters the job relies on for diagnosability (invariants
 * I6/I8 in src/jobs/scoreSnapshot.ts): every terminal outcome must be
 * observable, and label sets must stay bounded so scraping the metrics
 * endpoint can never leak identity data.
 */

import { beforeEach, describe, expect, it } from 'vitest'
import { register } from '../middleware/metrics.js'
import {
  scoreSnapshotBatchesSavedTotal,
  scoreSnapshotBatchRetriesTotal,
  scoreSnapshotBatchFailuresTotal,
  scoreSnapshotInvalidationFailuresTotal,
  scoreSnapshotRunsTotal,
  scoreSnapshotBatchSize,
  recordScoreSnapshotBatch,
} from './scoreSnapshotMetrics.js'

async function samplesFor(name: string): Promise<
  Array<{ metricName?: string; value: number; labels: Record<string, unknown> }>
> {
  const json = await register.getMetricsAsJSON()
  const metric = json.find((m) => m.name === name)
  return (metric?.values ?? []).map((v) => ({
    metricName: (v as { metricName?: string }).metricName,
    value: Number(v.value),
    labels: (v.labels ?? {}) as Record<string, unknown>,
  }))
}

async function valueOf(name: string, labels: Record<string, string> = {}): Promise<number> {
  const samples = await samplesFor(name)
  return samples
    .filter((s) => Object.entries(labels).every(([key, val]) => s.labels[key] === val))
    .reduce((sum, s) => sum + s.value, 0)
}

async function histogramStat(name: string, suffix: '_count' | '_sum'): Promise<number> {
  const samples = await samplesFor(name)
  return samples.find((s) => s.metricName === `${name}${suffix}`)?.value ?? 0
}

describe('scoreSnapshotMetrics', () => {
  beforeEach(() => {
    scoreSnapshotBatchesSavedTotal.reset()
    scoreSnapshotBatchRetriesTotal.reset()
    scoreSnapshotBatchFailuresTotal.reset()
    scoreSnapshotInvalidationFailuresTotal.reset()
    scoreSnapshotRunsTotal.reset()
    scoreSnapshotBatchSize.reset()
  })

  it('registers every instrument under its documented scrape name', async () => {
    const json = await register.getMetricsAsJSON()
    const names = json.map((m) => m.name)

    expect(names).toEqual(
      expect.arrayContaining([
        'score_snapshot_batches_saved_total',
        'score_snapshot_batch_retries_total',
        'score_snapshot_batch_failures_total',
        'score_snapshot_invalidation_failures_total',
        'score_snapshot_runs_total',
        'score_snapshot_batch_size',
      ]),
    )
  })

  it('recordScoreSnapshotBatch advances the saved counter and observes the batch size exactly once', async () => {
    recordScoreSnapshotBatch(3)

    expect(await valueOf('score_snapshot_batches_saved_total')).toBe(1)
    expect(await histogramStat('score_snapshot_batch_size', '_count')).toBe(1)
    expect(await histogramStat('score_snapshot_batch_size', '_sum')).toBe(3)
  })

  it('accumulates across batches so partial runs stay visible', async () => {
    recordScoreSnapshotBatch(2)
    recordScoreSnapshotBatch(5)

    expect(await valueOf('score_snapshot_batches_saved_total')).toBe(2)
    expect(await histogramStat('score_snapshot_batch_size', '_count')).toBe(2)
    expect(await histogramStat('score_snapshot_batch_size', '_sum')).toBe(7)
  })

  it('counts retries separately from terminal batch failures', async () => {
    scoreSnapshotBatchRetriesTotal.inc()
    scoreSnapshotBatchFailuresTotal.inc({ stage: 'save' })

    expect(await valueOf('score_snapshot_batch_retries_total')).toBe(1)
    expect(await valueOf('score_snapshot_batch_failures_total', { stage: 'save' })).toBe(1)
    expect(await valueOf('score_snapshot_batch_failures_total', { stage: 'load' })).toBe(0)
  })

  it('keeps failure labels bounded to a single known stage value', async () => {
    scoreSnapshotBatchFailuresTotal.inc({ stage: 'load' })

    const samples = await samplesFor('score_snapshot_batch_failures_total')
    expect(samples).toHaveLength(1)
    expect(Object.keys(samples[0].labels)).toEqual(['stage'])
    expect(samples[0].labels.stage).toBe('load')
  })

  it('keeps run labels bounded to a single known status value', async () => {
    scoreSnapshotRunsTotal.inc({ status: 'failed' })

    const samples = await samplesFor('score_snapshot_runs_total')
    expect(samples).toHaveLength(1)
    expect(Object.keys(samples[0].labels)).toEqual(['status'])
    expect(samples[0].labels.status).toBe('failed')
  })

  it('exposes the stale-cache (invalidation failure) risk as its own counter', async () => {
    expect(await valueOf('score_snapshot_invalidation_failures_total')).toBe(0)

    scoreSnapshotInvalidationFailuresTotal.inc()

    expect(await valueOf('score_snapshot_invalidation_failures_total')).toBe(1)
  })
})
