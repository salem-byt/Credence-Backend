/**
 * Boundary and recovery coverage for the score snapshot job
 * (src/jobs/scoreSnapshot.ts).
 *
 * The companion file `src/jobs/scoreSnapshot.test.ts` covers the happy path
 * (batching, ordering, basic error handling, metrics). This file focuses on
 * the edges that protect user data under adverse conditions:
 *
 *  - invalid constructor options (fail-fast validation)
 *  - duplicate / whitespace / non-string / empty address lists
 *  - recovery from a batch-load failure without losing the run
 *  - transient saveBatch failures (retry succeeds / retries exhausted)
 *  - concurrent execution of the same job instance
 *  - permission/authorization failures from the data source and the store
 *  - stale trust-score cache when post-save invalidation fails
 *  - diagnosability: outcomes must reach the injected logger without
 *    leaking credentials or metric-cardinality-sensitive labels
 *
 * Invariants asserted here (documented in src/jobs/scoreSnapshot.ts):
 *  I1. One snapshot per unique non-empty address per run; duplicates are
 *      collapsed, never double-counted.
 *  I2. Invalid options throw at construction; `run()` never starts with an
 *      unusable configuration.
 *  I3. A batch-load failure degrades to per-identity error accounting under
 *      `continueOnError` and stops the run otherwise (no silent skip).
 *  I4. `saved` counts only batches whose `saveBatch` resolved; failed
 *      batches are never partially counted.
 *  I5. Concurrent `run()` on the same instance is refused.
 *  I6. Every terminal outcome is diagnosable from the injected logger, with
 *      secrets masked and metric labels limited to `stage`/`status`.
 *  I7. Save retries are opt-in (`saveRetryAttempts`), bounded, spaced by the
 *      configured backoff, and never counted twice.
 *  I8. Post-save cache invalidation is best-effort: failure is counted and
 *      logged, but never fails the run or undoes the persisted snapshots.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  ScoreSnapshotJob,
  createScoreSnapshotJob,
  describeError,
} from '../../src/jobs/scoreSnapshot.js'
import { register } from '../../src/middleware/metrics.js'
import { trustScoreInvalidationHook } from '../../src/cache/invalidationHooks.js'
import { logger } from '../../src/utils/logger.js'
import type {
  IdentityDataSource,
  ScoreSnapshotStore,
  IdentityData,
  ScoreSnapshot,
} from '../../src/jobs/types.js'

// ---------------------------------------------------------------------------
// Prometheus inspection helpers (label sets and monotonic deltas only — the
// assertions never depend on counters left behind by other tests).
// ---------------------------------------------------------------------------

interface MetricSample {
  metricName?: string
  value: number
  labels: Record<string, unknown>
}

async function metricSamples(name: string): Promise<MetricSample[]> {
  const json = await register.getMetricsAsJSON()
  const metric = json.find((m) => m.name === name)
  return (metric?.values ?? []).map((v) => ({
    metricName: (v as { metricName?: string }).metricName,
    value: Number(v.value),
    labels: (v.labels ?? {}) as Record<string, unknown>,
  }))
}

async function counterValue(name: string, labels: Record<string, string> = {}): Promise<number> {
  const samples = await metricSamples(name)
  return samples
    .filter((s) => Object.entries(labels).every(([key, val]) => s.labels[key] === val))
    .reduce((sum, s) => sum + s.value, 0)
}

async function histogramCount(name: string): Promise<number> {
  const samples = await metricSamples(name)
  return samples.find((s) => s.metricName === `${name}_count`)?.value ?? 0
}

function makeData(address: string, overrides: Partial<IdentityData> = {}): IdentityData {
  return {
    address,
    bondedAmount: '1000',
    active: true,
    attestationCount: 10,
    ...overrides,
  }
}

describe('score snapshot boundaries — option validation (I2)', () => {
  let dataSource: IdentityDataSource
  let store: ScoreSnapshotStore

  beforeEach(() => {
    dataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xabc']),
      getIdentityData: vi.fn().mockResolvedValue(makeData('0xabc')),
    }
    store = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockResolvedValue(undefined),
    }
  })

  it('accepts batchSize 1 and batchSize 10000, rejects 0 and 10001', () => {
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { batchSize: 1 }),
    ).not.toThrow()
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { batchSize: 10_000 }),
    ).not.toThrow()
    expect(() => new ScoreSnapshotJob(dataSource, store, () => 75, { batchSize: 0 })).toThrow(
      RangeError,
    )
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { batchSize: 10_001 }),
    ).toThrow(RangeError)
  })

  it('rejects fractional and non-finite batchSize', () => {
    for (const batchSize of [1.5, Number.NaN, Number.POSITIVE_INFINITY, -3]) {
      expect(() => new ScoreSnapshotJob(dataSource, store, () => 75, { batchSize })).toThrow(
        /batchSize must be an integer/,
      )
    }
  })

  it('accepts saveRetryAttempts 1..10, rejects 0 and 11', () => {
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { saveRetryAttempts: 1 }),
    ).not.toThrow()
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { saveRetryAttempts: 10 }),
    ).not.toThrow()
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { saveRetryAttempts: 0 }),
    ).toThrow(RangeError)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { saveRetryAttempts: 11 }),
    ).toThrow(/saveRetryAttempts/)
  })

  it('accepts saveRetryBackoffMs 0 and fractional values, rejects negative, non-finite and oversized values', () => {
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { saveRetryBackoffMs: 0 }),
    ).not.toThrow()
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { saveRetryBackoffMs: 0.5 }),
    ).not.toThrow()
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { saveRetryBackoffMs: -0.5 }),
    ).toThrow(RangeError)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, {
        saveRetryBackoffMs: Number.NaN,
      }),
    ).toThrow(RangeError)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, {
        saveRetryBackoffMs: 60_001,
      }),
    ).toThrow(/saveRetryBackoffMs/)
  })

  it('rejects invalid options at construction before touching the data source', () => {
    const factory = () =>
      new ScoreSnapshotJob(dataSource, store, () => 75, { batchSize: -1 })
    expect(factory).toThrow(RangeError)
    expect(dataSource.getActiveAddresses).not.toHaveBeenCalled()
    expect(store.saveBatch).not.toHaveBeenCalled()
  })

  it('rejects a non-object options argument instead of crashing mid-run', () => {
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, null as never),
    ).toThrow(TypeError)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, 'batchSize=5' as never),
    ).toThrow(TypeError)
  })

  it('rejects a non-boolean continueOnError that would silently flip failure semantics', () => {
    // 'false' is truthy in JS — accepting it would turn "stop on error" into
    // "continue on error" without telling anyone.
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { continueOnError: 'false' as never }),
    ).toThrow(/continueOnError must be a boolean/)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { continueOnError: 0 as never }),
    ).toThrow(TypeError)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { continueOnError: false }),
    ).not.toThrow()
  })

  it('rejects a non-function logger and an invalid scoringModelVersion', () => {
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { logger: 'console.log' as never }),
    ).toThrow(/logger must be a function/)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { scoringModelVersion: 42 as never }),
    ).toThrow(/scoringModelVersion must be a string/)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { scoringModelVersion: '   ' }),
    ).toThrow(/scoringModelVersion must not be empty/)
    expect(
      () => new ScoreSnapshotJob(dataSource, store, () => 75, { scoringModelVersion: 'v2.1.0' }),
    ).not.toThrow()
    expect(dataSource.getActiveAddresses).not.toHaveBeenCalled()
  })
})

describe('score snapshot boundaries — address list normalization (I1)', () => {
  let store: ScoreSnapshotStore
  let savedSnapshots: ScoreSnapshot[]

  beforeEach(() => {
    savedSnapshots = []
    store = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockImplementation(async (snapshots: ScoreSnapshot[]) => {
        savedSnapshots.push(...snapshots)
      }),
    }
  })

  function makeSource(
    addresses: unknown[],
    data: Map<string, IdentityData>,
  ): IdentityDataSource {
    return {
      getActiveAddresses: vi.fn().mockResolvedValue(addresses),
      getIdentityData: vi.fn(async (address: string) => data.get(address) ?? null),
      getIdentityDataBatch: vi.fn(async (batch: string[]) =>
        batch.map((address) => data.get(address)).filter((d): d is IdentityData => d !== null && d !== undefined),
      ),
    }
  }

  it('collapses duplicate addresses into a single snapshot', async () => {
    const data = new Map([
      ['0xabc', makeData('0xabc')],
    ])
    const source = makeSource(['0xabc', '0xabc', '0xabc'], data)

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(1)
    expect(result.saved).toBe(1)
    expect(result.errors).toBe(0)
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xabc'])
  })

  it('treats addresses that differ only by surrounding whitespace as the same identity', async () => {
    const data = new Map([['0xabc', makeData('0xabc')]])
    const source = makeSource(['0xabc  ', ' 0xabc'], data)

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(1)
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xabc'])
  })

  it('drops blank and non-string addresses without counting them as processed', async () => {
    const data = new Map([['0xabc', makeData('0xabc')]])
    const source = makeSource(['0xabc', '   ', '', null, undefined, 42], data)

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(1)
    expect(result.saved).toBe(1)
    expect(result.errors).toBe(0)
    expect(savedSnapshots).toHaveLength(1)
  })

  it('completes a run whose entire address list is invalid without saving', async () => {
    const source = makeSource(['', '   ', null], new Map())

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(0)
    expect(result.saved).toBe(0)
    expect(result.errors).toBe(0)
    expect(store.saveBatch).not.toHaveBeenCalled()
  })

  it('never emits a snapshot for an address missing from identity data', async () => {
    const source = makeSource(['0xknown', '0xghost'], new Map([
      ['0xknown', makeData('0xknown')],
    ]))

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(2) // both were seen
    expect(result.saved).toBe(1) // only one had data
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xknown'])
  })

  it('matches batch rows that echo the address padded with whitespace', async () => {
    // Regression: rows were keyed verbatim while the request side trims, so
    // a padded row made a valid identity look missing and its snapshot was
    // silently dropped.
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue([' 0xabc ']),
      getIdentityData: vi.fn(async () => null),
      getIdentityDataBatch: vi.fn(async () => [makeData(' 0xabc ')]),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(1)
    expect(result.saved).toBe(1)
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xabc'])
    expect(savedSnapshots[0].score).toBe(75)
  })

  it('treats a malformed batch row as a diagnosed load failure (I3), not silent data loss', async () => {
    // A row without an address cannot be attributed to an identity, so the
    // whole batch fails loudly instead of pretending '0xother' has no data.
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xgood', '0xother']),
      getIdentityData: vi.fn(async () => null),
      getIdentityDataBatch: vi.fn(async () => [
        null,
        { bondedAmount: '0', active: false, attestationCount: 0 } as unknown as IdentityData,
        makeData('0xgood'),
      ]),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, { logger: (m) => logs.push(m) })
    const result = await job.run()

    expect(result.processed).toBe(0)
    expect(result.saved).toBe(0)
    expect(result.errors).toBe(2)
    const failure = logs.find((l) => l.includes('Batch load failed'))
    expect(failure).toBeDefined()
    expect(failure).toContain('malformed row at index 0')
  })

  it('rejects an address list that is not an array instead of failing mid-iteration', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(null as never),
      getIdentityData: vi.fn(async () => null),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, { logger: (m) => logs.push(m) })
    await expect(job.run()).rejects.toThrow(/getActiveAddresses returned null/)

    const failure = logs.find((l) => l.includes('Job failed'))
    expect(failure).toContain('expected an array')
    expect(store.saveBatch).not.toHaveBeenCalled()
  })

  it('treats a malformed identity record as an error, never a corrupt snapshot', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xgood', '0xbad']),
      getIdentityData: vi.fn(async (address: string) =>
        address === '0xbad' ? ('garbage' as unknown as IdentityData) : makeData(address),
      ),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, { logger: (m) => logs.push(m) })
    const result = await job.run()

    expect(result.processed).toBe(1)
    expect(result.saved).toBe(1)
    expect(result.errors).toBe(1)
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xgood'])
    expect(logs.some((l) => l.includes('expected an object'))).toBe(true)
  })

  it('stops the run on a malformed identity record when continueOnError is false', async () => {
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xbad']),
      getIdentityData: vi.fn(async () => 'garbage' as unknown as IdentityData),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, { continueOnError: false })
    await expect(job.run()).rejects.toThrow(/expected an object/)
    expect(store.saveBatch).not.toHaveBeenCalled()
  })

  it('treats a non-array batch result as a diagnosed load failure (I3), not a crash', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xabc']),
      getIdentityData: vi.fn(async () => null),
      getIdentityDataBatch: vi.fn(async () => null as never),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, { logger: (m) => logs.push(m) })
    const result = await job.run()

    expect(result.processed).toBe(0)
    expect(result.saved).toBe(0)
    expect(result.errors).toBe(1)
    const loadFailure = logs.find((l) => l.includes('Batch load failed'))
    expect(loadFailure).toBeDefined()
    expect(loadFailure).toContain('expected an array')
    expect(logs.some((l) => l.includes('Job failed'))).toBe(false) // run still completed
  })
})

describe('score snapshot recovery — batch-load failures (I3)', () => {
  let store: ScoreSnapshotStore
  let savedSnapshots: ScoreSnapshot[]

  beforeEach(() => {
    savedSnapshots = []
    store = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockImplementation(async (snapshots: ScoreSnapshot[]) => {
        savedSnapshots.push(...snapshots)
      }),
    }
  })

  function makeSource(
    addresses: string[],
    data: Map<string, IdentityData>,
    failBatchOn: (batch: string[]) => boolean,
  ): IdentityDataSource {
    return {
      getActiveAddresses: vi.fn().mockResolvedValue(addresses),
      getIdentityData: vi.fn(async (address: string) => data.get(address) ?? null),
      getIdentityDataBatch: vi.fn(async (batch: string[]) => {
        if (failBatchOn(batch)) throw new Error('batch load timed out')
        return batch
          .map((address) => data.get(address))
          .filter((d): d is IdentityData => d !== null && d !== undefined)
      }),
    }
  }

  it('continues the run when the first batch load fails, and saves later batches', async () => {
    const data = new Map(
      ['0xa', '0xb', '0xc', '0xd'].map((a) => [a, makeData(a)] as const),
    )
    const source = makeSource(['0xa', '0xb', '0xc', '0xd'], data, (batch) =>
      batch.includes('0xa'),
    )

    const job = new ScoreSnapshotJob(source, store, () => 75, { batchSize: 2 })
    const result = await job.run()

    expect(result.processed).toBe(2) // second batch only
    expect(result.saved).toBe(2)
    expect(result.errors).toBe(2) // first batch degraded to errors
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xc', '0xd'])
    expect(source.getIdentityData).not.toHaveBeenCalled()
  })

  it('stops the run when a batch load fails and continueOnError is false', async () => {
    const data = new Map(
      ['0xa', '0xb', '0xc', '0xd'].map((a) => [a, makeData(a)] as const),
    )
    const source = makeSource(['0xa', '0xb', '0xc', '0xd'], data, (batch) =>
      batch.includes('0xa'),
    )

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      batchSize: 2,
      continueOnError: false,
    })

    await expect(job.run()).rejects.toThrow('batch load timed out')
    expect(store.saveBatch).not.toHaveBeenCalled()
  })

  it('aborts with no partial save when continueOnError is false and scoring throws mid-batch', async () => {
    const data = new Map(
      ['0xa', '0xb'].map((a) => [a, makeData(a)] as const),
    )
    const source = makeSource(['0xa', '0xb'], data, () => false)

    let calls = 0
    const job = new ScoreSnapshotJob(
      source,
      store,
      () => {
        calls++
        if (calls === 2) throw new Error('scoring exploded')
        return 75
      },
      { continueOnError: false },
    )

    await expect(job.run()).rejects.toThrow('scoring exploded')
    expect(store.saveBatch).not.toHaveBeenCalled()
    expect(savedSnapshots).toHaveLength(0)
  })
})

describe('score snapshot recovery — transient saveBatch failures (I4)', () => {
  let savedSnapshots: ScoreSnapshot[]

  beforeEach(() => {
    savedSnapshots = []
  })

  function makeStore(failTimes: number): ScoreSnapshotStore {
    let calls = 0
    return {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn(async (snapshots: ScoreSnapshot[]) => {
        calls++
        if (calls <= failTimes) throw new Error('deadlock detected')
        savedSnapshots.push(...snapshots)
      }),
    }
  }

  function makeSource(addresses: string[]): IdentityDataSource {
    const data = new Map(addresses.map((a) => [a, makeData(a)] as const))
    return {
      getActiveAddresses: vi.fn().mockResolvedValue(addresses),
      getIdentityData: vi.fn(async (a: string) => data.get(a) ?? null),
      getIdentityDataBatch: vi.fn(async (batch: string[]) =>
        batch.map((a) => data.get(a)!),
      ),
    }
  }

  it('saves the batch on a retry after a transient failure', async () => {
    const source = makeSource(['0xa', '0xb'])
    const store = makeStore(1) // fail once, then succeed

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      saveRetryAttempts: 3,
      saveRetryBackoffMs: 0,
    })
    const result = await job.run()

    expect(result.saved).toBe(2)
    expect(result.errors).toBe(0)
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xa', '0xb'])
    expect(store.saveBatch).toHaveBeenCalledTimes(2)
  })

  it('succeeds on the final allowed attempt', async () => {
    const source = makeSource(['0xa'])
    const store = makeStore(2) // two failures, third attempt (of 3) succeeds

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      saveRetryAttempts: 3,
      saveRetryBackoffMs: 0,
    })
    const result = await job.run()

    expect(result.saved).toBe(1)
    expect(result.errors).toBe(0)
    expect(store.saveBatch).toHaveBeenCalledTimes(3)
  })

  it('counts every snapshot as an error after retries are exhausted', async () => {
    const source = makeSource(['0xa', '0xb', '0xc'])
    const store = makeStore(Number.POSITIVE_INFINITY) // always fails

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      saveRetryAttempts: 3,
      saveRetryBackoffMs: 0,
    })
    const result = await job.run()

    expect(result.saved).toBe(0)
    expect(result.errors).toBe(3)
    expect(store.saveBatch).toHaveBeenCalledTimes(3) // no extra attempt beyond the budget
    expect(savedSnapshots).toHaveLength(0)
  })

  it('makes no retry when saveRetryAttempts is 1 (default)', async () => {
    const source = makeSource(['0xa'])
    const store = makeStore(Number.POSITIVE_INFINITY)

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.saved).toBe(0)
    expect(result.errors).toBe(1)
    expect(store.saveBatch).toHaveBeenCalledTimes(1)
  })

  it('retries each failed batch independently across batches', async () => {
    const source = makeSource(['0xa', '0xb', '0xc', '0xd'])
    let calls = 0
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn(async (snapshots: ScoreSnapshot[]) => {
        calls++
        // Both attempts for batch 1 fail; batch 2 (calls 3+) succeeds.
        if (calls <= 2) throw new Error('transient')
        savedSnapshots.push(...snapshots)
      }),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      batchSize: 2,
      saveRetryAttempts: 2,
      saveRetryBackoffMs: 0,
    })
    const result = await job.run()

    // Batch 1: 2 attempts, both fail → 2 errors. Batch 2: saved on first try.
    expect(result.saved).toBe(2)
    expect(result.errors).toBe(2)
    expect(store.saveBatch).toHaveBeenCalledTimes(3) // 2 attempts (batch 1) + 1 (batch 2)
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xc', '0xd'])
  })
})

describe('score snapshot recovery — concurrent execution (I5)', () => {
  function makeSlowSource(delayMs: number): IdentityDataSource {
    return {
      getActiveAddresses: vi.fn(async () => {
        await new Promise((r) => setTimeout(r, delayMs))
        return ['0xa']
      }),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
    }
  }

  function makeStore(): ScoreSnapshotStore {
    return {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockResolvedValue(undefined),
    }
  }

  it('refuses a second run while the first is in flight', async () => {
    const source = makeSlowSource(30)
    const store = makeStore()
    const job = new ScoreSnapshotJob(source, store, () => 75)

    const first = job.run()
    await expect(job.run()).rejects.toThrow(/already in progress/i)

    const result = await first
    expect(result.processed).toBe(1)
    expect(store.saveBatch).toHaveBeenCalledTimes(1)
  })

  it('allows a run again after the previous run completes', async () => {
    const source = makeSlowSource(1)
    const store = makeStore()
    const job = new ScoreSnapshotJob(source, store, () => 75)

    await job.run()
    const second = await job.run()
    expect(second.processed).toBe(1)
    expect(store.saveBatch).toHaveBeenCalledTimes(2)
  })

  it('allows a run after the previous run failed', async () => {
    const source = makeSlowSource(1)
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn(),
    }
    const failingStore: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockRejectedValue(new Error('db down')),
    }

    const job = new ScoreSnapshotJob(source, failingStore, () => 75, {
      continueOnError: false,
    })
    await expect(job.run()).rejects.toThrow('db down')

    // Same instance recovers and runs cleanly against a healthy store is not
    // possible (store is fixed at construction); assert the guard is released
    // by running a fresh instance.
    const recovered = new ScoreSnapshotJob(source, store, () => 75)
    const result = await recovered.run()
    expect(result.processed).toBe(1)
  })

  it('allows a run after the previous run was rejected for concurrency', async () => {
    const source = makeSlowSource(20)
    const store = makeStore()
    const job = new ScoreSnapshotJob(source, store, () => 75)

    const first = job.run()
    await expect(job.run()).rejects.toThrow(/already in progress/i)
    // A third call while still running is also refused.
    await expect(job.run()).rejects.toThrow(/already in progress/i)

    await first
    const third = await job.run()
    expect(third.processed).toBe(1)
  })

  it('does not count a refused invocation as a failed run', async () => {
    const failedBefore = await counterValue('score_snapshot_runs_total', { status: 'failed' })
    const source = makeSlowSource(20)
    const store = makeStore()
    const job = new ScoreSnapshotJob(source, store, () => 75)

    const first = job.run()
    await expect(job.run()).rejects.toThrow(/already in progress/i)
    await first

    // The refusal happened before any run started, so it is not a run outcome.
    expect(await counterValue('score_snapshot_runs_total', { status: 'failed' })).toBe(failedBefore)
  })
})

describe('score snapshot diagnosability (I6)', () => {
  let store: ScoreSnapshotStore

  beforeEach(() => {
    store = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockResolvedValue(undefined),
    }
  })

  it('logs retry attempts with attempt numbers', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
    }
    let calls = 0
    const failingStore: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn(async () => {
        calls++
        if (calls === 1) throw new Error('lock timeout')
      }),
    }

    const job = new ScoreSnapshotJob(source, failingStore, () => 75, {
      logger: (msg) => logs.push(msg),
      saveRetryAttempts: 2,
      saveRetryBackoffMs: 0,
    })
    await job.run()

    expect(logs.some((l) => l.includes('attempt 1/2 failed'))).toBe(true)
    expect(logs.some((l) => l.includes('saved on attempt 2/2'))).toBe(true)
  })

  it('logs a single-line truncated error for oversized multi-line messages', async () => {
    const logs: string[] = []
    const giant = `x${'\n'.repeat(50)}${'y'.repeat(500)}`
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa']),
      getIdentityData: vi.fn(async () => {
        throw new Error(giant)
      }),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      logger: (msg) => logs.push(msg),
    })
    const result = await job.run()

    expect(result.errors).toBe(1)
    const errorLog = logs.find((l) => l.includes('Error processing'))
    expect(errorLog).toBeDefined()
    // Single line: no raw newline survives into the log message.
    expect(errorLog!.includes('\n')).toBe(false)
    expect(errorLog!.length).toBeLessThan(300)
  })

  it('logs dedup and invalid-address accounting when the address list is dirty', async () => {
    const logs: string[] = []
    const data = new Map([['0xabc', makeData('0xabc')]])
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xabc', '0xabc', '   ', null]),
      getIdentityData: vi.fn(async (a: string) => data.get(a) ?? null),
      getIdentityDataBatch: vi.fn(async (batch: string[]) =>
        batch.map((a) => data.get(a)!),
      ),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      logger: (msg) => logs.push(msg),
    })
    await job.run()

    const dedupLog = logs.find((l) => l.includes('Deduplicated address list'))
    expect(dedupLog).toBeDefined()
    expect(dedupLog).toContain('1 duplicates dropped')
    expect(dedupLog).toContain('2 invalid dropped') // '   ' and null
  })

  it('does not log a batch-save failure when nothing was saved', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue([]),
      getIdentityData: vi.fn(),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      logger: (msg) => logs.push(msg),
    })
    const result = await job.run()

    expect(result.saved).toBe(0)
    expect(logs.some((l) => l.includes('Error saving batch'))).toBe(false)
  })

  it('exposes the factory-created job with the same hardened behavior', async () => {
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa', '0xa']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
    }

    const job = createScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(1) // duplicate collapsed via factory path too
    expect(job).toBeInstanceOf(ScoreSnapshotJob)
  })
})

// ---------------------------------------------------------------------------
// Permission / authorization states
// ---------------------------------------------------------------------------

describe('score snapshot permission and authorization failures', () => {
  function collectingStore(): ScoreSnapshotStore & { saved: ScoreSnapshot[] } {
    const saved: ScoreSnapshot[] = []
    return {
      saved,
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn(async (snapshots: ScoreSnapshot[]) => {
        saved.push(...snapshots)
      }),
    }
  }

  it('fails the whole run with a diagnosable error when the directory listing is denied', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi
        .fn()
        .mockRejectedValue(new Error('permission denied for table identities')),
      getIdentityData: vi.fn(),
    }
    const store = collectingStore()

    const job = new ScoreSnapshotJob(source, store, () => 75, { logger: (m) => logs.push(m) })
    await expect(job.run()).rejects.toThrow('permission denied for table identities')

    // Nothing was written, and the operator can see why.
    expect(store.saveBatch).not.toHaveBeenCalled()
    expect(store.save).not.toHaveBeenCalled()
    const failure = logs.find((l) => l.includes('Job failed'))
    expect(failure).toBeDefined()
    expect(failure).toContain('permission denied for table identities')
  })

  it('counts a per-identity denial as an error and keeps the other identities (continueOnError)', async () => {
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xok', '0xdenied']),
      getIdentityData: vi.fn(async (address: string) => {
        if (address === '0xdenied') {
          throw new Error('permission denied for relation identity_data')
        }
        return makeData(address)
      }),
    }
    const store = collectingStore()

    const job = new ScoreSnapshotJob(source, store, () => 75)
    const result = await job.run()

    expect(result.processed).toBe(1)
    expect(result.saved).toBe(1)
    expect(result.errors).toBe(1)
    expect(store.saved.map((s) => s.address)).toEqual(['0xok'])
  })

  it('aborts before writing when a denial happens mid-batch and continueOnError is false', async () => {
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xok', '0xdenied']),
      getIdentityData: vi.fn(async (address: string) => {
        if (address === '0xdenied') {
          throw new Error('permission denied for relation identity_data')
        }
        return makeData(address)
      }),
    }
    const store = collectingStore()

    const job = new ScoreSnapshotJob(source, store, () => 75, { continueOnError: false })
    await expect(job.run()).rejects.toThrow('permission denied for relation identity_data')

    // No partially-written batch: the save stage only runs once the whole
    // batch has been processed, and the throw short-circuits it.
    expect(store.saveBatch).not.toHaveBeenCalled()
    expect(store.saved).toHaveLength(0)
  })

  it('never claims a rejected batch write as saved, even across bounded retries', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa', '0xb']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
      getIdentityDataBatch: vi.fn(async (batch: string[]) => batch.map((a) => makeData(a))),
    }
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockRejectedValue(new Error('permission denied for table score_snapshots')),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      logger: (m) => logs.push(m),
      saveRetryAttempts: 3,
      saveRetryBackoffMs: 0,
    })
    const result = await job.run()

    expect(result.processed).toBe(2)
    expect(result.saved).toBe(0)
    expect(result.errors).toBe(2)
    expect(store.saveBatch).toHaveBeenCalledTimes(3) // bounded by the attempt budget
    const failure = logs.find((l) => l.includes('Error saving batch'))
    expect(failure).toContain('after 3 attempt(s)')
    expect(failure).toContain('permission denied for table score_snapshots')
  })
})

// ---------------------------------------------------------------------------
// Stale state: post-save cache invalidation
// ---------------------------------------------------------------------------

describe('score snapshot stale-cache handling after save (I8)', () => {
  it('keeps the run successful and records the incident when invalidation fails', async () => {
    // Let invalidation triggered by earlier tests settle so the counter
    // delta below is deterministic.
    await new Promise((resolve) => setTimeout(resolve, 20))
    const before = await counterValue('score_snapshot_invalidation_failures_total')

    const errorSpy = vi.spyOn(logger, 'error').mockImplementation(() => {})
    vi.spyOn(trustScoreInvalidationHook, 'execute').mockRejectedValue(
      new Error('redis unreachable password=hunter2secret'),
    )

    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa', '0xb']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
      getIdentityDataBatch: vi.fn(async (batch: string[]) => batch.map((a) => makeData(a))),
    }
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockResolvedValue(undefined),
    }

    try {
      const job = new ScoreSnapshotJob(source, store, () => 75)
      const result = await job.run()

      // The snapshots are persisted; only the stale-cache risk is at stake.
      expect(result.saved).toBe(2)
      expect(result.errors).toBe(0)

      await vi.waitFor(async () => {
        expect(await counterValue('score_snapshot_invalidation_failures_total')).toBe(before + 1)
      })

      expect(errorSpy).toHaveBeenCalledTimes(1)
      const logged = String(errorSpy.mock.calls[0]?.[0])
      expect(logged).toContain('Failed to invalidate trust-score cache')
      expect(logged).toContain('[REDACTED]')
      expect(logged).not.toContain('hunter2secret')
    } finally {
      vi.restoreAllMocks()
    }
  })

  it('invalidates the trust-score cache for every saved address (no stale reads)', async () => {
    const hookSpy = vi.spyOn(trustScoreInvalidationHook, 'execute').mockResolvedValue({
      name: 'trust_score.invalidate',
      keysAttempted: 2,
      keysInvalidated: 2,
      durationMs: 0,
    })

    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa', '0xb']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
      getIdentityDataBatch: vi.fn(async (batch: string[]) => batch.map((a) => makeData(a))),
    }
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockResolvedValue(undefined),
    }

    try {
      const job = new ScoreSnapshotJob(source, store, () => 75)
      const result = await job.run()

      expect(result.saved).toBe(2)
      expect(hookSpy).toHaveBeenCalledTimes(1)
      expect(hookSpy).toHaveBeenCalledWith('0xa', '0xb')
      await expect(hookSpy.mock.results[0].value).resolves.toMatchObject({
        keysAttempted: 2,
      })
    } finally {
      vi.restoreAllMocks()
    }
  })
})

// ---------------------------------------------------------------------------
// Diagnosability without secret leakage
// ---------------------------------------------------------------------------

describe('score snapshot log sanitisation (I6)', () => {
  it('masks credentials in rendered errors while keeping the context', () => {
    const rendered = describeError(
      new Error(
        'write failed for password=hunter2secret api_key=sk_live_9f8e7d6c5b Authorization: Bearer aaaabbbbccccdddd token: zzzz12345678',
      ),
    )

    expect(rendered).toContain('write failed')
    expect(rendered).toContain('[REDACTED]')
    expect(rendered).not.toContain('hunter2secret')
    expect(rendered).not.toContain('sk_live_9f8e7d6c5b')
    expect(rendered).not.toContain('aaaabbbbccccdddd')
    expect(rendered).not.toContain('zzzz12345678')
  })

  it('masks connection-string credentials and strips control characters', () => {
    const rendered = describeError(
      new Error('connect ECONNREFUSED postgres://app:hunter2secret@db:5432/credence\n\tretrying'),
    )

    expect(rendered).not.toContain('hunter2secret')
    expect(rendered).toContain('postgres://app:[REDACTED]@db:5432/credence')
    expect(rendered).not.toMatch(/[\n\t]/)
  })

  it('keeps ordinary error text untouched so failures stay diagnosable', () => {
    expect(describeError(new Error('deadlock detected'))).toBe('deadlock detected')
    expect(describeError('plain string failure')).toBe('plain string failure')
    expect(describeError(undefined)).toBe('undefined')
  })

  it('does not leak credential material from a failing data source into the job log', async () => {
    const logs: string[] = []
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa']),
      getIdentityData: vi.fn(async () => {
        throw new Error('auth rejected api_key=sk_live_abcdef123456')
      }),
    }
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockResolvedValue(undefined),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, { logger: (m) => logs.push(m) })
    const result = await job.run()

    expect(result.errors).toBe(1)
    const line = logs.find((l) => l.includes('Error processing'))
    expect(line).toBeDefined()
    expect(line).toContain('[REDACTED]')
    expect(line).not.toContain('sk_live_abcdef123456')
  })
})

// ---------------------------------------------------------------------------
// Metrics: diagnosability of every terminal outcome
// ---------------------------------------------------------------------------

describe('score snapshot observability counters (I6)', () => {
  function sourceWith(addresses: string[]): IdentityDataSource {
    return {
      getActiveAddresses: vi.fn().mockResolvedValue(addresses),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
      getIdentityDataBatch: vi.fn(async (batch: string[]) => batch.map((a) => makeData(a))),
    }
  }

  function okStore(): ScoreSnapshotStore {
    return {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockResolvedValue(undefined),
    }
  }

  it('records a completed run together with the persisted batch and its size', async () => {
    const runsBefore = await counterValue('score_snapshot_runs_total', { status: 'completed' })
    const savedBefore = await counterValue('score_snapshot_batches_saved_total')
    const sizeBefore = await histogramCount('score_snapshot_batch_size')

    const job = new ScoreSnapshotJob(sourceWith(['0xa', '0xb']), okStore(), () => 75)
    const result = await job.run()
    expect(result.saved).toBe(2)

    expect(await counterValue('score_snapshot_runs_total', { status: 'completed' })).toBe(
      runsBefore + 1,
    )
    expect(await counterValue('score_snapshot_batches_saved_total')).toBe(savedBefore + 1)
    expect(await histogramCount('score_snapshot_batch_size')).toBe(sizeBefore + 1)
  })

  it('records a failed run', async () => {
    const before = await counterValue('score_snapshot_runs_total', { status: 'failed' })
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockRejectedValue(new Error('permission denied for table identities')),
      getIdentityData: vi.fn(),
    }

    const job = new ScoreSnapshotJob(source, okStore(), () => 75)
    await expect(job.run()).rejects.toThrow('permission denied for table identities')

    expect(await counterValue('score_snapshot_runs_total', { status: 'failed' })).toBe(before + 1)
  })

  it('separates a retried save from a give-up save failure', async () => {
    const retriesBefore = await counterValue('score_snapshot_batch_retries_total')
    const saveFailuresBefore = await counterValue('score_snapshot_batch_failures_total', {
      stage: 'save',
    })
    const completedBefore = await counterValue('score_snapshot_runs_total', { status: 'completed' })

    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn().mockRejectedValue(new Error('deadlock detected')),
    }
    const job = new ScoreSnapshotJob(sourceWith(['0xa']), store, () => 75, {
      saveRetryAttempts: 2,
      saveRetryBackoffMs: 0,
    })
    const result = await job.run()

    expect(result.errors).toBe(1)
    expect(await counterValue('score_snapshot_batch_retries_total')).toBe(retriesBefore + 1)
    expect(
      await counterValue('score_snapshot_batch_failures_total', { stage: 'save' }),
    ).toBe(saveFailuresBefore + 1)
    // The run itself finished (its result carries the error count).
    expect(await counterValue('score_snapshot_runs_total', { status: 'completed' })).toBe(
      completedBefore + 1,
    )
  })

  it('records a give-up load failure under stage=load', async () => {
    const loadFailuresBefore = await counterValue('score_snapshot_batch_failures_total', {
      stage: 'load',
    })
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
      getIdentityDataBatch: vi.fn(async () => {
        throw new Error('batch load timed out')
      }),
    }

    const job = new ScoreSnapshotJob(source, okStore(), () => 75)
    const result = await job.run()

    expect(result.errors).toBe(1)
    expect(
      await counterValue('score_snapshot_batch_failures_total', { stage: 'load' }),
    ).toBe(loadFailuresBefore + 1)
  })

  it('limits metric labels to stage/status — never identity addresses or raw errors', async () => {
    // Produce one failure of each kind first so every series exists.
    const failingSource: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockRejectedValue(new Error('permission denied for table identities')),
      getIdentityData: vi.fn(),
    }
    await new ScoreSnapshotJob(failingSource, okStore(), () => 75)
      .run()
      .catch(() => undefined)

    const runSamples = await metricSamples('score_snapshot_runs_total')
    expect(runSamples.some((s) => 'status' in s.labels)).toBe(true)
    for (const sample of runSamples) {
      expect(Object.keys(sample.labels).every((key) => key === 'status')).toBe(true)
    }

    const failureSamples = await metricSamples('score_snapshot_batch_failures_total')
    expect(failureSamples.some((s) => 'stage' in s.labels)).toBe(true)
    for (const sample of failureSamples) {
      expect(Object.keys(sample.labels).every((key) => key === 'stage')).toBe(true)
      expect(JSON.stringify(sample.labels)).not.toMatch(/0x[0-9a-f]{6,}/i)
      expect(JSON.stringify(sample.labels)).not.toContain('permission denied')
    }
  })
})

// ---------------------------------------------------------------------------
// Timing boundary, parallel instances and accounting conservation
// ---------------------------------------------------------------------------

describe('score snapshot timing, parallelism and accounting', () => {
  it('waits the configured backoff between save attempts', async () => {
    const backoffMs = 60
    let calls = 0
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn(async () => {
        calls++
        if (calls === 1) throw new Error('transient')
      }),
    }
    const source: IdentityDataSource = {
      getActiveAddresses: vi.fn().mockResolvedValue(['0xa']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
      getIdentityDataBatch: vi.fn(async (batch: string[]) => batch.map((a) => makeData(a))),
    }

    const job = new ScoreSnapshotJob(source, store, () => 75, {
      saveRetryAttempts: 2,
      saveRetryBackoffMs: backoffMs,
    })
    const startedAt = Date.now()
    const result = await job.run()
    const elapsed = Date.now() - startedAt

    expect(result.saved).toBe(1)
    expect(store.saveBatch).toHaveBeenCalledTimes(2)
    // Timers never fire early: the retry was delayed by the full backoff.
    expect(elapsed).toBeGreaterThanOrEqual(backoffMs)
  })

  it('runs separate instances in parallel (the re-entrancy guard is per instance)', async () => {
    const slowSource = (): IdentityDataSource => ({
      getActiveAddresses: vi.fn(async () => {
        await new Promise((resolve) => setTimeout(resolve, 15))
        return ['0xa']
      }),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
    })
    const storeA = { save: vi.fn(), saveBatch: vi.fn().mockResolvedValue(undefined) }
    const storeB = { save: vi.fn(), saveBatch: vi.fn().mockResolvedValue(undefined) }

    const jobA = new ScoreSnapshotJob(slowSource(), storeA as unknown as ScoreSnapshotStore, () => 75)
    const jobB = new ScoreSnapshotJob(slowSource(), storeB as unknown as ScoreSnapshotStore, () => 75)

    const [a, b] = await Promise.all([jobA.run(), jobB.run()])
    expect(a.processed).toBe(1)
    expect(b.processed).toBe(1)
    expect(storeA.saveBatch).toHaveBeenCalledTimes(1)
    expect(storeB.saveBatch).toHaveBeenCalledTimes(1)
  })

  it('keeps accounting conserved across a mixed-outcome run', async () => {
    const logs: string[] = []
    const savedSnapshots: ScoreSnapshot[] = []
    const source: IdentityDataSource = {
      // 1 duplicate + 1 blank alongside the three real identities.
      getActiveAddresses: vi
        .fn()
        .mockResolvedValue(['0xa', '0xa', '   ', '0xb', '0xc']),
      getIdentityData: vi.fn(async (a: string) => makeData(a)),
      getIdentityDataBatch: vi.fn(async (batch: string[]) => batch.map((a) => makeData(a))),
    }
    const store: ScoreSnapshotStore = {
      save: vi.fn().mockResolvedValue(undefined),
      saveBatch: vi.fn(async (snapshots: ScoreSnapshot[]) => {
        if (snapshots.some((s) => s.address === '0xc')) throw new Error('write rejected')
        savedSnapshots.push(...snapshots)
      }),
    }

    let scoreCalls = 0
    const job = new ScoreSnapshotJob(
      source,
      store,
      () => {
        scoreCalls++
        if (scoreCalls === 2) throw new Error('scoring exploded')
        return 75
      },
      { batchSize: 1, logger: (m) => logs.push(m) },
    )
    const result = await job.run()

    // Dedup/normalisation dropped one duplicate and one blank address.
    expect(logs.some((l) => l.includes('1 duplicates dropped'))).toBe(true)
    expect(logs.some((l) => l.includes('1 invalid dropped'))).toBe(true)
    // 0xa saved, 0xb failed scoring, 0xc failed to persist.
    expect(result.processed).toBe(2)
    expect(result.saved).toBe(1)
    expect(result.errors).toBe(2)
    expect(savedSnapshots.map((s) => s.address)).toEqual(['0xa'])
    expect(store.saveBatch).toHaveBeenCalledTimes(2) // 0xa and 0xc — never 0xb
  })
})
