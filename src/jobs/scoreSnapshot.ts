import type {
  IdentityDataSource,
  IdentityData,
  ScoreSnapshotStore,
  ScoreComputer,
  SnapshotJobResult,
  ScoreSnapshot,
} from './types.js'
import { trustScoreInvalidationHook } from '../cache/invalidationHooks.js'
import { logger } from '../utils/logger.js'
import { loadConfig } from '../config/index.js'
import {
  scoreSnapshotBatchRetriesTotal,
  scoreSnapshotBatchFailuresTotal,
  scoreSnapshotInvalidationFailuresTotal,
  scoreSnapshotRunsTotal,
  recordScoreSnapshotBatch,
} from './scoreSnapshotMetrics.js'

/**
 * Options for score snapshot job.
 */
export interface SnapshotJobOptions {
  /** Batch size for processing identities (default: 100). */
  batchSize?: number
  /** Whether to continue on errors (default: true). */
  continueOnError?: boolean
  /** Logger function for progress/errors. */
  logger?: (message: string) => void
  /** Scoring model version to record in snapshots (optional). */
  scoringModelVersion?: string
  /**
   * Number of attempts made to persist each batch (default: 1 — no retry).
   * Must be a positive integer. Attempts are spaced by `saveRetryBackoffMs`.
   * Values above 1 make persistence at-least-once: only enable them when
   * `saveBatch` is idempotent for a snapshot (upsert), as per invariant I7.
   */
  saveRetryAttempts?: number
  /**
   * Fixed delay in milliseconds between `saveBatch` retry attempts
   * (default: 100). Must be a non-negative finite number.
   */
  saveRetryBackoffMs?: number
}

// ---------------------------------------------------------------------------
// Invariants (asserted in tests/jobs/scoreSnapshotBoundaryRecovery.test.ts)
// ---------------------------------------------------------------------------
// I1. Exactly one snapshot is produced per unique, non-empty address per run;
//     duplicate addresses in `getActiveAddresses()` are collapsed and are
//     never counted twice in `processed`/`saved`.
// I2. Invalid constructor options fail fast at construction; `run()` never
//     starts with an unusable configuration.
// I3. A data-source contract violation never aborts the whole run and is
//     never silently skipped: with `continueOnError` it degrades to
//     per-identity error accounting; with `continueOnError: false` it stops
//     the run. Malformed payloads (non-array address list, non-array batch
//     result, malformed batch row, non-object record) surface as diagnosed
//     errors instead of corrupt or missing snapshots.
// I4. `saved` counts only snapshots whose `saveBatch` promise resolved. A
//     transient save failure is retried up to `saveRetryAttempts`; a failed
//     batch is never partially counted.
// I5. Concurrent `run()` invocations on the same instance are refused; the
//     second caller gets a rejected promise and the first run is unaffected.
// I6. Every terminal outcome is diagnosable without leaking secrets: run,
//     retry and per-stage failure counters plus the injected logger record
//     batch indexes, sizes, attempt numbers, and the affected identity
//     address (a public on-chain identifier — required to triage a single
//     identity, and never a credential). Error text is sanitised before it
//     is logged: secrets masked, control characters stripped, truncated to
//     200 chars. Metric labels only ever carry `stage`/`status` — never an
//     identity address or raw error text.
// I7. Persistence retries are opt-in and at-least-once: the default
//     `saveRetryAttempts: 1` never re-invokes a failed write, so enabling
//     retries (`> 1`) is only safe when `saveBatch` is idempotent (upsert on
//     the snapshot primary key). A partially applied batch is therefore
//     either completed by the retry or re-applied identically — never
//     duplicated by this job's own bookkeeping (`saved` advances only once
//     per batch, after the write resolves).
// I8. Post-save cache invalidation is best-effort: a failure never fails the
//     run nor undoes the persisted snapshots, but it is counted
//     (`score_snapshot_invalidation_failures_total`) so a stale trust-score
//     cache stays observable instead of silently serving old scores.
// ---------------------------------------------------------------------------

const MAX_BATCH_SIZE = 10_000
const MAX_SAVE_RETRY_ATTEMPTS = 10
const MAX_SAVE_RETRY_BACKOFF_MS = 60_000

/**
 * Fully validated job options. `scoringModelVersion` is left `undefined`
 * when the caller did not supply one so the constructor can resolve it from
 * the app config lazily (and only then pay for `loadConfig()`).
 */
interface ResolvedJobOptions {
  batchSize: number
  continueOnError: boolean
  saveRetryAttempts: number
  saveRetryBackoffMs: number
  logger: (message: string) => void
  scoringModelVersion: string | undefined
}

/**
 * Validate and coerce constructor options. Throws on invalid input so bad
 * configuration is caught before any data is touched (invariant I2).
 */
function resolveJobOptions(options: SnapshotJobOptions): ResolvedJobOptions {
  if (options === null || typeof options !== 'object') {
    throw new TypeError(`SnapshotJobOptions must be an object, got ${String(options)}`)
  }

  const batchSize = options.batchSize ?? 100
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > MAX_BATCH_SIZE) {
    throw new RangeError(
      `SnapshotJobOptions.batchSize must be an integer between 1 and ${MAX_BATCH_SIZE}, got ${String(batchSize)}`,
    )
  }

  const saveRetryAttempts = options.saveRetryAttempts ?? 1
  if (
    !Number.isInteger(saveRetryAttempts) ||
    saveRetryAttempts < 1 ||
    saveRetryAttempts > MAX_SAVE_RETRY_ATTEMPTS
  ) {
    throw new RangeError(
      `SnapshotJobOptions.saveRetryAttempts must be an integer between 1 and ${MAX_SAVE_RETRY_ATTEMPTS}, got ${String(saveRetryAttempts)}`,
    )
  }

  const saveRetryBackoffMs = options.saveRetryBackoffMs ?? 100
  if (
    typeof saveRetryBackoffMs !== 'number' ||
    !Number.isFinite(saveRetryBackoffMs) ||
    saveRetryBackoffMs < 0 ||
    saveRetryBackoffMs > MAX_SAVE_RETRY_BACKOFF_MS
  ) {
    throw new RangeError(
      `SnapshotJobOptions.saveRetryBackoffMs must be a finite number between 0 and ${MAX_SAVE_RETRY_BACKOFF_MS}, got ${String(saveRetryBackoffMs)}`,
    )
  }

  // A non-boolean `continueOnError` (e.g. the string 'false') would silently
  // coerce to a truthy value and change failure semantics mid-run.
  const continueOnError = options.continueOnError ?? true
  if (typeof continueOnError !== 'boolean') {
    throw new TypeError(
      `SnapshotJobOptions.continueOnError must be a boolean, got ${typeof continueOnError}`,
    )
  }

  const { logger: jobLogger } = options
  if (jobLogger !== undefined && typeof jobLogger !== 'function') {
    throw new TypeError(
      `SnapshotJobOptions.logger must be a function, got ${typeof jobLogger}`,
    )
  }

  const { scoringModelVersion } = options
  if (scoringModelVersion !== undefined) {
    if (typeof scoringModelVersion !== 'string') {
      throw new TypeError(
        `SnapshotJobOptions.scoringModelVersion must be a string, got ${typeof scoringModelVersion}`,
      )
    }
    if (scoringModelVersion.trim().length === 0) {
      throw new TypeError('SnapshotJobOptions.scoringModelVersion must not be empty')
    }
  }

  return {
    batchSize,
    continueOnError,
    saveRetryAttempts,
    saveRetryBackoffMs,
    logger: jobLogger ?? (() => {}),
    scoringModelVersion,
  }
}

/**
 * Normalize an address for snapshot identity: trim surrounding whitespace.
 * Returns `null` for addresses that are empty after trimming so callers can
 * skip (and count) them instead of persisting unusable snapshots.
 */
function normalizeAddress(address: unknown): string | null {
  if (typeof address !== 'string') return null
  const trimmed = address.trim()
  return trimmed.length > 0 ? trimmed : null
}

/**
 * Validate one identity record coming back from the data source.
 *
 * Returns `null` when the identity genuinely has no data — a normal outcome
 * that the caller counts as processed. Throws when the source violates its
 * contract: a malformed record must surface as an error (I3), never become a
 * snapshot whose `bondedAmount`/`attestationCount` are `undefined`.
 */
function requireIdentityData(data: unknown, address: string): IdentityData | null {
  if (data === null || data === undefined) return null
  if (typeof data !== 'object' || Array.isArray(data)) {
    throw new TypeError(
      `identity data for ${address} is ${Array.isArray(data) ? 'an array' : typeof data}, expected an object`,
    )
  }
  return data as IdentityData
}

/**
 * Credential shapes that must never reach log pipelines, even when they
 * ride in on an error message (drivers and HTTP clients happily echo query
 * strings, connection strings and auth headers back).
 *
 * Applied *before* truncation on purpose: truncating first could slice a
 * token in half so the pattern no longer matches and a usable fragment
 * leaks through.
 */
const SECRET_PATTERNS: ReadonlyArray<readonly [RegExp, string]> = [
  // Scheme-prefixed credentials: "Bearer eyJ…", "Basic dXNlcjpwYXNz"
  [/\b(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]{8,}/gi, '$1 [REDACTED]'],
  // Bare JSON Web Tokens
  [/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}/g, '[REDACTED]'],
  // Connection-string credentials: "postgres://user:password@host"
  [/\b([a-z][a-z0-9+.-]*:\/\/[^/\s:@]+):([^\s@]+)@/gi, '$1:[REDACTED]@'],
  // key=value pairs (query strings, connection strings, KV logs)
  [
    /\b(password|passwd|pwd|secret|client_secret|api[_-]?key|apikey|access[_-]?key|private[_-]?key|token|access[_-]?token|auth[_-]?token|authorization|credential|cookie)(\s*=\s*)("?)[^\s&"']+\3/gi,
    '$1$2$3[REDACTED]$3',
  ],
  // key: value pairs (JSON-ish / log-ish forms). Values already masked by a
  // previous pattern are left alone so messages stay readable.
  [
    /\b(password|passwd|pwd|secret|client_secret|api[_-]?key|apikey|access[_-]?key|private[_-]?key|token|access[_-]?token|auth[_-]?token|authorization|credential|cookie)(\s*:\s*)((?!\[REDACTED\])[^\s,;]{6,})/gi,
    '$1$2[REDACTED]',
  ],
]

/**
 * Log-safe error rendering: maps any thrown value to a short single-line
 * string with credentials masked. Long messages are truncated so a runaway
 * error (e.g. a giant SQL blob) cannot bloat logs, and control characters
 * are removed so messages stay single-line and safe for log pipelines.
 */
export function describeError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  // C0 control characters, including tab, so the result is always a single
  // plain line that no log pipeline can be confused or split by.
  const singleLine = raw.replace(/[\r\n\t\x00-\x08\x0b\x0c\x0e-\x1f]+/g, ' ')
  let masked = singleLine
  for (const [pattern, replacement] of SECRET_PATTERNS) {
    masked = masked.replace(pattern, replacement)
  }
  return masked.length > 200 ? `${masked.slice(0, 200)}…` : masked
}

/**
 * Score snapshot job: computes and persists score snapshots for all active identities.
 *
 * Features:
 * - Batch processing for large datasets
 * - Error handling with optional continue-on-error
 * - Progress logging
 * - Performance metrics
 *
 * @example
 * ```typescript
 * const job = new ScoreSnapshotJob(dataSource, store, computeScore)
 * const result = await job.run()
 * console.log(`Processed ${result.processed} identities in ${result.duration}ms`)
 * ```
 */
export class ScoreSnapshotJob {
  private readonly batchSize: number
  private readonly continueOnError: boolean
  private readonly logger: (message: string) => void
  private readonly scoringModelVersion: string
  private readonly saveRetryAttempts: number
  private readonly saveRetryBackoffMs: number
  /** Re-entrancy guard: non-null while a `run()` is in flight (invariant I5). */
  private inFlight: Promise<SnapshotJobResult> | null = null

  constructor(
    private readonly dataSource: IdentityDataSource,
    private readonly store: ScoreSnapshotStore,
    private readonly scoreComputer: ScoreComputer,
    options: SnapshotJobOptions = {}
  ) {
    const resolved = resolveJobOptions(options)
    this.batchSize = resolved.batchSize
    this.continueOnError = resolved.continueOnError
    this.saveRetryAttempts = resolved.saveRetryAttempts
    this.saveRetryBackoffMs = resolved.saveRetryBackoffMs
    this.logger = resolved.logger
    this.scoringModelVersion =
      resolved.scoringModelVersion ?? loadConfig().reputation.scoringModelVersion
  }

  /**
   * Run the snapshot job.
   *
   * Concurrent invocations on the same instance are rejected with an error
   * (invariant I5); run separate instances for parallel execution.
   *
   * @returns Job execution result with metrics
   */
  async run(): Promise<SnapshotJobResult> {
    if (this.inFlight) {
      throw new Error('ScoreSnapshotJob.run() is already in progress on this instance')
    }

    const promise = this.executeRun()
    this.inFlight = promise
    try {
      const result = await promise
      scoreSnapshotRunsTotal.inc({ status: 'completed' })
      return result
    } catch (error) {
      scoreSnapshotRunsTotal.inc({ status: 'failed' })
      throw error
    } finally {
      this.inFlight = null
    }
  }

  private async executeRun(): Promise<SnapshotJobResult> {
    const startTime = new Date().toISOString()
    const startMs = Date.now()

    this.logger('Starting score snapshot job')

    let processed = 0
    let saved = 0
    let errors = 0
    let aggregationDuration = 0

    try {
      const addresses = await this.dataSource.getActiveAddresses()
      if (!Array.isArray(addresses)) {
        throw new TypeError(
          `getActiveAddresses returned ${addresses === null ? 'null' : typeof addresses}, expected an array`,
        )
      }
      this.logger(`Found ${addresses.length} active identities`)

      // I1: collapse duplicate / blank addresses once, up front, so every
      // downstream stage sees a unique, usable identity list.
      const uniqueAddresses: string[] = []
      const seen = new Set<string>()
      let duplicates = 0
      let invalid = 0
      for (const raw of addresses) {
        const address = normalizeAddress(raw)
        if (address === null) {
          invalid++
          continue
        }
        if (seen.has(address)) {
          duplicates++
          continue
        }
        seen.add(address)
        uniqueAddresses.push(address)
      }
      if (duplicates > 0 || invalid > 0) {
        this.logger(
          `Deduplicated address list: ${uniqueAddresses.length} unique, ${duplicates} duplicates dropped, ${invalid} invalid dropped`,
        )
      }

      // Process in batches
      for (let i = 0; i < uniqueAddresses.length; i += this.batchSize) {
        const batch = uniqueAddresses.slice(i, i + this.batchSize)
        const batchNum = Math.floor(i / this.batchSize) + 1
        const totalBatches = Math.ceil(uniqueAddresses.length / this.batchSize)

        this.logger(`Processing batch ${batchNum}/${totalBatches} (${batch.length} identities)`)

        const snapshots: ScoreSnapshot[] = []
        if (typeof this.dataSource.getIdentityDataBatch === 'function') {
          const aggregationStartedAt = Date.now()
          let batchData: Map<string, IdentityData> | null = null
          try {
            batchData = await this.loadBatchData(batch)
          } catch (error) {
            // I3: a batch-load failure must not silently abort the run when
            // continueOnError is set — degrade to per-identity accounting.
            errors += batch.length
            const errorMsg = describeError(error)
            this.logger(
              `Batch load failed for batch ${batchNum}/${totalBatches} (${batch.length} identities): ${errorMsg}`,
            )
            scoreSnapshotBatchFailuresTotal.inc({ stage: 'load' })

            if (!this.continueOnError) {
              throw error
            }
          }
          aggregationDuration += Date.now() - aggregationStartedAt

          if (batchData) {
            for (const address of batch) {
              try {
                const data = requireIdentityData(batchData.get(address) ?? null, address)

                if (!data) {
                  this.logger(`No data found for ${address}`)
                  processed++
                  continue
                }

                const score = this.scoreComputer(data)
                const snapshot: ScoreSnapshot = {
                  address,
                  score,
                  bondedAmount: data.bondedAmount,
                  attestationCount: data.attestationCount,
                  timestamp: new Date().toISOString(),
                  scoringModelVersion: this.scoringModelVersion,
                }

                snapshots.push(snapshot)
                processed++
              } catch (error) {
                errors++
                const errorMsg = describeError(error)
                this.logger(`Error processing ${address}: ${errorMsg}`)

                if (!this.continueOnError) {
                  throw error
                }
              }
            }
          }
        } else {
          for (const address of batch) {
            try {
              const aggregationStartedAt = Date.now()
              const data = requireIdentityData(
                await this.dataSource.getIdentityData(address),
                address,
              )
              aggregationDuration += Date.now() - aggregationStartedAt

              if (!data) {
                this.logger(`No data found for ${address}`)
                processed++
                continue
              }

              const score = this.scoreComputer(data)
              const snapshot: ScoreSnapshot = {
                address,
                score,
                bondedAmount: data.bondedAmount,
                attestationCount: data.attestationCount,
                timestamp: new Date().toISOString(),
                scoringModelVersion: this.scoringModelVersion,
              }

              snapshots.push(snapshot)
              processed++
            } catch (error) {
              errors++
              const errorMsg = describeError(error)
              this.logger(`Error processing ${address}: ${errorMsg}`)

              if (!this.continueOnError) {
                throw error
              }
            }
          }
        }

        // Save batch
        if (snapshots.length > 0) {
          try {
            await this.saveBatchWithRetry(snapshots, batchNum, totalBatches)
            saved += snapshots.length

            // Post-commit hook: invalidate trust-score cache for all
            // addresses in this batch so the next read fetches fresh scores.
            // Fire-and-forget: invalidation failure must not fail the run,
            // but it is counted so stale-cache incidents stay observable.
            const addresses = snapshots.map((s) => s.address)
            trustScoreInvalidationHook.execute(...addresses).catch((err) => {
              scoreSnapshotInvalidationFailuresTotal.inc()
              // I8: the raw error is deliberately not forwarded — hook errors
              // can carry connection strings — so the log gets the sanitised
              // rendering (secrets masked, single line, truncated) instead.
              logger.error(
                `[ScoreSnapshotJob] Failed to invalidate trust-score cache for ${addresses.length} addresses after batch save: ${describeError(err)}`,
              )
            })
          } catch (error) {
            // I4: the batch was not saved — count every snapshot in it as an
            // error rather than silently dropping it from the metrics.
            errors += snapshots.length
            const errorMsg = describeError(error)
            this.logger(
              `Error saving batch ${batchNum}/${totalBatches} (${snapshots.length} snapshots) after ${this.saveRetryAttempts} attempt(s): ${errorMsg}`,
            )
            scoreSnapshotBatchFailuresTotal.inc({ stage: 'save' })

            if (!this.continueOnError) {
              throw error
            }
          }
        }
      }
    } catch (error) {
      const errorMsg = describeError(error)
      this.logger(`Job failed: ${errorMsg}`)
      throw error
    }

    const duration = Date.now() - startMs
    this.logger(
      `Job completed: ${processed} processed, ${saved} saved, ${errors} errors, ${duration}ms`,
    )

    return {
      processed,
      saved,
      errors,
      duration,
      aggregationDuration,
      startTime,
    }
  }

  /**
   * Persist a batch with bounded retries for transient failures (invariant
   * I4). Throws after the final attempt; `saved` is only advanced by the
   * caller when this resolves.
   */
  private async saveBatchWithRetry(
    snapshots: ScoreSnapshot[],
    batchNum: number,
    totalBatches: number,
  ): Promise<void> {
    let lastError: unknown
    for (let attempt = 1; attempt <= this.saveRetryAttempts; attempt++) {
      try {
        await this.store.saveBatch(snapshots)
        recordScoreSnapshotBatch(snapshots.length)
        if (attempt > 1) {
          this.logger(
            `Batch ${batchNum}/${totalBatches} saved on attempt ${attempt}/${this.saveRetryAttempts}`,
          )
        }
        return
      } catch (error) {
        lastError = error
        const isLastAttempt = attempt >= this.saveRetryAttempts
        if (isLastAttempt) break

        scoreSnapshotBatchRetriesTotal.inc()
        const errorMsg = describeError(error)
        this.logger(
          `saveBatch attempt ${attempt}/${this.saveRetryAttempts} failed for batch ${batchNum}/${totalBatches}, retrying in ${this.saveRetryBackoffMs}ms: ${errorMsg}`,
        )
        await new Promise((resolve) => setTimeout(resolve, this.saveRetryBackoffMs))
      }
    }
    throw lastError
  }

  private async loadBatchData(batch: string[]): Promise<Map<string, IdentityData>> {
    const rows = await this.dataSource.getIdentityDataBatch!(batch)
    if (!Array.isArray(rows)) {
      // Surface a diagnosable load failure (handled under I3) instead of a
      // generic "rows is not iterable" TypeError deep inside the map.
      throw new TypeError(
        `getIdentityDataBatch returned ${rows === null ? 'null' : typeof rows}, expected an array`,
      )
    }
    const byAddress = new Map<string, IdentityData>()
    for (let i = 0; i < rows.length; i++) {
      const row: unknown = rows[i]
      if (row === null || typeof row !== 'object' || Array.isArray(row)) {
        // A malformed row carries no address, so it cannot be attributed to
        // a single identity — failing the batch (I3) is the only way to keep
        // the loss visible instead of pretending the identity has no data.
        throw new TypeError(
          `getIdentityDataBatch returned a malformed row at index ${i} (${row === null ? 'null' : Array.isArray(row) ? 'array' : typeof row})`,
        )
      }
      // I1: key rows with the same normalization the request side used, so a
      // data source that echoes addresses with surrounding whitespace cannot
      // make a valid identity look missing (silent snapshot loss).
      const key = normalizeAddress((row as IdentityData).address)
      if (key !== null) byAddress.set(key, row as IdentityData)
    }
    return byAddress
  }
}

/**
 * Create a score snapshot job.
 *
 * @param dataSource - Source for identity data
 * @param store - Store for persisting snapshots
 * @param scoreComputer - Function to compute scores
 * @param options - Job options
 * @returns ScoreSnapshotJob instance
 */
export function createScoreSnapshotJob(
  dataSource: IdentityDataSource,
  store: ScoreSnapshotStore,
  scoreComputer: ScoreComputer,
  options?: SnapshotJobOptions
): ScoreSnapshotJob {
  return new ScoreSnapshotJob(dataSource, store, scoreComputer, options)
}
