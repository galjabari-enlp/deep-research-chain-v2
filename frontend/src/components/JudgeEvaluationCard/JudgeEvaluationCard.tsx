import React, { useMemo, useState } from 'react'
import type { Evaluation, Metadata } from '../../types/judge'

export type JudgeEvaluationCardProps = {
  topic?: string
  reportId?: string
  evaluation: Evaluation
  metadata?: Metadata
  publishState?: {
    status: 'idle' | 'publishing' | 'published' | 'error'
    publishedAt?: string
    error?: string
  }
  onPublish?: () => void
  onRequestRevision?: () => void
}

function clamp01(n: number) {
  if (!Number.isFinite(n)) return 0
  return Math.max(0, Math.min(1, n))
}

function clamp(n: number, min: number, max: number) {
  if (!Number.isFinite(n)) return min
  return Math.max(min, Math.min(max, n))
}

function toStars(overall10: number) {
  return clamp(overall10 / 2, 0, 5)
}

function toPct(n: number | undefined) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return 0
  return clamp(n, 0, 100)
}

function takeTop(items: unknown, max: number) {
  if (!Array.isArray(items)) return []
  return items
    .map((s) => String(s || '').trim())
    .filter(Boolean)
    .slice(0, max)
}

function StatusBadge({ recommendation }: { recommendation?: Evaluation['recommendation'] }) {
  const { label, classes } = useMemo(() => {
    if (recommendation === 'publish') {
      return {
        label: 'Approved for publication',
        classes: 'bg-green-500/10 border-green-500/30 text-green-200',
      }
    }
    if (recommendation === 'revise') {
      return {
        label: 'Needs revision',
        classes: 'bg-amber-500/10 border-amber-500/30 text-amber-200',
      }
    }
    if (recommendation === 'reject') {
      return {
        label: 'Rejected (do not publish)',
        classes: 'bg-red-500/10 border-red-500/30 text-red-200',
      }
    }
    return {
      label: 'Evaluation complete',
      classes: 'bg-[#233648] border-[#324d67] text-[#92adc9]',
    }
  }, [recommendation])

  return <span className={`inline-flex items-center px-2 py-0.5 rounded-full border text-[11px] font-bold ${classes}`}>{label}</span>
}

let starGradientIdCounter = 0

function StarRating({ value }: { value: number }) {
  const stars = useMemo(() => {
    const v = clamp(value, 0, 5)
    const full = Math.floor(v)
    const frac = v - full
    const half = frac >= 0.5 ? 1 : 0
    const empty = 5 - full - half
    return { full, half, empty }
  }, [value])

  const halfId = useMemo(() => {
    starGradientIdCounter += 1
    return `half-${starGradientIdCounter}`
  }, [])

  const Star = ({ fill }: { fill: 'full' | 'half' | 'empty' }) => {
    const base = 'w-4 h-4'
    if (fill === 'empty') {
      return (
        <svg className={base} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M12 17.3 18.2 21l-1.7-7.2L22 9.2l-7.4-.6L12 2 9.4 8.6 2 9.2l5.5 4.6L5.8 21 12 17.3Z"
            stroke="currentColor"
            strokeWidth="1.4"
            className="text-white/30"
          />
        </svg>
      )
    }

    if (fill === 'half') {
      return (
        <svg className={base} viewBox="0 0 24 24" aria-hidden="true">
          <defs>
            <linearGradient id={halfId} x1="0" x2="1" y1="0" y2="0">
              <stop offset="50%" stopColor="currentColor" />
              <stop offset="50%" stopColor="transparent" />
            </linearGradient>
          </defs>
          <path
            d="M12 17.3 18.2 21l-1.7-7.2L22 9.2l-7.4-.6L12 2 9.4 8.6 2 9.2l5.5 4.6L5.8 21 12 17.3Z"
            fill={`url(#${halfId})`}
            stroke="currentColor"
            strokeWidth="1.2"
            className="text-amber-300"
          />
        </svg>
      )
    }

    return (
      <svg className={base} viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M12 17.3 18.2 21l-1.7-7.2L22 9.2l-7.4-.6L12 2 9.4 8.6 2 9.2l5.5 4.6L5.8 21 12 17.3Z"
          fill="currentColor"
          className="text-amber-300"
        />
      </svg>
    )
  }

  return (
    <div className="inline-flex items-center gap-1" aria-label={`Rating: ${value.toFixed(1)} out of 5`}>
      {Array.from({ length: stars.full }).map((_, i) => (
        <Star key={`f-${i}`} fill="full" />
      ))}
      {stars.half ? <Star key="h" fill="half" /> : null}
      {Array.from({ length: stars.empty }).map((_, i) => (
        <Star key={`e-${i}`} fill="empty" />
      ))}
    </div>
  )
}

function ScoreBar({ pct }: { pct: number }) {
  const w = clamp(pct, 0, 100)
  return (
    <div className="h-2 w-full bg-[#233648] rounded-full overflow-hidden" role="progressbar" aria-valuenow={w} aria-valuemin={0} aria-valuemax={100}>
      <div className="h-full bg-primary rounded-full" style={{ width: `${w}%` }} />
    </div>
  )
}

function MetricBlock({
  title,
  scoreText,
  pct,
  strengths,
  weaknesses,
}: {
  title: string
  scoreText: string
  pct: number
  strengths: string[]
  weaknesses: string[]
}) {
  const showLists = strengths.length > 0 || weaknesses.length > 0
  return (
    <div className="bg-[#111a22] border border-[#233648] rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] text-[#92adc9] font-bold uppercase tracking-widest">{title}</div>
        <div className="text-xs font-bold text-white">{scoreText}</div>
      </div>
      <ScoreBar pct={pct} />

      {showLists ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
          <div className="space-y-1">
            <div className="text-[10px] text-[#92adc9] font-bold uppercase">Strengths</div>
            {strengths.length ? (
              <ul className="text-[12px] text-white/85 space-y-1 list-disc pl-4">
                {strengths.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            ) : (
              <div className="text-[12px] text-white/40">—</div>
            )}
          </div>
          <div className="space-y-1">
            <div className="text-[10px] text-[#92adc9] font-bold uppercase">Weaknesses</div>
            {weaknesses.length ? (
              <ul className="text-[12px] text-white/85 space-y-1 list-disc pl-4">
                {weaknesses.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            ) : (
              <div className="text-[12px] text-white/40">—</div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default function JudgeEvaluationCard({
  topic,
  reportId,
  evaluation,
  metadata,
  publishState,
  onPublish,
  onRequestRevision,
}: JudgeEvaluationCardProps) {
  const [improvementsOpen, setImprovementsOpen] = useState(false)

  const overall10 = Number(evaluation?.overall_score ?? 0)
  const stars = toStars(overall10)

  const grade = String(evaluation?.grade || '').trim()
  const recommendation = evaluation?.recommendation

  const factual = evaluation?.factual_accuracy
  const completeness = evaluation?.completeness

  const flags = takeTop(evaluation?.flags, 6)
  const suggested = takeTop(evaluation?.suggested_improvements, 12)

  const confidence = typeof evaluation?.confidence === 'number' ? clamp01(evaluation.confidence) : null

  const canPublish = recommendation === 'publish'
  const publishStatus = publishState?.status || 'idle'
  const isPublishing = publishStatus === 'publishing'
  const isPublished = publishStatus === 'published'

  const handlePublish = () => {
    if (!canPublish) return
    if (onPublish) onPublish()
    else console.log('publish-now clicked', { reportId })
  }

  const handleRequestRevision = () => {
    if (onRequestRevision) onRequestRevision()
    else console.log('request-revision clicked', { reportId })
  }

  const title = topic ? `Research Report: “${topic}”` : 'Research Report Evaluation'
  const titleForDisplay = title
  const titleForTooltip = title

  return (
    <section className="border border-[#233648] rounded-xl bg-[#0f1a25] p-4 space-y-4" aria-label="Judge evaluation">
      <header className="space-y-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0" style={{ maxWidth: '70%' }}>
            <h3 className="text-sm font-bold text-white" title={titleForTooltip} style={{ overflowWrap: 'anywhere' }}>
              {titleForDisplay}
            </h3>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <StatusBadge recommendation={recommendation} />
              {grade ? (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full border bg-[#233648] border-[#324d67] text-[11px] font-bold text-[#92adc9]">
                  Grade: {grade}
                </span>
              ) : null}
              {metadata?.judge_model ? (
                <span
                  className="inline-flex items-center px-2 py-0.5 rounded-full border bg-[#111a22] border-[#233648] text-[11px] font-bold text-white/70"
                  title={String(metadata.judge_model)}
                >
                  {String(metadata.judge_model)}
                </span>
              ) : null}
            </div>
          </div>

          <div className="shrink-0 text-right">
            <div className="flex items-center justify-end gap-2">
              <StarRating value={stars} />
              <div className="text-sm font-extrabold text-white tabular-nums">{`${overall10.toFixed(1)}/10`}</div>
            </div>
            {confidence != null ? (
              <div className="text-[11px] text-[#92adc9] mt-1">Confidence: {confidence.toFixed(2)}</div>
            ) : null}
          </div>
        </div>

        {flags.length ? (
          <div className="flex flex-wrap gap-2">
            {flags.map((f) => (
              <span
                key={f}
                className="inline-flex items-center px-2 py-0.5 rounded-full border bg-amber-500/10 border-amber-500/20 text-[11px] font-bold text-amber-200"
              >
                {f}
              </span>
            ))}
          </div>
        ) : null}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <MetricBlock
          title="Factual Accuracy"
          scoreText={`${Number(factual?.score ?? 0)}/${Number(factual?.max_score ?? 10)}`}
          pct={toPct(factual?.percentage)}
          strengths={takeTop(factual?.strengths, 2)}
          weaknesses={takeTop(factual?.weaknesses, 2)}
        />
        <MetricBlock
          title="Completeness"
          scoreText={`${Number(completeness?.score ?? 0)}/${Number(completeness?.max_score ?? 10)}`}
          pct={toPct(completeness?.percentage)}
          strengths={takeTop(completeness?.strengths, 2)}
          weaknesses={takeTop(completeness?.weaknesses, 2)}
        />
      </div>

      {evaluation?.overall_assessment ? (
        <blockquote className="border-l-4 border-primary/40 bg-[#111a22] rounded-lg p-3 text-[13px] text-white/85 leading-relaxed italic">
          {String(evaluation.overall_assessment)}
        </blockquote>
      ) : null}

      {suggested.length ? (
        <div className="space-y-2">
          <button
            type="button"
            className="text-[11px] font-bold text-[#92adc9] hover:text-white inline-flex items-center gap-1"
            onClick={() => setImprovementsOpen((v) => !v)}
            aria-expanded={improvementsOpen}
          >
            <span className="material-symbols-outlined text-[16px]">{improvementsOpen ? 'expand_less' : 'expand_more'}</span>
            Suggested improvements ({suggested.length})
          </button>
          {improvementsOpen ? (
            <ul className="text-[12px] text-white/85 space-y-1 list-disc pl-5">
              {suggested.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      <div className="flex flex-wrap gap-2 pt-1">
        <button
          type="button"
          onClick={handlePublish}
          disabled={!canPublish || isPublishing || isPublished}
          className={
            canPublish && !isPublished
              ? 'px-3 py-2 rounded-lg bg-primary text-white text-xs font-bold hover:brightness-105 transition-colors disabled:opacity-60'
              : 'px-3 py-2 rounded-lg bg-[#233648] text-white/50 text-xs font-bold cursor-not-allowed border border-[#324d67]'
          }
          aria-label="Publish now"
        >
          {isPublished ? 'Published' : isPublishing ? 'Publishing…' : 'Publish now'}
        </button>

        <button
          type="button"
          onClick={handleRequestRevision}
          className="px-3 py-2 rounded-lg bg-[#233648] border border-[#324d67] text-[#92adc9] hover:text-white hover:border-primary text-xs font-bold transition-colors"
          aria-label="Request revision"
        >
          Request revision
        </button>
      </div>

      {publishState?.error ? <div className="text-[11px] text-red-200">{publishState.error}</div> : null}
      {isPublished && publishState?.publishedAt ? (
        <div className="text-[10px] text-white/45">Published at: {publishState.publishedAt}</div>
      ) : null}

      {reportId ? <div className="text-[10px] text-white/35">Report ID: {reportId}</div> : null}
    </section>
  )
}
