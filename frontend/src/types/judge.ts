export type Citation = {
  id: string
  url: string
  title?: string
}

export type ReportBlock = {
  id: string
  heading?: string
  text: string
  citations: Citation[]
}

export type Report = {
  id: string
  topic?: string
  content?: string
  sources?: unknown[]
  word_count?: number
  created_at?: string
  blocks?: ReportBlock[]
  // Legacy fields used by the existing dashboard rendering.
  key_findings?: string[]
  evidence_and_sources?: string[]
  limitations?: string[]
}

export type RubricScore = {
  score: number
  max_score: number
  percentage: number
  reasoning?: string
  strengths?: string[]
  weaknesses?: string[]
}

export type CompletenessEval = RubricScore & {
  coverage?: Record<string, unknown>
}

export type Evaluation = {
  factual_accuracy: RubricScore
  completeness: CompletenessEval
  overall_score: number
  grade?: string
  overall_assessment?: string
  recommendation?: 'publish' | 'revise' | 'reject'
  confidence?: number
  flags?: string[]
  suggested_improvements?: string[]
}

export type Metadata = {
  evaluation_id?: string
  evaluated_at?: string
  judge_model?: string
  processing_time_ms?: number
  evaluation_version?: string
}

export type RunResult = {
  status?: string
  query?: string
  iteration_count?: number
  trace?: string[] | string
  sources?: unknown[]
  critic?: unknown
  report?: Report
  evaluation?: Evaluation
  metadata?: Metadata
}

export function hasEvaluation(payload: unknown): payload is { evaluation: Evaluation } {
  if (!payload || typeof payload !== 'object') return false
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const p: any = payload
  const ev = p.evaluation
  return !!(
    ev &&
    typeof ev === 'object' &&
    ev.factual_accuracy &&
    typeof ev.factual_accuracy.score === 'number' &&
    ev.completeness &&
    typeof ev.completeness.score === 'number' &&
    typeof ev.overall_score === 'number'
  )
}
