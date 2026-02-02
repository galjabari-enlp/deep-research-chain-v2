import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ChatPanel from './components/ChatPanel/ChatPanel.jsx'

function pad2(n) {
  return String(n).padStart(2, '0')
}

function fmtElapsed(ms) {
  const s = Math.max(0, Math.floor(ms / 1000))
  const hh = Math.floor(s / 3600)
  const mm = Math.floor((s % 3600) / 60)
  const ss = s % 60
  return `${pad2(hh)}:${pad2(mm)}:${pad2(ss)}`
}

function safeHostname(url) {
  try {
    return new URL(url).hostname
  } catch {
    return ''
  }
}

function cleanLine(s) {
  return String(s || '')
    .replace(/^\s*[*•\-\u2022]+\s*/g, '')
    .trim()
}

// IMPORTANT: Do not use the word "Reasoning" anywhere in UI labels.
// If the raw trace contains "Reasoning produced ...", it is categorized under "Model Notes".
function classifyTraceLine(line) {
  const s = line
  if (/^Planned\b/i.test(s)) return 'Plans'
  if (/^Critic (decision|evidence):/i.test(s)) return 'Critic'
  if (/^Routing:/i.test(s) || /Report generated/i.test(s)) return 'Routing & Report'
  if (/^Searched:/i.test(s)) return 'Search Queries'
  if (/^Source considered:/i.test(s)) return 'Sources Considered'
  if (/^Fetched:/i.test(s)) return 'Fetch Results'
  if (/^Reasoning produced/i.test(s)) return 'Model Notes'
  return 'Other'
}

function normalizeTrace(trace) {
  if (!trace) return []
  if (Array.isArray(trace)) return trace.map(String).flatMap((s) => String(s).split('\n'))
  return String(trace).split('\n')
}

function extractUrls(s) {
  return String(s || '').match(/https?:\/\/[^\s)]+/g) || []
}

function compactUrlParts(url) {
  try {
    const u = new URL(url)
    const host = u.hostname
    let path = (u.pathname || '') + (u.search || '')
    if (path.length > 34) path = path.slice(0, 34) + '…'
    return { host, path }
  } catch {
    return { host: url, path: '' }
  }
}

function statusPillClass(statusCode) {
  const n = Number(statusCode)
  if (n >= 200 && n < 300) return 'bg-green-500/10 border-green-500/30 text-green-300'
  if (n >= 300 && n < 400) return 'bg-blue-500/10 border-blue-500/30 text-blue-300'
  if (n >= 400 && n < 500) return 'bg-amber-500/10 border-amber-500/30 text-amber-200'
  if (n >= 500) return 'bg-red-500/10 border-red-500/30 text-red-200'
  return 'bg-[#233648] border-[#324d67] text-[#92adc9]'
}

function computeTrace(lines) {
  const order = [
    'Plans',
    'Critic',
    'Routing & Report',
    'Search Queries',
    'Sources Considered',
    'Fetch Results',
    'Model Notes',
    'Other',
  ]

  const defaultsOpen = {
    Plans: true,
    Critic: true,
    'Routing & Report': true,
    'Search Queries': false,
    'Sources Considered': false,
    'Fetch Results': false,
    'Model Notes': false,
    Other: false,
  }

  const sections = {}
  for (const k of order) sections[k] = []

  let searches = 0
  let sources = 0
  let fetchTotal = 0
  let fetchOk = 0

  for (const raw of lines) {
    const line = cleanLine(raw)
    if (!line) continue
    const sec = classifyTraceLine(line)

    if (/^Searched:/i.test(line)) searches += 1
    if (/^Source considered:/i.test(line)) sources += 1
    if (/^Fetched:/i.test(line)) {
      fetchTotal += 1
      const m = line.match(/\(status=(\d{3})/i)
      if (m && Number(m[1]) >= 200 && Number(m[1]) < 300) fetchOk += 1
    }

    sections[sec].push({ line })
  }

  return {
    order,
    defaultsOpen,
    sections,
    counts: { searches, sources, fetchOk, fetchTotal },
  }
}

export default function App() {
  // When running on Vite dev server, we rely on Vite proxy rules for /health and /research/*.
  // So API_BASE should stay empty (same-origin). When running elsewhere, allow override.
  const API_BASE = useMemo(() => (window.API_BASE || '').toString().replace(/\/$/, ''), [])

  const [connState, setConnState] = useState('checking') // checking|connected|disconnected
  const [connDetail, setConnDetail] = useState('')

  const [statusMode, setStatusMode] = useState('idle') // idle|working
  const [stage, setStage] = useState(null)

  // Chat (controlled by App so it can drive + reflect backend state)
  const [maxRevisions, setMaxRevisions] = useState(10)
  const [draft, setDraft] = useState('')

  // Publish UI state (keyed by reportId)
  const [publishByReportId, setPublishByReportId] = useState({})

  const [messages, setMessages] = useState(() => [
    {
      id: 'seed-user',
      role: 'user',
      text: 'Analyze the architectural impact of Large Language Models on enterprise software. Focus on latency\ntradeoffs and the move towards event-driven patterns.',
    },
    {
      id: 'seed-agent',
      role: 'agent',
      text: "I am currently analyzing the architectural patterns. I've completed the initial literature review and am\nnow moving into the entity extraction phase.",
      statusText: 'Ready when you are.',
    },
  ])

  const [iterationCount, setIterationCount] = useState(0)
  const [elapsedMs, setElapsedMs] = useState(0)

  const [traceLines, setTraceLines] = useState([])
  const [critic, setCritic] = useState(null)
  const [sources, setSources] = useState([])
  const [report, setReport] = useState({ key_findings: [], evidence_and_sources: [], limitations: [] })
  const [evaluation, setEvaluation] = useState(null)
  const [judgeMetadata, setJudgeMetadata] = useState(null)

  // Revision UI state
  const [reviseUI, setReviseUI] = useState({ open: false, messageId: null, text: '' })

  const [errorMsg, setErrorMsg] = useState('')

  const [traceOpen, setTraceOpen] = useState(() => {
    const base = computeTrace([]).defaultsOpen
    return { ...base }
  })

  const [lastResponse, setLastResponse] = useState(null)

  const runRef = useRef({ es: null, startedAt: null })
  const elapsedTimerRef = useRef(null)

  const stageMap = useMemo(
    () => ({
      policy_guard: 'policy_guard',
      blocked: 'blocked',
      planning: 'planning',
      search: 'search',
      reasoning: 'search',
      critic_step: 'critic_step',
      report_step: 'report_step',
      judge_step: 'judge_step',
      published: 'published',
    }),
    [],
  )

  // When blocked, the run is terminated early; hide the whole dashboard timeline.
  const stageKey = stage ? stageMap[stage] || stage : null

  const completedStages = useMemo(() => {
    // Mark steps as completed only when we have clearly advanced past them.
    // This fixes a UI bug where when policy_guard is active the Plan/Search cards show as completed.
    const s = new Set()

    // Plan is completed once we have moved into (or past) Search.
    if (stageKey && ['search', 'critic_step', 'report_step', 'judge_step', 'published'].includes(stageKey)) {
      s.add('planning')
    }

    // Search is completed once we have moved into (or past) Critic.
    if (stageKey && ['critic_step', 'report_step', 'judge_step', 'published'].includes(stageKey)) {
      s.add('search')
    }

    // Critic is completed once we have moved into (or past) Report.
    if (stageKey && ['report_step', 'judge_step', 'published'].includes(stageKey)) {
      s.add('critic_step')
    }

    // Report is completed once we have moved into (or past) Judge.
    if (stageKey && ['judge_step', 'published'].includes(stageKey)) {
      s.add('report_step')
    }

    // Judge completes only when we are at judge step (or beyond, if we add more in the future).
    if (stageKey && ['judge_step', 'published'].includes(stageKey)) {
      s.add('judge_step')
    }

    if (stageKey === 'published') s.add('published')

    return s
  }, [stageKey])

  const stopCurrentRun = useCallback(() => {
    const cur = runRef.current
    if (cur.es) {
      try {
        cur.es.close()
      } catch {
        // ignore
      }
      cur.es = null
    }

    if (elapsedTimerRef.current) {
      window.clearInterval(elapsedTimerRef.current)
      elapsedTimerRef.current = null
    }

    if (cur.startedAt != null) {
      setElapsedMs(Math.max(0, performance.now() - cur.startedAt))
    }

    setStatusMode('idle')
  }, [])

  const resetPanelsForNewRun = useCallback(() => {
    setTraceLines([])
    setCritic(null)
    setSources([])
    setReport({ key_findings: [], evidence_and_sources: [], limitations: [] })
    setEvaluation(null)
    setJudgeMetadata(null)
    setErrorMsg('')
    setStage(null)
    setIterationCount(0)
    setElapsedMs(0)

    // reset trace open defaults
    setTraceOpen({ ...computeTrace([]).defaultsOpen })
  }, [])

  const addMessage = useCallback((role, text, extra = null) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`
    setMessages((prev) => [...prev, { id, role, text, ...(extra || {}) }])
  }, [])

  const setSendEnabled = useCallback((enabled) => {
    // React controls disabled on the button directly; no-op retained for parity.
    return enabled
  }, [])

  const checkHealth = useCallback(async () => {
    const base = API_BASE || window.location.origin
    const url = `${base}/health`
    setConnState('checking')
    setConnDetail(`Checking ${url}`)

    try {
      const res = await fetch(url, {
        cache: 'no-store',
        headers: { Accept: 'application/json' },
      })

      if (!res.ok) {
        setConnState('disconnected')
        setConnDetail(`Healthcheck failed: ${res.status} ${res.statusText} (${url})`)
        setSendEnabled(true)
        return
      }

      setConnState('connected')
      setConnDetail(`Connected: ${url}`)
      setSendEnabled(true)
    } catch (e) {
      setConnState('disconnected')
      setConnDetail(`Healthcheck error: ${String(e)} (${url})`)
      setSendEnabled(true)
    }
  }, [API_BASE, setSendEnabled])

  useEffect(() => {
    checkHealth()
    const t = window.setInterval(checkHealth, 5000)
    return () => window.clearInterval(t)
  }, [checkHealth])

  // Clipboard copy for trace links
  const onCopyUrl = useCallback(async (url) => {
    try {
      await navigator.clipboard.writeText(url)
    } catch {
      // ignore
    }
  }, [])

  const runResearch = useCallback(async () => {
    // Start a fresh run; do not let previous run evaluation bleed into the next.
    setEvaluation(null)
    setJudgeMetadata(null)

    stopCurrentRun()
    setErrorMsg('')

    const q = (draft || '').trim()
    const maxRev = Number(maxRevisions || 10)
    if (!q) return

    setDraft('')

    addMessage('user', q)
    setStatusMode('working')
    setStage('policy_guard')
    setIterationCount(0)

    runRef.current.startedAt = performance.now()
    elapsedTimerRef.current = window.setInterval(() => {
      const started = runRef.current.startedAt
      if (started != null) setElapsedMs(Math.max(0, performance.now() - started))
    }, 250)

    resetPanelsForNewRun()

    // “Agent” placeholder so the left chat shows progress while SSE runs.
    const agentRunId = `${Date.now()}-${Math.random().toString(16).slice(2)}`
    setMessages((prev) => [
      ...prev,
      {
        id: agentRunId,
        role: 'agent',
        text: 'Working on it…',
        statusText: 'Starting…',
        // Ensure we never render stale evaluation while the new run is in progress.
        evaluation: undefined,
        judgeMetadata: undefined,
        reportBlocks: undefined,
        topic: undefined,
        reportId: undefined,
      },
    ])

    try {
      const base = API_BASE || window.location.origin

      // SSE endpoint is `/research/stream` (see backend OpenAPI). If you see a 404 + readyState=0,
      // it typically means the frontend is pointing at a non-existent stream route.
      const streamUrl = `${base}/research/stream?query=${encodeURIComponent(q)}&max_revisions=${encodeURIComponent(String(maxRev))}`

      // Cache-bust to avoid any intermediary/browser caching oddities for SSE.
      const streamUrlNoCache = `${streamUrl}&_ts=${Date.now()}`

      const es = new EventSource(streamUrlNoCache)
      runRef.current.es = es

      let finalData = await new Promise((resolve, reject) => {
        es.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data)

            // TEMP DEBUG: print SSE payloads so we can verify whether `evaluation` is actually sent.
            console.log('[SSE]', msg)

            if (msg.type === 'state') {
              if (msg.stage) setStage(msg.stage)
              if (msg.iteration_count != null) setIterationCount(msg.iteration_count)
              if (msg.trace) setTraceLines(normalizeTrace(msg.trace).map(cleanLine).filter(Boolean))
              if (msg.execution_trace) {
                // Keep the richer trace object for future UI; dashboard currently uses traceLines.
                // (We may render `execution_trace` later if desired.)
              }
              if (msg.critic) setCritic(msg.critic)

              // Some backends may stream evaluation separately; wire it defensively.
              // When this happens, treat it as the judge stage.
              if (msg.evaluation) {
                setEvaluation(msg.evaluation)
                if (msg.metadata) setJudgeMetadata(msg.metadata)
                setStage('judge_step')
              }

              if (msg.plan && msg.stage === 'planning') {
                // Show plan objective in chat once it’s known
                const summary = msg.plan.research_objective || ''
                if (summary) {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === agentRunId
                        ? {
                            ...m,
                            text: summary,
                            statusText: `Stage: ${msg.stage}…`,
                          }
                        : m,
                    ),
                  )
                }
              } else if (msg.stage) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === agentRunId
                      ? {
                          ...m,
                          statusText: `Stage: ${msg.stage}…`,
                        }
                      : m,
                  ),
                )
              }

              return
            }

            if (msg.type === 'final') {
              resolve(msg.response)
              return
            }

            // Defensive: some servers may emit final payload directly on the wire.
            if (msg.status === 'complete' && (msg.report || msg.evaluation)) {
              resolve(msg)
              return
            }

            if (msg.type === 'error') {
              reject(new Error(msg.message || 'stream error'))
            }
          } catch (e) {
            reject(e)
          }
        }

        es.onerror = (ev) => {
          // EventSource errors are noisy and opaque; log current readyState to diagnose transient disconnects.
          // readyState: 0=CONNECTING, 1=OPEN, 2=CLOSED
          console.warn('[SSE ERROR]', { readyState: es.readyState, ev })
          reject(new Error(`SSE connection error (readyState=${es.readyState})`))
        }
      })

      // NOTE: backend is currently inconsistent: sometimes it sets top-level `evaluation`,
      // sometimes only writes judge info into `execution_trace`.
      //
      // Rule: ALWAYS prefer the evaluation derived from `execution_trace` if present, because
      // the top-level `evaluation` can be stale (from a prior run).
      finalData = {
        ...(finalData || {}),
        evaluation: finalData?.evaluation || null,
        metadata: finalData?.metadata || null,
      }

      // If judge info is present in execution_trace, derive evaluation and overwrite any stale top-level one.
      if (finalData.execution_trace && typeof finalData.execution_trace === 'object') {
        try {
          const iters = Array.isArray(finalData.execution_trace.iterations) ? finalData.execution_trace.iterations : []
          const flatItems = []
          for (const it of iters) {
            const sections = Array.isArray(it?.sections) ? it.sections : []
            for (const sec of sections) {
              const items = Array.isArray(sec?.items) ? sec.items : []
              for (const item of items) flatItems.push(item)
            }
          }

          const judgeItem = [...flatItems]
            .reverse()
            .find((x) => typeof x?.label === 'string' && String(x.label).toLowerCase().startsWith('judge:'))

          if (judgeItem && judgeItem.data && typeof judgeItem.data === 'object') {
            const d = judgeItem.data

            // Parse from label: "Judge: publish (overall=9.65, grade=A)"
            const label = String(judgeItem.label || '')
            const mRec = label.match(/^Judge:\s*(publish|revise|reject)\b/i)
            const mOverall = label.match(/overall=([0-9]+(?:\.[0-9]+)?)/i)
            const mGrade = label.match(/grade=([A-F][+-]?)/i)

            const recommendation = mRec ? mRec[1].toLowerCase() : typeof d.recommendation === 'string' ? d.recommendation : null
            const overall = mOverall ? Number(mOverall[1]) : typeof d.overall_score === 'number' ? d.overall_score : null
            const grade = mGrade ? mGrade[1] : typeof d.grade === 'string' ? d.grade : null

            const accuracy = typeof d.accuracy === 'number' ? d.accuracy : null
            const completeness = typeof d.completeness === 'number' ? d.completeness : null

            const maxScore = 10
            const mkRubric = (score) => ({
              score: typeof score === 'number' ? score : 0,
              max_score: maxScore,
              percentage: typeof score === 'number' ? Math.round((score / maxScore) * 100) : 0,
              strengths: [],
              weaknesses: [],
            })

            finalData.evaluation = {
              factual_accuracy: mkRubric(accuracy),
              completeness: mkRubric(completeness),
              overall_score:
                typeof overall === 'number'
                  ? overall
                  : typeof accuracy === 'number' && typeof completeness === 'number'
                    ? (accuracy + completeness) / 2
                    : 0,
              grade: grade || undefined,
              overall_assessment: typeof judgeItem.detail === 'string' ? judgeItem.detail : undefined,
              recommendation: recommendation || undefined,
              confidence: undefined,
              flags: Array.isArray(d.flags) ? d.flags.map(String) : [],
              suggested_improvements: [],
            }

            finalData.metadata = {
              ...(finalData.metadata || {}),
              evaluation_id: typeof d.evaluation_id === 'string' ? d.evaluation_id : finalData.metadata?.evaluation_id,
            }
          }
        } catch {
          // ignore execution_trace parse errors
        }
      }

      // If no execution_trace-derived evaluation exists, fall back to streamed state values.
      if (!finalData.evaluation && evaluation) finalData.evaluation = evaluation
      if (!finalData.metadata && judgeMetadata) finalData.metadata = judgeMetadata

      // Fallback: derive a minimal evaluation from the rich execution_trace if the top-level
      // `evaluation` is null (backend is currently emitting judge data only inside execution_trace).
      if (!finalData.evaluation && finalData.execution_trace && typeof finalData.execution_trace === 'object') {
        try {
          const iters = Array.isArray(finalData.execution_trace.iterations) ? finalData.execution_trace.iterations : []
          const flatItems = []
          for (const it of iters) {
            const sections = Array.isArray(it?.sections) ? it.sections : []
            for (const sec of sections) {
              const items = Array.isArray(sec?.items) ? sec.items : []
              for (const item of items) flatItems.push(item)
            }
          }

          const judgeItem = [...flatItems]
            .reverse()
            .find((x) => typeof x?.label === 'string' && String(x.label).toLowerCase().startsWith('judge:'))

          if (judgeItem && judgeItem.data && typeof judgeItem.data === 'object') {
            const d = judgeItem.data
            const overall = typeof d.overall_score === 'number' ? d.overall_score : null
            const grade = typeof d.grade === 'string' ? d.grade : null
            const recommendation = typeof d.recommendation === 'string' ? d.recommendation : null

            // Support the "lite" data currently present in your payload: accuracy/completeness ints.
            const accuracy = typeof d.accuracy === 'number' ? d.accuracy : null
            const completeness = typeof d.completeness === 'number' ? d.completeness : null

            const maxScore = 10
            const mkRubric = (score) => ({
              score: typeof score === 'number' ? score : 0,
              max_score: maxScore,
              percentage: typeof score === 'number' ? Math.round((score / maxScore) * 100) : 0,
              strengths: [],
              weaknesses: [],
            })

            finalData.evaluation = {
              factual_accuracy: mkRubric(accuracy),
              completeness: mkRubric(completeness),
              overall_score:
                typeof overall === 'number'
                  ? overall
                  : typeof accuracy === 'number' && typeof completeness === 'number'
                    ? (accuracy + completeness) / 2
                    : 0,
              grade: grade || undefined,
              overall_assessment: typeof judgeItem.detail === 'string' ? judgeItem.detail : undefined,
              recommendation: recommendation || undefined,
              confidence: undefined,
              flags: Array.isArray(d.flags) ? d.flags.map(String) : [],
              suggested_improvements: [],
            }

            finalData.metadata = {
              ...(finalData.metadata || {}),
              evaluation_id: typeof d.evaluation_id === 'string' ? d.evaluation_id : finalData.metadata?.evaluation_id,
            }
          }
        } catch {
          // ignore fallback parsing errors
        }
      }

      console.log('[FINAL MERGED]', finalData)
      setLastResponse(finalData)

      setIterationCount(finalData.iteration_count ?? 0)
      setTraceLines(normalizeTrace(finalData.trace || []).map(cleanLine).filter(Boolean))
      setSources(finalData.sources || [])
      setCritic(finalData.critic || null)
      // If blocked, show a simple refusal message (no report/evaluation UI).
      if (finalData?.status === 'blocked') {
        const msg =
          String(finalData?.final_user_message || '').trim() ||
          "I can’t help with that request. Please try a different question."

        // Clear dashboard panels and stop stage tracking.
        setReport({ key_findings: [], evidence_and_sources: [], limitations: [] })
        setEvaluation(null)
        setJudgeMetadata(null)
        setStage('blocked')

        setMessages((prev) =>
          prev.map((m) =>
            m.id === agentRunId
              ? {
                  ...m,
                  text: msg,
                  reportBlocks: undefined,
                  evaluation: undefined,
                  judgeMetadata: undefined,
                  topic: undefined,
                  reportId: undefined,
                  statusText: 'Blocked.',
                }
              : m,
          ),
        )

        return
      }

      setReport(finalData.report || { key_findings: [], evidence_and_sources: [], limitations: [] })
      setEvaluation(finalData.evaluation || null)
      setJudgeMetadata(finalData.metadata || null)

      // If the backend includes evaluation (or we derived it), it is treated as the judge stage.
      setStage(finalData.evaluation ? 'judge_step' : 'report_step')

      // When the run completes, show the final report in the chat as well (rich rendering)
      // and optionally append the judge evaluation card.
      const blocks = (finalData.report && Array.isArray(finalData.report.blocks) && finalData.report.blocks) || []

      setMessages((prev) =>
        prev.map((m) =>
          m.id === agentRunId
            ? {
                ...m,
                text: 'Final report ready.',
                reportBlocks: blocks,
                evaluation: finalData.evaluation || undefined,
                judgeMetadata: finalData.metadata || undefined,
                topic: finalData.report?.topic || finalData.query || undefined,
                // Backends in this repo use either `report.id` or `report.report_id`.
                reportId: finalData.report?.id || finalData.report?.report_id || undefined,
                statusText: 'Done.',
              }
            : m,
        ),
      )
    } catch (err) {
      setErrorMsg(String(err))
      addMessage('agent', 'I hit an error calling the backend.', { statusText: 'Error.' })
      setStage(null)
    } finally {
      stopCurrentRun()
    }
  }, [API_BASE, addMessage, draft, maxRevisions, resetPanelsForNewRun, stopCurrentRun, evaluation, judgeMetadata])

  useEffect(() => {
    return () => {
      stopCurrentRun()
    }
  }, [stopCurrentRun])

  const onClear = useCallback(() => {
    stopCurrentRun()
    setDraft('')
    setMessages([])
    setLastResponse(null)
    setStage(null)
    setIterationCount(0)
    setElapsedMs(0)

    setTraceLines([])
    setSources([])
    setCritic(null)
    setReport({ key_findings: [], evidence_and_sources: [], limitations: [] })
    setEvaluation(null)
    setJudgeMetadata(null)
    setErrorMsg('')

    setStatusMode('idle')
    setTraceOpen({ ...computeTrace([]).defaultsOpen })
  }, [stopCurrentRun])

  // Currently unused; kept for easy wiring later if you want a Clear button in ChatPanel.
  void onClear

  const onExport = useCallback(() => {
    if (!lastResponse) return
    const q = (lastResponse.query || 'research').toString().slice(0, 40).replaceAll(/[^a-z0-9-_\s]/gi, '')
    const stamp = new Date().toISOString().replaceAll(':', '-')
    const blob = new Blob([JSON.stringify(lastResponse, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${stamp}-${q || 'research'}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }, [lastResponse])

  const parsedTrace = useMemo(() => computeTrace(traceLines), [traceLines])

  // When Final Report becomes available, move it into view.
  useEffect(() => {
    const hasBlocks = Array.isArray(report?.blocks) && report.blocks.length > 0
    const hasLegacy = Array.isArray(report?.key_findings) && report.key_findings.length > 0
    if (!hasBlocks && !hasLegacy) return

    window.setTimeout(() => {
      const el = document.getElementById('finalReportSection')
      if (el && typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ block: 'start', behavior: 'smooth' })
      }
    }, 50)
  }, [report])

  // Evidence list badges: map url -> index (1-based)
  // (Used by legacy report rendering; blocks UI uses explicit citation ids.)
  const urlToSourceIndex = useMemo(() => {
    const m = new Map()
    ;(sources || []).forEach((s, i) => {
      if (s && s.url) m.set(String(s.url), i + 1)
    })
    return m
  }, [sources])
  void urlToSourceIndex

  const handlePublish = useCallback(
    async (message) => {
      // If the backend doesn't provide an id, publish the latest report anyway using a deterministic local id.
      const reportId = String(
        message?.reportId ||
          lastResponse?.report?.id ||
          lastResponse?.report?.report_id ||
          lastResponse?.metadata?.evaluation_id ||
          `local-${Date.now()}`,
      )

      // optimistic UI
      setPublishByReportId((prev) => ({
        ...prev,
        [reportId]: { status: 'publishing' },
      }))

      try {
        // Use the message-specific report blocks when available (revision flow returns a new report)
        // so we don't accidentally publish a stale `lastResponse.report`.
        const reportFromMessage = {
          ...(lastResponse?.report || {}),
          ...(message?.topic ? { topic: message.topic } : {}),
          ...(Array.isArray(message?.reportBlocks) ? { blocks: message.reportBlocks } : {}),
          ...(message?.reportId ? { id: message.reportId } : {}),
        }

        const base = API_BASE || window.location.origin
        const url = `${base}/api/reports/${encodeURIComponent(String(reportId))}/publish`

        const payload = {
          query: String(message?.topic || lastResponse?.query || '').trim() || 'report',
          report: reportFromMessage,
          evaluation: message?.evaluation || null,
          metadata: message?.judgeMetadata || null,
          sources: lastResponse?.sources || [],
        }

        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify(payload),
        })

        const data = await res.json().catch(() => ({}))
        if (!res.ok) {
          const detail = data?.detail?.message || data?.detail || data?.message || `${res.status} ${res.statusText}`
          throw new Error(String(detail))
        }

        setPublishByReportId((prev) => ({
          ...prev,
          [reportId]: { status: 'published', publishedAt: data?.published_at },
        }))

        // Stage tracker: show Published state on success.
        setStage('published')

        addMessage('agent', `Published report ${reportId} at ${data?.published_at || ''}`.trim(), { statusText: 'Published.' })
      } catch (e) {
        setPublishByReportId((prev) => ({
          ...prev,
          [reportId]: { status: 'error', error: `Publish failed: ${String(e?.message || e)}` },
        }))
      }
    },
    [API_BASE, addMessage, lastResponse],
  )

  const handleRequestRevision = useCallback(
    (message) => {
      const mid = message?.id
      if (!mid) return
      setReviseUI({ open: true, messageId: mid, text: '' })
    },
    [setReviseUI],
  )

  const submitRevision = useCallback(async () => {
    const message = messages.find((m) => m.id === reviseUI.messageId)
    const reportId = String(
      message?.reportId ||
        lastResponse?.report?.id ||
        lastResponse?.report?.report_id ||
        lastResponse?.metadata?.evaluation_id ||
        '',
    )
    if (!message || !reportId) {
      setReviseUI({ open: false, messageId: null, text: '' })
      return
    }

    const note = (reviseUI.text || '').trim()

    // Preserve history: append user revision request, then start a new run.
    addMessage('user', `Revision request for ${reportId}: ${note}`.trim())

    // IMPORTANT: for revision, clear any prior run's evaluation/metadata so we don't show stale scoring.
    setEvaluation(null)
    setJudgeMetadata(null)

    stopCurrentRun()
    resetPanelsForNewRun()

    // IMPORTANT: revisions are hard-coded server-side to run exactly 1 iteration.
    // Ensure the UI counter matches the actual revision run.
    setMaxRevisions(1)

    setStatusMode('working')

    // IMPORTANT: stage must reflect the real first step.
    // The /revise endpoint is synchronous (not SSE) and the first actual work is planning.
    // Showing policy_guard here caused the timeline to appear wrong.
    setStage('planning')
    setIterationCount(0)

    runRef.current.startedAt = performance.now()
    elapsedTimerRef.current = window.setInterval(() => {
      const started = runRef.current.startedAt
      if (started != null) setElapsedMs(Math.max(0, performance.now() - started))
    }, 250)

    const agentRunId = `${Date.now()}-${Math.random().toString(16).slice(2)}`
    setMessages((prev) => [
      ...prev,
      {
        id: agentRunId,
        role: 'agent',
        text: 'Working on the revision…',
        statusText: 'Starting revision…',
      },
    ])

    try {
      const base = API_BASE || window.location.origin
      const url = `${base}/api/reports/${encodeURIComponent(String(reportId))}/revise`
      console.log('[revise] POST', url)
      // If you see “Method Not Allowed”, your backend likely doesn't have this route or expects a different method.
      // This log helps confirm the exact URL being hit.
      const payload = {
        query: String(message?.topic || lastResponse?.query || '').trim() || 'report',
        user_note: note,
        // IMPORTANT: send the prior *public* report content to the backend as revision context.
        // `lastResponse.report` is the dashboard report blocks, while the revise endpoint expects a PublicReport shape.
        report: lastResponse?.public_report || {},
        evaluation: message?.evaluation || {},
        metadata: message?.judgeMetadata || null,
        sources: lastResponse?.sources || [],
      }

      // Some environments intercept a normal POST here (seen as GET in server logs).
      // Force fetch with explicit options to avoid accidental form submission behavior.
      const res = await fetch(url, {
        method: 'POST',
        mode: 'cors',
        cache: 'no-store',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ ...payload, max_revisions: 1 }),
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const detail = data?.detail?.message || data?.detail || data?.message || `${res.status} ${res.statusText}`
        throw new Error(String(detail))
      }

      // If backend includes structured execution_trace, keep it on lastResponse
      // (the dashboard currently renders from traceLines, but we may expand it later).
      // No-op here; just leaving a clear hook.

      // Preserve previous response fields if the revise endpoint omits any optional ones.
      setLastResponse((prev) => ({ ...(prev || {}), ...(data || {}) }))

      // Revision should show full execution details like a normal run.
      // Use trace lines (existing dashboard renderer relies on these).
      setTraceLines(normalizeTrace(data.trace || []).map(cleanLine).filter(Boolean))
      setIterationCount(data.iteration_count ?? 0)
      setSources(data.sources || [])
      setCritic(data.critic || null)

      // Explicitly show that both judge feedback and the user's note were used.
      // Prefer the local note (source of truth from the modal) and only fall back to backend echo.
      const noteEcho = String(note || '').trim() || String(data?.metadata?.revision_user_note || '').trim()
      if (noteEcho) {
        addMessage('agent', `Revision inputs applied: user_note="${noteEcho}" (plus prior judge feedback).`, {
          statusText: 'Revision context applied.',
        })
      } else {
        addMessage('agent', 'Revision inputs applied: prior judge feedback (no user note provided).', {
          statusText: 'Revision context applied.',
        })
      }

      // IMPORTANT: /revise now returns the full report payload (blocks/limitations) at top-level `report`.
      setReport(data.report || { key_findings: [], evidence_and_sources: [], limitations: [] })
      setEvaluation(data.evaluation || null)
      setJudgeMetadata(data.metadata || null)

      // Revision is treated exactly like a normal pass, so show full timeline.
      setStage(data.evaluation ? 'judge_step' : 'report_step')

      const blocks = (data.report && Array.isArray(data.report.blocks) && data.report.blocks) || []

      setMessages((prev) =>
        prev.map((m) =>
          m.id === agentRunId
            ? {
                ...m,
                text: 'Revised report ready.',
                reportBlocks: blocks,
                evaluation: data.evaluation || undefined,
                judgeMetadata: data.metadata || undefined,
                topic: data.report?.topic || data.query || undefined,
                reportId: data.report?.id || undefined,
                statusText: 'Done.',
                isRevision: true,
              }
            : m,
        ),
      )
    } catch (e) {
      addMessage('agent', `Revision failed: ${String(e?.message || e)}`, { statusText: 'Error.' })
    } finally {
      setReviseUI({ open: false, messageId: null, text: '' })
      stopCurrentRun()
    }
  }, [API_BASE, addMessage, lastResponse, messages, resetPanelsForNewRun, reviseUI.messageId, reviseUI.text, stopCurrentRun])

  // Legacy flags (no longer used since ChatPanel drives Send)
  // const sendDisabled = statusMode === 'working'
  // const stopVisible = statusMode === 'working'

  return (
    <>
      {/* Top Navigation Bar */}
      <header className="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#233648] px-6 py-3 shrink-0 bg-background-dark">
        <div className="flex items-center gap-4 text-white">
          <div className="size-6 text-primary">
            <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M39.5563 34.1455V13.8546C39.5563 15.708 36.8773 17.3437 32.7927 18.3189C30.2914 18.916 27.263 19.2655 24 19.2655C20.737 19.2655 17.7086 18.916 15.2073 18.3189C11.1227 17.3437 8.44365 15.708 8.44365 13.8546V34.1455C8.44365 35.9988 11.1227 37.6346 15.2073 38.6098C17.7086 39.2069 20.737 39.5564 24 39.5564C27.263 39.5564 30.2914 39.2069 32.7927 38.6098C36.8773 37.6346 39.5563 35.9988 39.5563 34.1455Z"
                fill="currentColor"
              ></path>
              <path
                clipRule="evenodd"
                d="M10.4485 13.8519C10.4749 13.9271 10.6203 14.246 11.379 14.7361C12.298 15.3298 13.7492 15.9145 15.6717 16.3735C18.0007 16.9296 20.8712 17.2655 24 17.2655C27.1288 17.2655 29.9993 16.9296 32.3283 16.3735C34.2508 15.9145 35.702 15.3298 36.621 14.7361C37.3796 14.246 37.5251 13.9271 37.5515 13.8519C37.5287 13.7876 37.4333 13.5973 37.0635 13.2931C36.5266 12.8516 35.6288 12.3647 34.343 11.9175C31.79 11.0295 28.1333 10.4437 24 10.4437C19.8667 10.4437 16.2099 11.0295 13.657 11.9175C12.3712 12.3647 11.4734 12.8516 10.9365 13.2931C10.5667 13.5973 10.4713 13.7876 10.4485 13.8519ZM37.5563 18.7877C36.3176 19.3925 34.8502 19.8839 33.2571 20.2642C30.5836 20.9025 27.3973 21.2655 24 21.2655C20.6027 21.2655 17.4164 20.9025 14.7429 20.2642C13.1498 19.8839 11.6824 19.3925 10.4436 18.7877V34.1275C10.4515 34.1545 10.5427 34.4867 11.379 35.027C12.298 35.6207 13.7492 36.2054 15.6717 36.6644C18.0007 37.2205 20.8712 37.5564 24 37.5564C27.1288 37.5564 29.9993 37.2205 32.3283 36.6644C34.2508 36.2054 35.702 35.6207 36.621 35.027C37.4573 34.4867 37.5485 34.1546 37.5563 34.1275V18.7877ZM41.5563 13.8546V34.1455C41.5563 36.1078 40.158 37.5042 38.7915 38.3869C37.3498 39.3182 35.4192 40.0389 33.2571 40.5551C30.5836 41.1934 27.3973 41.5564 24 41.5564C20.6027 41.5564 17.4164 41.1934 14.7429 40.5551C12.5808 40.0389 10.6502 39.3182 9.20848 38.3869C7.84205 37.5042 6.44365 36.1078 6.44365 34.1455L6.44365 13.8546C6.44365 12.2684 7.37223 11.0454 8.39581 10.2036C9.43325 9.3505 10.8137 8.67141 12.343 8.13948C15.4203 7.06909 19.5418 6.44366 24 6.44366C28.4582 6.44366 32.5797 7.06909 35.657 8.13948C37.1863 8.67141 38.5667 9.3505 39.6042 10.2036C40.6278 11.0454 41.5563 12.2684 41.5563 13.8546Z"
                fill="currentColor"
                fillRule="evenodd"
              ></path>
            </svg>
          </div>
          <h2 className="text-white text-lg font-bold leading-tight tracking-[-0.015em]">Deep Research Agent</h2>

          {/* Connection badge */}
          <div
            id="conn-badge"
            title={connDetail}
            className={
              connState === 'connected'
                ? 'flex items-center gap-1.5 px-2 py-0.5 rounded-full border bg-green-500/10 border-green-500/20'
                : connState === 'disconnected'
                  ? 'flex items-center gap-1.5 px-2 py-0.5 rounded-full border bg-red-500/10 border-red-500/20'
                  : 'flex items-center gap-1.5 px-2 py-0.5 rounded-full border bg-amber-500/10 border-amber-500/20'
            }
          >
            <div
              id="conn-dot"
              className={
                connState === 'connected'
                  ? 'size-2 rounded-full bg-green-500'
                  : connState === 'disconnected'
                    ? 'size-2 rounded-full bg-red-500'
                    : 'size-2 rounded-full bg-amber-500'
              }
            ></div>
            <span
              id="conn-text"
              className={
                connState === 'connected'
                  ? 'text-[11px] font-bold text-green-500 uppercase tracking-wider'
                  : connState === 'disconnected'
                    ? 'text-[11px] font-bold text-red-500 uppercase tracking-wider'
                    : 'text-[11px] font-bold text-amber-500 uppercase tracking-wider'
              }
            >
              {connState === 'connected' ? 'Connected' : connState === 'disconnected' ? 'Disconnected' : 'Checking…'}
            </span>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            className="flex items-center justify-center rounded-lg h-9 w-9 bg-[#233648] text-white hover:bg-primary/20 transition-colors"
            title="Settings (not wired)"
            type="button"
          >
            <span className="material-symbols-outlined text-[20px]">settings</span>
          </button>
          <button
            className="flex items-center justify-center rounded-lg h-9 w-9 bg-[#233648] text-white hover:bg-primary/20 transition-colors"
            title="Profile (not wired)"
            type="button"
          >
            <span className="material-symbols-outlined text-[20px]">person</span>
          </button>
        </div>
      </header>

      <main className="flex flex-1 overflow-hidden min-h-0" style={{ height: 'calc(100vh - 57px)' }}>
        {/* Revision modal */}
        {reviseUI.open ? (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center"
            style={{ background: 'rgba(0,0,0,0.55)' }}
            role="dialog"
            aria-modal="true"
            aria-label="Request revision"
          >
            <div className="w-[min(680px,92vw)] rounded-xl border border-[#233648] bg-[#0f1a25] p-4">
              <div className="text-sm font-bold text-white">Request revision</div>
              <div className="text-[12px] text-white/70 mt-1">Describe what should change in the next version.</div>
              <textarea
                className="mt-3 w-full rounded-lg bg-[#111a22] border border-[#233648] p-3 text-sm text-white/90"
                rows={5}
                value={reviseUI.text}
                onChange={(e) => setReviseUI((prev) => ({ ...prev, text: e.target.value }))}
                placeholder="e.g., Address the judge weaknesses, add missing citations, tighten scope…"
              />
              <div className="mt-3 flex items-center justify-end gap-2">
                <button
                  type="button"
                  className="px-3 py-2 rounded-lg bg-[#233648] border border-[#324d67] text-[#92adc9] hover:text-white hover:border-primary text-xs font-bold"
                  onClick={() => setReviseUI({ open: false, messageId: null, text: '' })}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="px-3 py-2 rounded-lg bg-primary text-white text-xs font-bold disabled:opacity-60"
                  disabled={statusMode === 'working'}
                  onClick={(e) => {
                    e.preventDefault()
                    submitRevision()
                  }}
                >
                  Submit revision
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {/* Left Panel: Chat Interface (redone) */}
        <section className="w-1/2 flex flex-col min-h-0 border-r border-[#233648] bg-background-dark/50">
          <ChatPanel
            messages={messages.map((m) =>
              m.role === 'agent' && m.reportId
                ? {
                    ...m,
                    publishState: publishByReportId[m.reportId] || { status: 'idle' },
                  }
                : m,
            )}
            draft={draft}
            onDraftChange={setDraft}
            maxRevisions={maxRevisions}
            onMaxRevisionsChange={setMaxRevisions}
            onSend={runResearch}
            statusMode={statusMode}
            stage={stage}
            onPublish={handlePublish}
            onRequestRevision={handleRequestRevision}
          />
        </section>

        {/* Right Panel: Dashboard */}
        <section className="w-1/2 flex flex-col min-h-0 bg-background-dark custom-scrollbar overflow-y-auto">
          {/* Sticky Dashboard Header */}
          {stageKey === 'blocked' ? (
            <div className="p-6">
              <div className="border border-[#233648] rounded-xl bg-[#111a22] p-4 text-sm text-white/90">
                Request blocked.
              </div>
            </div>
          ) : (
            <div>
              <div className="sticky top-0 z-10 bg-background-dark/95 backdrop-blur-md border-b border-[#233648] p-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div
                    id="statusBadge"
                    className={
                      statusMode === 'working'
                        ? 'flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/20 border border-primary/30'
                        : 'flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#233648] border border-[#324d67]'
                    }
                  >
                    <span
                      id="statusIcon"
                      className={
                        statusMode === 'working'
                          ? 'material-symbols-outlined text-primary text-[18px] animate-spin'
                          : 'material-symbols-outlined text-[#92adc9] text-[18px]'
                      }
                    >
                      progress_activity
                    </span>
                    <span
                      id="statusText"
                      className={
                        statusMode === 'working'
                          ? 'text-xs font-bold text-primary uppercase tracking-widest'
                          : 'text-xs font-bold text-[#92adc9] uppercase tracking-widest'
                      }
                    >
                      {statusMode === 'working' ? 'Researching…' : 'Idle'}
                    </span>
                  </div>
                  <div className="h-6 w-px bg-[#233648]"></div>
                  <div className="flex flex-col">
                    <span className="text-[10px] text-[#92adc9] font-bold uppercase">Iteration</span>
                    <span id="iterText" className="text-sm font-bold text-white">
                      {iterationCount} / {maxRevisions}
                    </span>
                  </div>
                  <div className="flex flex-col ml-4">
                    <span className="text-[10px] text-[#92adc9] font-bold uppercase">Elapsed</span>
                    <span id="elapsedText" className="text-sm font-mono text-white">
                      {fmtElapsed(elapsedMs)}
                    </span>
                  </div>
                </div>
                <button
                  id="exportBtn"
                  type="button"
                  onClick={onExport}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#233648] text-white hover:bg-[#324d67] text-xs font-bold transition-colors"
                >
                  <span className="material-symbols-outlined text-[18px]">download</span>
                  Export Data
                </button>
              </div>

              {/* Execution Timeline */}
              <div className="px-6 pt-6">
            <h3 className="text-xs font-bold text-[#92adc9] uppercase tracking-widest">Execution Timeline</h3>
            <div className="grid grid-cols-6 gap-3 mt-4">
              <div
                id="card-planning"
                className={
                  stageKey === 'planning'
                    ? 'bg-[#1a2632] border border-primary/50 rounded-lg p-3 relative ring-1 ring-primary/20'
                    : 'bg-[#1a2632] border border-[#233648] rounded-lg p-3 relative opacity-70'
                }
              >
                <div id="check-planning" className={`${completedStages.has('planning') ? '' : 'hidden '}absolute top-2 right-2 text-green-500`}>
                  <span className="material-symbols-outlined text-[16px]">check_circle</span>
                </div>
                <span className="text-[10px] text-[#92adc9] font-bold uppercase">Stage 1</span>
                <p className="text-sm font-bold mt-1">Plan</p>
                <div className="mt-3 h-1 w-full bg-[#233648] rounded-full overflow-hidden">
                  <div
                    id="bar-planning"
                    className={
                      stageKey === 'planning' ? 'h-full bg-primary w-2/3 animate-pulse' : 'h-full bg-primary w-0'
                    }
                  ></div>
                </div>
                <p id="sub-planning" className="text-[11px] text-[#92adc9] mt-2">
                  {stageKey === 'planning' ? 'Working…' : '—'}
                </p>
              </div>

              <div
                id="card-search"
                className={
                  stageKey === 'search'
                    ? 'bg-[#1a2632] border border-primary/50 rounded-lg p-3 relative ring-1 ring-primary/20'
                    : 'bg-[#1a2632] border border-[#233648] rounded-lg p-3 relative opacity-70'
                }
              >
                <div id="check-search" className={`${completedStages.has('search') ? '' : 'hidden '}absolute top-2 right-2 text-green-500`}>
                  <span className="material-symbols-outlined text-[16px]">check_circle</span>
                </div>
                <span className="text-[10px] text-[#92adc9] font-bold uppercase">Stage 2</span>
                <p className="text-sm font-bold mt-1">Search</p>
                <div className="mt-3 h-1 w-full bg-[#233648] rounded-full overflow-hidden">
                  <div
                    id="bar-search"
                    className={stageKey === 'search' ? 'h-full bg-primary w-2/3 animate-pulse' : 'h-full bg-primary w-0'}
                  ></div>
                </div>
              </div>

              <div
                id="card-critic"
                className={
                  stageKey === 'critic_step'
                    ? 'bg-[#1a2632] border border-primary/50 rounded-lg p-3 relative ring-1 ring-primary/20'
                    : 'bg-[#1a2632] border border-[#233648] rounded-lg p-3 relative opacity-70'
                }
              >
                <div id="check-critic" className={`${completedStages.has('critic_step') ? '' : 'hidden '}absolute top-2 right-2 text-green-500`}>
                  <span className="material-symbols-outlined text-[16px]">check_circle</span>
                </div>
                <span className="text-[10px] text-[#92adc9] font-bold uppercase">Stage 3</span>
                <p className="text-sm font-bold mt-1">Critic</p>
                <div className="mt-3 h-1 w-full bg-[#233648] rounded-full overflow-hidden">
                  <div
                    id="bar-critic"
                    className={
                      stageKey === 'critic_step' ? 'h-full bg-primary w-2/3 animate-pulse' : 'h-full bg-primary w-0'
                    }
                  ></div>
                </div>
                <p id="sub-critic" className="text-[11px] text-[#92adc9] mt-2">
                  {stageKey === 'critic_step' ? 'Working…' : '—'}
                </p>
              </div>

              <div
                id="card-report"
                className={
                  stageKey === 'report_step'
                    ? 'bg-[#1a2632] border border-primary/50 rounded-lg p-3 relative ring-1 ring-primary/20'
                    : 'bg-[#1a2632] border border-[#233648] rounded-lg p-3 relative opacity-70'
                }
              >
                <div id="check-report" className={`${completedStages.has('report_step') ? '' : 'hidden '}absolute top-2 right-2 text-green-500`}>
                  <span className="material-symbols-outlined text-[16px]">check_circle</span>
                </div>
                <span className="text-[10px] text-[#92adc9] font-bold uppercase">Stage 4</span>
                <p className="text-sm font-bold mt-1">Report</p>
                <p id="sub-report" className="text-[11px] text-[#92adc9] mt-2">
                  {stageKey === 'report_step' ? 'Working…' : '—'}
                </p>
              </div>

              <div
                id="card-judge"
                className={
                  stageKey === 'judge_step'
                    ? 'bg-[#1a2632] border border-primary/50 rounded-lg p-3 relative ring-1 ring-primary/20'
                    : 'bg-[#1a2632] border border-[#233648] rounded-lg p-3 relative opacity-70'
                }
              >
                <div id="check-judge" className={`${completedStages.has('judge_step') ? '' : 'hidden '}absolute top-2 right-2 text-green-500`}>
                  <span className="material-symbols-outlined text-[16px]">check_circle</span>
                </div>
                <span className="text-[10px] text-[#92adc9] font-bold uppercase">Stage 5</span>
                <p className="text-sm font-bold mt-1">Judge</p>
                <p id="sub-judge" className="text-[11px] text-[#92adc9] mt-2">
                  {stageKey === 'judge_step' ? 'Working…' : '—'}
                </p>
              </div>

              <div
                id="card-published"
                className={
                  stageKey === 'published'
                    ? 'bg-[#1a2632] border border-primary/50 rounded-lg p-3 relative ring-1 ring-primary/20'
                    : 'bg-[#1a2632] border border-[#233648] rounded-lg p-3 relative opacity-70'
                }
              >
                <div
                  id="check-published"
                  className={`${completedStages.has('published') ? '' : 'hidden '}absolute top-2 right-2 text-green-500`}
                >
                  <span className="material-symbols-outlined text-[16px]">check_circle</span>
                </div>
                <span className="text-[10px] text-[#92adc9] font-bold uppercase">Stage 6</span>
                <p className="text-sm font-bold mt-1">Published</p>
                <p id="sub-published" className="text-[11px] text-[#92adc9] mt-2">
                  {stageKey === 'published' ? 'Done.' : '—'}
                </p>
              </div>
            </div>
          </div>

              <div className="p-6 space-y-8">
            {/* Critic Snapshot */}
            <div className="space-y-4">
              <h3 className="text-xs font-bold text-[#92adc9] uppercase tracking-widest">Critic Snapshot</h3>
              <div className="bg-[#1a2632] border border-[#233648] rounded-xl p-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-[#92adc9] font-bold uppercase">Latest Critic Action</span>
                    <div
                      id="criticDecision"
                      className="mt-1 inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#111a22] border border-[#324d67] text-[#92adc9] font-mono text-xs"
                    >
                      {critic?.decision || '—'}
                    </div>
                  </div>
                  <div className="h-8 w-px bg-[#324d67]"></div>
                  <div className="flex flex-col w-56">
                    <div className="flex items-center justify-between text-[10px] font-bold uppercase mb-1">
                      <span className="text-[#92adc9]">Sufficiency Score</span>
                      <span id="criticScoreText" className="text-primary">
                        {critic ? `${Number(critic.sufficiency_score ?? 0)}%` : '—'}
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-[#111a22] rounded-full">
                      <div
                        id="criticScoreBar"
                        className="h-full bg-primary rounded-full"
                        style={{ width: critic ? `${Math.max(0, Math.min(100, Number(critic.sufficiency_score ?? 0)))}%` : '0%' }}
                      ></div>
                    </div>
                  </div>
                </div>
                <div className="flex flex-col items-end">
                  <span className="text-[10px] text-[#92adc9] font-bold uppercase">Missing Info</span>
                  <ul id="missingInfo" className="text-[11px] text-white/80 text-right mt-1">
                    {(critic?.missing_gaps || []).map((g, idx) => (
                      <li key={idx}>{`• ${g.description || g.suggested_query || g.gap_id}`}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Execution Trace */}
            <div className="space-y-4">
              <h3 className="text-xs font-bold text-[#92adc9] uppercase tracking-widest">Execution Trace</h3>
              <div className="bg-[#1a2632] border border-[#233648] rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <button
                      id="traceExpandAll"
                      type="button"
                      onClick={() => {
                        const next = {}
                        for (const k of parsedTrace.order) next[k] = true
                        setTraceOpen(next)
                      }}
                      className="px-2 py-1 rounded bg-[#233648] border border-[#324d67] text-[11px] font-bold text-[#92adc9] hover:text-white hover:border-primary"
                    >
                      Expand all
                    </button>
                    <button
                      id="traceCollapseAll"
                      type="button"
                      onClick={() => {
                        const next = {}
                        for (const k of parsedTrace.order) next[k] = false
                        setTraceOpen(next)
                      }}
                      className="px-2 py-1 rounded bg-[#233648] border border-[#324d67] text-[11px] font-bold text-[#92adc9] hover:text-white hover:border-primary"
                    >
                      Collapse all
                    </button>
                  </div>
                  <div className="text-[11px] text-[#92adc9] font-bold" id="traceSummary">
                    {(() => {
                      const c = parsedTrace.counts
                      const fetchPart = `${c.fetchOk} ok / ${c.fetchTotal} total`
                      return `Searches: ${c.searches} • Sources: ${c.sources} • Fetch: ${fetchPart}`
                    })()}
                  </div>
                </div>

                <div id="traceSections" className="space-y-2">
                  {parsedTrace.order.map((name) => {
                    const items = parsedTrace.sections[name] || []
                    const open = !!traceOpen[name]

                    return (
                      <details
                        key={name}
                        data-trace-section=""
                        open={open}
                        onToggle={(e) => {
                          const isOpen = e.currentTarget.open
                          setTraceOpen((prev) => ({ ...prev, [name]: isOpen }))
                        }}
                        className="bg-[#111a22] border border-[#233648] rounded-lg"
                      >
                        <summary className="cursor-pointer select-none flex items-center justify-between gap-2 px-3 py-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span
                              className="material-symbols-outlined text-[18px] text-[#92adc9]"
                              style={{
                                transform: open ? 'rotate(90deg)' : 'rotate(0deg)',
                                transition: 'transform 120ms ease',
                              }}
                            >
                              chevron_right
                            </span>
                            <span className="text-[11px] font-bold text-[#92adc9] uppercase tracking-widest">{name}</span>
                          </div>
                          <div className="shrink-0 text-[11px] bg-[#233648] px-2 py-0.5 rounded border border-[#324d67] text-[#92adc9] font-bold">
                            {items.length}
                          </div>
                        </summary>

                        <div className="px-3 pb-3">
                          {items.length === 0 ? (
                            <div className="text-[11px] text-white/50 py-2">(none)</div>
                          ) : (
                            <div className="space-y-2">
                              {items.map((it, idx) => {
                                const line = it.line

                                if (name === 'Sources Considered' && /^Source considered:/i.test(line)) {
                                  const rest = line.replace(/^Source considered:\s*/i, '')
                                  const parts = rest.split('|').map((s) => s.trim())
                                  const title = parts[0] || '(untitled)'
                                  const url = (parts[1] || '').trim()
                                  const domain = url ? safeHostname(url) : ''

                                  return (
                                    <div
                                      key={idx}
                                      className="flex items-center justify-between gap-3 bg-[#1a2632] border border-[#233648] rounded-md px-3 py-2"
                                    >
                                      <div className="min-w-0">
                                        <div className="text-[12px] text-white/90 font-medium truncate" title={title}>
                                          {title}
                                        </div>
                                      </div>
                                      <div className="shrink-0 flex items-center gap-2">
                                        {domain ? (
                                          <span className="px-2 py-0.5 rounded-full bg-[#233648] border border-[#324d67] text-[10px] font-bold text-[#92adc9]">
                                            {domain}
                                          </span>
                                        ) : null}
                                        {url ? (
                                          <a
                                            href={url}
                                            target="_blank"
                                            rel="noreferrer"
                                            title="Open"
                                            className="text-[#92adc9] hover:text-white"
                                          >
                                            <span className="material-symbols-outlined text-[18px]">open_in_new</span>
                                          </a>
                                        ) : null}
                                        {url ? (
                                          <button
                                            type="button"
                                            onClick={() => onCopyUrl(url)}
                                            title="Copy URL"
                                            className="text-[#92adc9] hover:text-white"
                                          >
                                            <span className="material-symbols-outlined text-[18px]">content_copy</span>
                                          </button>
                                        ) : null}
                                      </div>
                                    </div>
                                  )
                                }

                                if (name === 'Fetch Results' && /^Fetched:/i.test(line)) {
                                  const urlMatch = line.match(/Fetched:\s*(https?:\/\/[^\s)]+)/i)
                                  const url = urlMatch ? urlMatch[1] : null
                                  const statusMatch = line.match(/status=(\d{3})/i)
                                  const status = statusMatch ? statusMatch[1] : null
                                  const ctMatch = line.match(/content_type=([^\s)]+)/i)
                                  const ct = ctMatch ? ctMatch[1] : null

                                  const link = url ? compactUrlParts(url) : null

                                  return (
                                    <div
                                      key={idx}
                                      className="flex items-center justify-between gap-3 bg-[#1a2632] border border-[#233648] rounded-md px-3 py-2"
                                    >
                                      <div className="min-w-0 text-[12px] text-white/90">
                                        {url && link ? (
                                          <span className="inline-flex items-center gap-1">
                                            <a
                                              className="text-primary hover:underline break-all"
                                              href={url}
                                              target="_blank"
                                              rel="noreferrer"
                                              title={url}
                                            >
                                              {`${link.host}${link.path}`}
                                            </a>
                                            <button
                                              type="button"
                                              className="ml-1 text-[#92adc9] hover:text-white"
                                              onClick={() => onCopyUrl(url)}
                                              title="Copy URL"
                                            >
                                              <span className="material-symbols-outlined text-[16px]">content_copy</span>
                                            </button>
                                          </span>
                                        ) : (
                                          <span>{line}</span>
                                        )}
                                      </div>
                                      <div className="shrink-0 flex items-center gap-2">
                                        {status ? (
                                          <span
                                            className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${statusPillClass(status)}`}
                                          >
                                            {status}
                                          </span>
                                        ) : null}
                                        {ct ? (
                                          <span className="text-[10px] text-white/50" title={ct}>
                                            {ct}
                                          </span>
                                        ) : null}
                                      </div>
                                    </div>
                                  )
                                }

                                // Generic line rendering with URL compaction
                                const urls = extractUrls(line)
                                let text = line
                                for (const u of urls) text = text.replaceAll(u, '')
                                text = text.replace(/\s{2,}/g, ' ').trim()

                                return (
                                  <div key={idx} className="bg-[#1a2632] border border-[#233648] rounded-md px-3 py-2">
                                    <div className="text-[12px] text-white/90 leading-relaxed">{text || line}</div>
                                    {urls.length ? (
                                      <div className="mt-1 flex flex-wrap gap-2">
                                        {urls.map((u) => {
                                          const link = compactUrlParts(u)
                                          const label = `${link.host}${link.path}`
                                          return (
                                            <span key={u} className="inline-flex items-center gap-1">
                                              <a
                                                className="text-primary hover:underline break-all"
                                                href={u}
                                                target="_blank"
                                                rel="noreferrer"
                                                title={u}
                                              >
                                                {label}
                                              </a>
                                              <button
                                                type="button"
                                                className="ml-1 text-[#92adc9] hover:text-white"
                                                onClick={() => onCopyUrl(u)}
                                                title="Copy URL"
                                              >
                                                <span className="material-symbols-outlined text-[16px]">content_copy</span>
                                              </button>
                                            </span>
                                          )
                                        })}
                                      </div>
                                    ) : null}
                                  </div>
                                )
                              })}
                            </div>
                          )}
                        </div>
                      </details>
                    )
                  })}
                </div>

                <pre id="traceRaw" className="hidden whitespace-pre-wrap text-[10px] text-white/70">
                  {traceLines.join('\n')}
                </pre>
              </div>
            </div>

            {/* Final Report (moved above Sources; highest priority) */}
            <div className="space-y-4" id="finalReportSection">
              <h3 className="text-xs font-bold text-[#92adc9] uppercase tracking-widest">Final Report</h3>
              <div className="border border-[#233648] rounded-xl bg-[#111a22] p-4">
                <div className="space-y-5">
                  {/* New block-based report rendering with per-paragraph citations */}
                  {Array.isArray(report?.blocks) && report.blocks.length > 0 ? (
                    <div className="space-y-4">
                      {report.blocks.map((b) => {
                        const heading = String(b?.heading || '').trim()
                        const text = String(b?.text || '').trim()
                        if (!heading && !text) return null
                        const citations = Array.isArray(b?.citations) ? b.citations : []

                        return (
                          <div key={String(b.id || (heading || text).slice(0, 16))} className="space-y-2">
                            {heading ? <div className="text-xs font-bold text-primary">{heading}</div> : null}
                            {text ? (
                              <p className="text-sm text-white/90 leading-relaxed">{text.replace(/^\*\*[^*]{1,120}\*\*:\s*/, '')}</p>
                            ) : null}
                            {citations.length ? (
                              <div className="flex flex-wrap gap-2">
                                {citations.map((c) => {
                                  const url = c?.url
                                  const label = String(c?.id || '').trim() || '↗'
                                  const title = String(c?.title || '')
                                  const domain = url ? safeHostname(String(url)) : ''
                                  const tooltip = [title, domain].filter(Boolean).join(' • ') || String(url || '')
                                  const href = url ? String(url) : null

                                  return href ? (
                                    <a
                                      key={String(c?.id || href)}
                                      href={href}
                                      target="_blank"
                                      rel="noreferrer"
                                      title={tooltip}
                                      className="shrink-0 w-7 h-7 rounded-full bg-[#233648] border border-[#324d67] text-[#92adc9] flex items-center justify-center text-[10px] font-bold hover:border-primary hover:text-white hover:bg-primary/20 transition-colors"
                                    >
                                      {label}
                                    </a>
                                  ) : null
                                })}
                              </div>
                            ) : null}
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    // Legacy fallback
                    <div className="space-y-3">
                      <div>
                        <div className="text-[11px] text-[#92adc9] font-bold uppercase">Key findings</div>
                        <ul id="reportFindings" className="mt-2 text-sm space-y-2">
                          {(report?.key_findings || []).map((raw, idx) => {
                            const text = String(raw || '').trim()
                            if (!text) return null
                            const m = text.match(/^\*\*(.+?)\*\*:\s*(.+)$/)
                            return (
                              <li key={idx} className="rounded-lg border border-[#233648] bg-[#1a2632] px-3 py-2">
                                {m ? (
                                  <>
                                    <div className="text-xs font-bold text-primary">{m[1].trim()}</div>
                                    <div className="text-sm text-white/90 mt-1 leading-relaxed">{m[2].trim()}</div>
                                  </>
                                ) : (
                                  <div className="text-sm text-white/90 leading-relaxed">{text}</div>
                                )}
                              </li>
                            )
                          })}
                        </ul>
                      </div>
                    </div>
                  )}

                  <div>
                    <div className="text-[11px] text-[#92adc9] font-bold uppercase">Limitations</div>
                    <ul id="reportLimitations" className="mt-2 text-sm space-y-1">
                      {(report?.limitations || []).map((it, idx) => (
                        <li key={idx}>{it}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Sources Table (hidden by default; only show when report has no blocks) */}
            {Array.isArray(report?.blocks) && report.blocks.length > 0 ? null : (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-[#92adc9] uppercase tracking-widest">Sources</h3>
                  <span
                    id="sourcesCount"
                    className="text-[11px] bg-[#111a22] px-2 py-0.5 rounded border border-[#324d67] text-[#92adc9]"
                  >
                    {sources.length} result{sources.length === 1 ? '' : 's'}
                  </span>
                </div>
                <div className="bg-[#1a2632] border border-[#233648] rounded-xl overflow-hidden">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#111a22] text-[#92adc9] font-bold uppercase">
                      <tr>
                        <th className="px-4 py-3">Source Title</th>
                        <th className="px-4 py-3">Domain</th>
                        <th className="px-4 py-3 text-right">URL</th>
                      </tr>
                    </thead>
                    <tbody id="sourcesBody" className="divide-y divide-[#233648]">
                      {sources.map((s, idx) => {
                        const url = s.url || ''
                        const domain = url ? safeHostname(url) : ''
                        const title = s.title || ''
                        return (
                          <tr key={idx} className="hover:bg-[#233648]/30 transition-colors">
                            <td className="px-4 py-3 font-medium truncate max-w-[220px]" title={title}>
                              {title}
                            </td>
                            <td className="px-4 py-3 text-[#92adc9]">{domain}</td>
                            <td className="px-4 py-3 text-right">
                              <a className="text-primary hover:underline" href={url} target="_blank" rel="noreferrer">
                                open
                              </a>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Errors */}
            <div
              id="errorPanel"
              className={`${errorMsg ? '' : 'hidden '}border border-red-500/40 rounded-xl bg-red-500/10 p-4`}
            >
              <div className="text-[11px] text-red-200 font-bold uppercase">Error</div>
              <pre id="errorText" className="mt-2 text-xs whitespace-pre-wrap text-red-100">
                {errorMsg}
              </pre>
            </div>
          </div>
            </div>
          )}
        </section>
      </main>
    </>
  )
}
