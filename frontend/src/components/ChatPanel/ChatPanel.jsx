import { useEffect, useMemo, useRef } from 'react'
import styles from './ChatPanel.module.css'

/**
 * @typedef {{ id: string; url: string; title?: string }} Citation
 * @typedef {{ id: string; heading?: string; text: string; citations: Citation[] }} ReportBlock
 *
 * @typedef {(
 *   | { id: string; role: 'user'; text: string }
 *   | { id: string; role: 'agent'; text: string; statusText?: string; reportBlocks?: ReportBlock[] }
 * )} Message
 */

/**
 * @typedef {Object} ChatPanelProps
 * @property {Message[]} messages
 * @property {string} draft
 * @property {(value: string) => void} onDraftChange
 * @property {number} maxRevisions
 * @property {(value: number) => void} onMaxRevisionsChange
 * @property {() => void} onSend
 * @property {'idle'|'working'} statusMode
 * @property {string | null | undefined} stage
 */

/**
 * Controlled chat UI.
 * All backend integration happens in the parent (App) so the chat can update the dashboard.
 *
 * @param {ChatPanelProps} props
 */
export default function ChatPanel({
  messages,
  draft,
  onDraftChange,
  maxRevisions,
  onMaxRevisionsChange,
  onSend,
  statusMode,
  stage,
}) {
  const listRef = useRef(null)

  const canSend = useMemo(() => (draft || '').trim().length > 0 && statusMode !== 'working', [draft, statusMode])

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages.length])

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.logo} aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M12 2c3.4 0 6.2 2.6 6.2 5.8v.8c1 .4 1.8 1.4 1.8 2.6v2.2c0 1.5-1.2 2.8-2.8 2.8H6.8C5.2 16.2 4 15 4 13.4v-2.2c0-1.2.8-2.2 1.8-2.6v-.8C5.8 4.6 8.6 2 12 2Z"
                stroke="currentColor"
                strokeWidth="1.6"
              />
              <path
                d="M8.4 21.2c1.1.5 2.3.8 3.6.8 1.3 0 2.5-.3 3.6-.8"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
              <path
                d="M9 11.4h.01M15 11.4h.01"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <div className={styles.titleBlock}>
            <div className={styles.title}>Deep Research Agent</div>
            <div className={styles.subtitle}>v2.4.0-pro</div>
          </div>
        </div>
      </div>

      <div className={styles.messages} ref={listRef}>
        <div className={styles.gutterAction}>
          <button className={styles.gutterButton} type="button" aria-label="Gutter action (stub)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M7.5 7.8V6.5c0-1 0-1.5.2-2 .2-.4.6-.8 1-1 .5-.2 1-.2 2-.2h2.6c1 0 1.5 0 2 .2.4.2.8.6 1 1 .2.5.2 1 .2 2v1.3"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
              <path
                d="M4.6 9.3c0-.8.6-1.4 1.4-1.4h12c.8 0 1.4.6 1.4 1.4v9.8c0 .8-.6 1.4-1.4 1.4h-12c-.8 0-1.4-.6-1.4-1.4V9.3Z"
                stroke="currentColor"
                strokeWidth="1.6"
              />
              <path d="M9 12.2h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className={styles.messagesInner}>
          {messages.map((m) => {
            const isUser = m.role === 'user'
            return (
              <div key={m.id} className={isUser ? styles.messageGroupUser : styles.messageGroupAgent}>
                <div className={isUser ? styles.labelUser : styles.labelAgent}>{isUser ? 'You' : 'Agent'}</div>

                {isUser ? (
                  <div className={styles.userBubble}>
                    <div className={styles.bubbleText}>{m.text}</div>
                  </div>
                ) : (
                   <div className={styles.agentBubble}>
                     {Array.isArray(m.reportBlocks) && m.reportBlocks.length > 0 ? (
                       <div className={styles.reportBubble}>
                         {m.reportBlocks.map((b) => {
                           const heading = String(b?.heading || '').trim()
                           const text = String(b?.text || '').trim()
                           const citations = Array.isArray(b?.citations) ? b.citations : []
                           if (!heading && !text) return null

                           return (
                             <div key={String(b?.id || (heading || text).slice(0, 16))} className={styles.reportBlock}>
                               {heading ? <div className={styles.reportHeading}>{heading}</div> : null}
                               {text ? <div className={styles.reportText}>{text}</div> : null}
                               {citations.length ? (
                                 <div className={styles.reportCitations}>
                                   {citations.map((c) => {
                                     const url = String(c?.url || '').trim()
                                     if (!url) return null
                                     const label = String(c?.id || '').trim() || '↗'
                                     const title = String(c?.title || '')
                                     return (
                                       <a
                                         key={String(c?.id || url)}
                                         className={styles.citationChip}
                                         href={url}
                                         target="_blank"
                                         rel="noreferrer"
                                         title={title || url}
                                       >
                                         {label}
                                       </a>
                                     )
                                   })}
                                 </div>
                               ) : null}
                             </div>
                           )
                         })}
                       </div>
                     ) : (
                       <div className={styles.bubbleText}>{m.text}</div>
                     )}

                     {m.statusText ? (
                      <div className={styles.agentStatusRow}>
                        <div className={styles.statusIcon} aria-hidden="true">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path
                              d="M10.7 18.2a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Z"
                              stroke="currentColor"
                              strokeWidth="1.8"
                            />
                            <path
                              d="M16.8 16.7 21 20.9"
                              stroke="currentColor"
                              strokeWidth="1.8"
                              strokeLinecap="round"
                            />
                          </svg>
                        </div>
                        <div className={styles.statusText}>{m.statusText}</div>
                      </div>
                    ) : null}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className={styles.composer}>
        <div className={styles.inputSurface}>
          <textarea
            className={styles.textarea}
            placeholder="Ask a follow-up or refine the scope…"
            value={draft}
            onChange={(e) => onDraftChange(e.target.value)}
            rows={3}
            onKeyDown={(e) => {
              // Enter sends; Shift+Enter inserts newline.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                if ((draft || '').trim().length > 0 && statusMode !== 'working') onSend()
              }
            }}
          />
        </div>

        <div className={styles.controls}>
          <div className={styles.controlsLeft}>
            <label className={styles.revisionsPill}>
              <span className={styles.revisionsText}>Max Revisions: {maxRevisions}</span>
              <select
                className={styles.revisionsSelect}
                value={String(maxRevisions)}
                onChange={(e) => onMaxRevisionsChange(Number(e.target.value))}
                aria-label="Max revisions"
              >
                <option value="1">1</option>
                <option value="3">3</option>
                <option value="10">10</option>
              </select>
              <span className={styles.caret} aria-hidden="true">
                ▾
              </span>
            </label>

            <button className={styles.iconButton} type="button" aria-label="Attach (stub)">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M8.4 12.8 14.7 6.5a3.2 3.2 0 0 1 4.5 4.5l-8.4 8.4a5 5 0 0 1-7.1-7.1l8.6-8.6"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>

          <button className={styles.sendButton} type="button" onClick={onSend} disabled={!canSend}>
            <span>{statusMode === 'working' ? 'Running…' : 'Send'}</span>
            <span className={styles.sendIcon} aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4.5 11.8 20 4.6l-7.2 15.5-1.6-6-6.7-2.3Z" fill="currentColor" />
              </svg>
            </span>
          </button>
        </div>

        {statusMode === 'working' && stage ? <div className={styles.runHint}>Stage: {String(stage)}</div> : null}
      </div>
    </div>
  )
}
