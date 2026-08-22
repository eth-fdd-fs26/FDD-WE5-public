import { useEffect, useState } from 'react'
import {
  clearDocuments,
  getDocuments,
  getSearchTypes,
  getStatus,
  ingestFiles,
  runQuery,
} from './api.js'

// Key lives in sessionStorage only: it survives a page reload within the tab
// but is never sent anywhere except as a per-request header to the backend.
const KEY_STORAGE = 'openrouter_key'

const STRATEGY_BLURB = {
  keyword: 'Lexical match (BM25)',
  embedding: 'Semantic similarity',
  hybrid: 'Fused lexical + semantic',
}

export default function App() {
  const [apiKey, setApiKey] = useState(() => sessionStorage.getItem(KEY_STORAGE) || '')
  const [searchTypes, setSearchTypes] = useState(['hybrid'])
  const [searchType, setSearchType] = useState('hybrid')
  const [documents, setDocuments] = useState([])
  const [status, setStatus] = useState(null)

  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState(null)
  const [sources, setSources] = useState([])
  const [askedWith, setAskedWith] = useState('')

  const [busy, setBusy] = useState('') // '', 'ingest', or 'query'
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    sessionStorage.setItem(KEY_STORAGE, apiKey)
  }, [apiKey])

  useEffect(() => {
    getSearchTypes()
      .then((data) => {
        setSearchTypes(data.search_types)
        if (data.search_types.includes('hybrid')) setSearchType('hybrid')
      })
      .catch(() => {})
    getStatus()
      .then(setStatus)
      .catch(() => {})
    refreshDocuments()
  }, [])

  function refreshDocuments() {
    getDocuments()
      .then((data) => setDocuments(data.documents))
      .catch((e) => setError(e.message))
  }

  function requireKey() {
    if (!apiKey.trim()) {
      setError('Enter your OpenRouter API key first.')
      return false
    }
    return true
  }

  async function handleIngest(event) {
    const files = [...event.target.files]
    event.target.value = '' // allow re-selecting the same file
    if (!files.length) return
    if (!requireKey()) return

    setBusy('ingest')
    setError('')
    setNotice('')
    try {
      const data = await ingestFiles(apiKey, files)
      const total = data.ingested.reduce((sum, d) => sum + d.chunks, 0)
      setNotice(`Ingested ${data.ingested.length} file(s) into ${total} chunk(s).`)
      setDocuments(data.documents)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }

  async function handleQuery(event) {
    event.preventDefault()
    if (!query.trim()) return
    if (!requireKey()) return

    setBusy('query')
    setError('')
    setAnswer(null)
    setSources([])
    try {
      const data = await runQuery(apiKey, { query, searchType })
      setAnswer(data.answer)
      setSources(data.sources)
      setAskedWith(searchType)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }

  async function handleClear() {
    if (!confirm('Delete all ingested documents?')) return
    setError('')
    try {
      await clearDocuments()
      setDocuments([])
      setNotice('All documents cleared.')
      setAnswer(null)
      setSources([])
    } catch (e) {
      setError(e.message)
    }
  }

  const totalChunks = documents.reduce((sum, d) => sum + d.chunks, 0)

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <span className="eyebrow">Retrieval-Augmented Generation</span>
          <h1>RAG&nbsp;Playground</h1>
          <p className="lede">
            Upload your documents, then ask questions answered strictly from their contents.
          </p>
        </div>

        <div className="keyfield">
          <label htmlFor="key">OpenRouter API key</label>
          <div className="key-input">
            <KeyIcon />
            <input
              id="key"
              type="password"
              placeholder="sk-or-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              autoComplete="off"
              spellCheck="false"
            />
            <span className={`dot ${apiKey ? 'on' : ''}`} title={apiKey ? 'Key set' : 'No key'} />
          </div>
          <span className="key-hint">Kept in this tab only · sent per request, never stored.</span>
        </div>
      </header>

      {status && <ImplPanel status={status} />}

      <main className="workspace">
        <section className="panel panel--docs">
          <div className="panel-head">
            <div>
              <h2>Library</h2>
              <span className="panel-meta">
                {documents.length === 0
                  ? 'Empty'
                  : `${documents.length} document${documents.length > 1 ? 's' : ''} · ${totalChunks} chunks`}
              </span>
            </div>
            {documents.length > 0 && (
              <button className="ghost-btn" onClick={handleClear} type="button">
                <TrashIcon /> Clear
              </button>
            )}
          </div>

          <label className={`dropzone ${busy === 'ingest' ? 'is-busy' : ''}`}>
            <input
              type="file"
              accept=".txt,.md,.pdf,text/plain,text/markdown,application/pdf"
              multiple
              onChange={handleIngest}
              disabled={busy === 'ingest'}
            />
            <UploadIcon />
            <span className="dz-title">{busy === 'ingest' ? 'Ingesting…' : 'Drop or choose files'}</span>
            <span className="dz-sub">.txt, .md, or .pdf</span>
          </label>

          {documents.length === 0 ? (
            <p className="empty">Nothing here yet — add a document to begin.</p>
          ) : (
            <ul className="doc-list">
              {documents.map((doc) => (
                <li key={doc.source}>
                  <DocIcon />
                  <div className="doc-body">
                    <span className="doc-title">{doc.title}</span>
                    <span className="doc-meta">
                      {doc.source} · {doc.chunks} chunks
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="panel panel--ask">
          <div className="panel-head">
            <h2>Ask a question</h2>
          </div>

          <form onSubmit={handleQuery}>
            <textarea
              className="prompt"
              placeholder="e.g. How long is job applicant data kept before deletion?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleQuery(e)
              }}
              rows={3}
            />

            <div className="controls">
              <div className="strategy">
                <span className="strategy-label">Retrieval</span>
                <div className="segment" role="group" aria-label="Retrieval strategy">
                  {searchTypes.map((type) => (
                    <button
                      key={type}
                      type="button"
                      className={searchType === type ? 'active' : ''}
                      onClick={() => setSearchType(type)}
                      title={STRATEGY_BLURB[type] || type}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <button className="primary" type="submit" disabled={busy === 'query'}>
                {busy === 'query' ? (
                  <>
                    <span className="spinner" /> Thinking…
                  </>
                ) : (
                  <>
                    Ask <ArrowIcon />
                  </>
                )}
              </button>
            </div>
            <span className="kbd-hint">⌘ / Ctrl + Enter to send</span>
          </form>

          {answer !== null && (
            <article className="answer">
              <div className="answer-head">
                <span className="answer-eyebrow">Answer</span>
                {askedWith && <span className="tag">{askedWith}</span>}
              </div>
              <p className="answer-body">{answer}</p>

              {sources.length > 0 && (
                <details className="sources">
                  <summary>
                    Show {sources.length} source passage{sources.length > 1 ? 's' : ''}
                  </summary>
                  <ul>
                    {sources.map((src, i) => (
                      <li key={i}>
                        <div className="src-meta">
                          <span className="src-title">{src.document_title || src.source}</span>
                          <span className="chip">{src.score.toFixed(3)}</span>
                        </div>
                        <div className="src-text">{src.text}</div>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </article>
          )}
        </section>
      </main>

      {error && (
        <div className="toast error" role="alert" onClick={() => setError('')}>
          {error}
        </div>
      )}
      {notice && !error && (
        <div className="toast notice" role="status" onClick={() => setNotice('')}>
          {notice}
        </div>
      )}
    </div>
  )
}

/* --- implementation progress dashboard --- */
function ImplPanel({ status }) {
  const { functions, implemented, total, complete } = status
  const pct = total ? Math.round((implemented / total) * 100) : 0
  const parts = [...new Set(functions.map((f) => f.part))]

  return (
    <section className={`panel impl ${complete ? 'is-complete' : 'is-incomplete'}`}>
      <div className="panel-head">
        <div>
          <h2>Implementation</h2>
          <span className="panel-meta">
            {implemented}/{total} of your notebook functions are live
          </span>
        </div>
        <span className={`impl-badge ${complete ? 'ok' : 'todo'}`}>
          {complete ? 'All implemented' : `${total - implemented} to go`}
        </span>
      </div>

      <div className="impl-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <span style={{ width: `${pct}%` }} />
      </div>

      <div className="impl-grid">
        {parts.map((part) => (
          <div className="impl-col" key={part}>
            <h3>{part}</h3>
            <ul>
              {functions
                .filter((f) => f.part === part)
                .map((f) => (
                  <li key={f.name} className={f.implemented ? 'done' : 'todo'}>
                    <div className="impl-row">
                      <span className={`impl-mark ${f.implemented ? 'done' : 'todo'}`}>
                        {f.implemented ? '✓' : ''}
                      </span>
                      <code>{f.name}</code>
                      {!f.implemented && <span className="impl-reason">{f.reason}</span>}
                    </div>
                    {f.description && <p className="impl-desc">{f.description}</p>}
                  </li>
                ))}
            </ul>
          </div>
        ))}
      </div>

      {!complete && (
        <p className="impl-hint">
          {total - implemented} function{total - implemented > 1 ? 's are' : ' is'} still missing,
          and this app ships <strong>no</strong> implementation of its own — asking a question will
          fail until every one is in place. Export them from the notebook
          (Parts&nbsp;1.4 / 2.7) into <code>solutions/</code>.
        </p>
      )}
    </section>
  )
}

/* --- inline icons (currentColor) --- */
function KeyIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="8" cy="15" r="4" />
      <path d="M10.8 12.2 20 3m-3 3 2 2m-4 0 2 2" strokeLinecap="round" />
    </svg>
  )
}
function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M12 16V4m0 0L7 9m5-5 5 5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5 18v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1" strokeLinecap="round" />
    </svg>
  )
}
function DocIcon() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M7 3h7l4 4v14a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" />
      <path d="M14 3v4h4M9 12h6M9 16h6" strokeLinecap="round" />
    </svg>
  )
}
function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-9 0 1 13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1l1-13" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M5 12h14m-6-6 6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
