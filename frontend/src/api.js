// Thin wrapper around the FastAPI backend. The OpenRouter key is passed per
// request in the X-OpenRouter-Key header and never persisted server-side.

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function authHeaders(apiKey) {
  return apiKey ? { 'X-OpenRouter-Key': apiKey } : {}
}

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail || detail
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail)
  }
  return res.json()
}

export async function getSearchTypes() {
  return handle(await fetch(`${API_BASE}/search-types`))
}

export async function getStatus() {
  return handle(await fetch(`${API_BASE}/status`))
}

export async function getDocuments() {
  return handle(await fetch(`${API_BASE}/documents`))
}

export async function clearDocuments() {
  return handle(await fetch(`${API_BASE}/documents`, { method: 'DELETE' }))
}

export async function ingestFiles(apiKey, files) {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  return handle(
    await fetch(`${API_BASE}/ingest`, {
      method: 'POST',
      headers: authHeaders(apiKey),
      body: form,
    }),
  )
}

export async function runQuery(apiKey, { query, searchType }) {
  return handle(
    await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey) },
      body: JSON.stringify({ query, search_type: searchType || null }),
    }),
  )
}
