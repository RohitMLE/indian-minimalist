import { useState, useCallback } from 'react'
import './App.css'

// ── Types ──────────────────────────────────────────
interface AnalysisResult {
  roomType: string
  currentStyle: string
  elementsDetected: string[]
  sdPrompt: string
  negativePrompt: string
  designNarrative: string
  keyCategories: string[]
}

interface Product {
  category: string
  name: string
  description: string
  price: string
  emoji: string
  thumbnail: string
  link: string
  source: string
  searchQuery: string
}

type Step = 'idle' | 'analysing' | 'transforming' | 'products' | 'done'

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

// ── Constants ──────────────────────────────────────
const STYLES = [
  { name: 'Japandi', emoji: '🌿', desc: 'Natural wood, zen calm' },
  { name: 'Indian Minimalist', emoji: '🏺', desc: 'Terracotta, brass, cane' },
  { name: 'Scandinavian', emoji: '❄️', desc: 'White, oak, hygge' },
  { name: 'Bohemian', emoji: '🌈', desc: 'Kilim, macrame, plants' },
  { name: 'Coastal', emoji: '🌊', desc: 'Airy blues, rattan, linen' },
  { name: 'Industrial', emoji: '⚙️', desc: 'Exposed brick, metal, walnut' },
  { name: 'Art Deco', emoji: '✨', desc: 'Velvet, gold, marble glamour' },
  { name: 'Wabi-Sabi', emoji: '🪨', desc: 'Aged, imperfect, earthy' },
]

const BUDGETS = [
  { label: 'Under ₹10k', sub: 'Budget-friendly picks' },
  { label: '₹25k', sub: 'Mid-range refresh' },
  { label: '₹50k', sub: 'Premium upgrade' },
  { label: '₹1 Lakh+', sub: 'Luxury transformation' },
]

// ── Image resize util ──────────────────────────────
function resizeImage(file: File, maxW = 768, maxH = 512): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    const url = URL.createObjectURL(file)
    img.onload = () => {
      const scale = Math.min(maxW / img.width, maxH / img.height, 1)
      const canvas = document.createElement('canvas')
      canvas.width = Math.round(img.width * scale)
      canvas.height = Math.round(img.height * scale)
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(url)
      resolve(canvas.toDataURL('image/jpeg', 0.88))
    }
    img.onerror = reject
    img.src = url
  })
}


// ── Main App ───────────────────────────────────────
export default function App() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string>('')
  const [imageBase64, setImageBase64] = useState<string>('')
  const [selectedStyle, setSelectedStyle] = useState<string>('')
  const [selectedBudget, setSelectedBudget] = useState<string>('')
  const [dragOver, setDragOver] = useState(false)
  const [step, setStep] = useState<Step>('idle')
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [transformedUrl, setTransformedUrl] = useState<string>('')
  const [products, setProducts] = useState<Product[]>([])
  const [activeTab, setActiveTab] = useState<'before' | 'after'>('before')
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string>('')
  // ── Chat / refine state ──
  const [chat, setChat] = useState<ChatMsg[]>([])
  const [chatInput, setChatInput] = useState('')
  const [refining, setRefining] = useState(false)

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    const b64 = await resizeImage(file)
    setImageBase64(b64)
  }, [])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const canTransform = imageBase64 && selectedStyle && selectedBudget && step === 'idle'

  const transform = async () => {
    setError('')
    setStep('analysing')
    setAnalysis(null)
    setTransformedUrl('')
    setProducts([])
    setActiveTab('before')

    try {
      // Step 1: Analyse the room
      const analyseRes = await fetch('http://localhost:8000/analyse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageBase64, style: selectedStyle, budget: selectedBudget }),
      })
      if (!analyseRes.ok) throw new Error('Analysis failed')
      const analysisData: AnalysisResult = await analyseRes.json()
      setAnalysis(analysisData)

      // Step 2: Find REAL available products first
      setStep('products')
      const productsRes = await fetch('http://localhost:8000/products', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          roomType: analysisData.roomType,
          style: selectedStyle,
          budget: selectedBudget,
          keyCategories: analysisData.keyCategories,
        }),
      })
      if (!productsRes.ok) throw new Error('Products fetch failed')
      const productsData: Product[] = await productsRes.json()
      setProducts(productsData)

      // Step 3: Generate the room image USING those real products
      setStep('transforming')
      const transformRes = await fetch('http://localhost:8000/transform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageBase64,
          sdPrompt: analysisData.sdPrompt,
          negativePrompt: analysisData.negativePrompt,
          style: selectedStyle,
          products: productsData.map(p => ({
            category: p.category,
            name: p.name,
            thumbnail: p.thumbnail,
          })),
        }),
      })
      if (!transformRes.ok) throw new Error('Image transformation failed')
      const transformData = await transformRes.json()
      setTransformedUrl(transformData.imageUrl)
      setActiveTab('after')
      setStep('done')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
      setStep('idle')
    }
  }

  const sendRefine = async (msg?: string) => {
    const text = (msg ?? chatInput).trim()
    if (!text || refining || !transformedUrl) return
    setChatInput('')
    setError('')
    const newChat: ChatMsg[] = [...chat, { role: 'user', content: text }]
    setChat(newChat)
    setRefining(true)
    setActiveTab('after')
    try {
      const res = await fetch('http://localhost:8000/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: transformedUrl,
          message: text,
          style: selectedStyle,
          products: products.map(p => ({ category: p.category, name: p.name, thumbnail: p.thumbnail })),
          history: chat,
        }),
      })
      if (!res.ok) throw new Error('Refine failed')
      const data = await res.json()
      setTransformedUrl(data.imageUrl)
      setChat([...newChat, { role: 'assistant', content: data.reply || 'Done — updated the design.' }])
    } catch (err) {
      setChat([...newChat, { role: 'assistant', content: '⚠ Could not apply that change. Try rephrasing.' }])
      setError(err instanceof Error ? err.message : 'Refine failed')
    } finally {
      setRefining(false)
    }
  }

  const QUICK_EDITS = ['Warmer lighting', 'Add more plants', 'Declutter the room', 'Cozier, softer textures']

  const copyMoodboard = () => {
    if (!products.length) return
    const lines = [
      `✦ @indian.minimalist Mood Board`,
      `Style: ${selectedStyle} | Budget: ${selectedBudget}`,
      `Room: ${analysis?.roomType || ''}`,
      '',
      ...products.map(p =>
        `${p.emoji} ${p.name} — ${p.price}\n   Buy: ${p.link}\n   Retailer: ${p.source}`
      ),
    ]
    navigator.clipboard.writeText(lines.join('\n'))
    setCopied(true)
    setTimeout(() => setCopied(false), 2500)
  }

  const stepState = (s: 'analyse' | 'products' | 'transform') => {
    if (step === 'idle' || step === 'done') {
      return step === 'done' ? 'done' : 'idle'
    }
    // New order: analyse → find products → generate image
    const order = ['analysing', 'products', 'transforming']
    const stepMap: Record<string, string> = { analyse: 'analysing', products: 'products', transform: 'transforming' }
    const idx = order.indexOf(step)
    const sIdx = order.indexOf(stepMap[s])
    if (idx > sIdx) return 'done'
    if (idx === sIdx) return 'active'
    return 'idle'
  }

  const isProcessing = step !== 'idle' && step !== 'done'
  const showResults = step === 'done'

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-handle"><span>@indian</span>.minimalist</div>
          <div className="header-tagline">Space Transformer · AI Interior Design</div>
        </div>
        <div className="header-dot" />
      </header>

      <main className="main">
        {/* ── Hero ── */}
        <div className="intro-hero">
          <h1>Transform your <em>space</em></h1>
          <p>Upload a room photo, choose your style, and let AI reimagine your interior — with shoppable Indian products.</p>
        </div>

        {/* ── Upload ── */}
        <div className="upload-section">
          <div className="section-label">01 — Upload your room</div>
          {!imagePreview ? (
            <div
              className={`upload-zone${dragOver ? ' drag-over' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <input type="file" accept="image/*" onChange={onInputChange} />
              <div className="upload-icon">📷</div>
              <div className="upload-title">Drop your room photo here</div>
              <div className="upload-sub">or click to browse · JPG, PNG, WEBP</div>
            </div>
          ) : (
            <div className="upload-preview">
              <img src={imagePreview} alt="Room preview" />
              <div className="upload-preview-overlay">
                <label>
                  <input type="file" accept="image/*" onChange={onInputChange} style={{ display: 'none' }} />
                  <span className="upload-change-btn">Change photo</span>
                </label>
              </div>
            </div>
          )}
        </div>

        {/* ── Style ── */}
        <div className="style-section">
          <div className="section-label">02 — Choose your style</div>
          <div className="style-grid">
            {STYLES.map(s => (
              <div
                key={s.name}
                className={`style-card${selectedStyle === s.name ? ' selected' : ''}`}
                onClick={() => setSelectedStyle(s.name)}
              >
                <div className="style-emoji">{s.emoji}</div>
                <div className="style-name">{s.name}</div>
                <div className="style-desc">{s.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Budget ── */}
        <div className="budget-section">
          <div className="section-label">03 — Set your budget</div>
          <div className="budget-grid">
            {BUDGETS.map(b => (
              <div
                key={b.label}
                className={`budget-card${selectedBudget === b.label ? ' selected' : ''}`}
                onClick={() => setSelectedBudget(b.label)}
              >
                <div className="budget-amount">{b.label}</div>
                <div className="budget-label">{b.sub}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── CTA ── */}
        {!isProcessing && !showResults && (
          <div className="cta-row">
            <button className="cta-btn" disabled={!canTransform} onClick={transform}>
              Transform this space
              <span className="cta-btn-arrow">→</span>
            </button>
          </div>
        )}

        {/* ── Error ── */}
        {error && (
          <div style={{ textAlign: 'center', color: 'var(--terracotta)', marginBottom: 32, fontSize: '0.9rem' }}>
            ⚠ {error}
            <button onClick={() => setStep('idle')} style={{ marginLeft: 12, textDecoration: 'underline', background: 'none', border: 'none', color: 'var(--terracotta)', cursor: 'pointer', fontSize: '0.9rem' }}>Try again</button>
          </div>
        )}

        {/* ── Progress ── */}
        {isProcessing && (
          <div className="progress-section">
            <h2 className="progress-title">Crafting your space…</h2>
            <div className="progress-steps">
              {[
                { key: 'analyse', label: 'Analysing room', icon: '🔍' },
                { key: 'products', label: 'Finding real products', icon: '🛍' },
                { key: 'transform', label: 'Generating design', icon: '✨' },
              ].map(s => {
                const state = stepState(s.key as 'analyse' | 'products' | 'transform')
                return (
                  <div key={s.key} className={`progress-step ${state}`}>
                    <div className="step-icon">
                      {state === 'done' ? '✓' : state === 'active' ? <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> : s.icon}
                    </div>
                    <div className="step-label">{s.label}</div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {showResults && (
          <div className="results-section">
            {/* Left: image panel */}
            <div>
              <div className="image-panel">
                <div className="tab-row">
                  <button className={`tab-btn${activeTab === 'before' ? ' active' : ''}`} onClick={() => setActiveTab('before')}>Before</button>
                  <button className={`tab-btn${activeTab === 'after' ? ' active' : ''}`} onClick={() => setActiveTab('after')}>After</button>
                </div>
                <div className="image-display">
                  {activeTab === 'before' ? (
                    imagePreview ? <img src={imagePreview} alt="Before" /> : <div className="image-placeholder">📷</div>
                  ) : (
                    transformedUrl
                      ? <img src={transformedUrl} alt="Transformed" />
                      : <div className="image-generating"><div className="spinner" /><p>Generating your space…</p></div>
                  )}
                </div>
                {analysis && (
                  <div className="narrative-band">
                    <div className="narrative-text">{analysis.designNarrative}</div>
                  </div>
                )}
              </div>
              {step === 'done' && (
                <div className="refine-panel">
                  <div className="refine-header">
                    <span className="refine-title">✦ Refine your design</span>
                    <span className="refine-sub">Ask to swap products, change lighting, declutter…</span>
                  </div>

                  {chat.length > 0 && (
                    <div className="chat-log">
                      {chat.map((m, i) => (
                        <div key={i} className={`chat-msg ${m.role}`}>{m.content}</div>
                      ))}
                      {refining && (
                        <div className="chat-msg assistant">
                          <span className="spinner" style={{ width: 12, height: 12, borderTopColor: 'var(--terracotta)' }} /> redrawing…
                        </div>
                      )}
                    </div>
                  )}

                  <div className="quick-edits">
                    {QUICK_EDITS.map(q => (
                      <button key={q} className="quick-chip" disabled={refining} onClick={() => sendRefine(q)}>{q}</button>
                    ))}
                    {products.slice(0, 4).map((p, i) => (
                      <button key={`swap-${i}`} className="quick-chip" disabled={refining} onClick={() => sendRefine(`Swap the ${p.category.toLowerCase()} for product ${i + 1} (${p.name})`)}>
                        ↻ {p.category}
                      </button>
                    ))}
                  </div>

                  <form className="chat-input-row" onSubmit={e => { e.preventDefault(); sendRefine() }}>
                    <input
                      className="chat-input"
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      placeholder="e.g. make the sofa green, add a floor lamp…"
                      disabled={refining}
                    />
                    <button className="chat-send" type="submit" disabled={refining || !chatInput.trim()}>
                      {refining ? '…' : 'Send'}
                    </button>
                  </form>

                  <div className="cta-row" style={{ marginBottom: 0, marginTop: 14 }}>
                    <button className="cta-btn" style={{ fontSize: '0.85rem', padding: '14px 32px' }} onClick={() => { setStep('idle'); setTransformedUrl(''); setProducts([]); setAnalysis(null); setChat([]); setChatInput('') }}>
                      ← Try another style
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Right: products */}
            <div className="products-panel">
              <div className="products-header">
                <h2 className="products-title">Shop the look</h2>
                {products.length > 0 && (
                  <button className={`copy-btn${copied ? ' copied' : ''}`} onClick={copyMoodboard}>
                    {copied ? '✓ Copied!' : '⎘ Copy mood board'}
                  </button>
                )}
              </div>
              {products.length === 0 && step === 'products' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'rgba(28,25,23,0.5)', fontSize: '0.85rem' }}>
                  <div className="spinner" style={{ borderColor: 'rgba(28,25,23,0.1)', borderTopColor: 'var(--terracotta)' }} />
                  Curating products for you…
                </div>
              )}
              <div className="product-cards">
                {products.map((p, i) => (
                  <div key={i} className="product-card">
                    <div className="product-thumb">
                      {p.thumbnail
                        ? <img src={p.thumbnail} alt={p.name} />
                        : <span>{p.emoji}</span>
                      }
                    </div>
                    <div className="product-info">
                      <div className="product-category">
                        {p.category}
                        {p.source && <span className="product-source">{p.source}</span>}
                      </div>
                      <div className="product-name">{p.name}</div>
                      <div className="product-desc">{p.description}</div>
                      <div className="product-footer">
                        {p.price && <span className="product-price">{p.price}</span>}
                        <a className="shop-btn buy" href={p.link} target="_blank" rel="noreferrer">Buy now ↗</a>
                        <a className="shop-btn search" href={`https://www.google.com/search?tbm=shop&q=${encodeURIComponent(p.searchQuery)}&gl=in`} target="_blank" rel="noreferrer">More options</a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
