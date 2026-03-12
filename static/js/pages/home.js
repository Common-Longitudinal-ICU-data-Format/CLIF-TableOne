import { api } from '../api.js';
import * as state from '../state.js';
import { navigate } from '../router.js';

const GUIDE_STEPS = [
  {
    title: 'Review Validation Status',
    description: 'See all CLIF tables at a glance — each card shows validation status, DQA scores, and when it was last reviewed.',
    video: '/images/guide/step-1-overview.mp4',
    image: '/images/guide/step-1-overview.gif',
  },
  {
    title: 'Drill Into Errors',
    description: 'Click any table to see detailed results. The "Review & Resolve" section lists each error with its category, check, and message.',
    video: '/images/guide/step-2-review.mp4',
    image: '/images/guide/step-2-review.gif',
  },
  {
    title: 'Provide Feedback & Save',
    description: 'Set each error to Accepted, Rejected, or Pending. Scores update live, and reports regenerate when you save.',
    video: '/images/guide/step-3-feedback.mp4',
    image: '/images/guide/step-3-feedback.gif',
  },
];

function renderGuideSection(steps) {
  const stepsHtml = steps.map((step, i) => {
    const mediaHtml = step.video
      ? `<video autoplay loop muted playsinline
               onerror="this.outerHTML='<img src=&quot;${step.image}&quot; alt=&quot;${step.title}&quot; onerror=&quot;this.parentElement.innerHTML=\\'<div class=guide-step-placeholder>${i + 1}</div>\\'&quot;>'"
         ><source src="${step.video}" type="video/mp4"></video>`
      : `<img src="${step.image}" alt="${step.title}" loading="lazy"
              onerror="this.parentElement.innerHTML='<div class=\\'guide-step-placeholder\\'>${i + 1}</div>'">`;
    return `
      <div class="guide-step">
        <div class="guide-step-image">
          ${mediaHtml}
        </div>
        <div class="guide-step-text">
          <span class="guide-step-number">${i + 1}</span>
          <h4 class="guide-step-title">${step.title}</h4>
          <p class="guide-step-desc">${step.description}</p>
        </div>
      </div>
    `;
  }).join('');

  return `
    <div class="guide-section">
      <h3>Validation Made Easy</h3>
      <div class="guide-steps">
        ${stepsHtml}
      </div>
    </div>
  `;
}

export async function renderHome(el) {
  el.innerHTML = `
    <!-- Hero Nav Cards -->
    <div class="home-hero">
      <div class="hero-card" id="hero-tableone">
        <div class="hero-card-body">
          <h3>Table One Results</h3>
          <p>Explore cohorts, demographics, medications, ventilation, outcomes</p>
        </div>
        <span class="hero-arrow" aria-hidden="true">&rarr;</span>
      </div>
      <div class="hero-card" id="hero-validation">
        <div class="hero-card-body">
          <h3>Validation</h3>
          <p>Review DQA validation status for all CLIF tables</p>
        </div>
        <span class="hero-arrow" aria-hidden="true">&rarr;</span>
      </div>
    </div>

    <!-- AI All-Tables Interpretation -->
    <div id="ai-all-section" style="display:none;margin-bottom:28px;">
      <div class="card ai-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
          <div class="loading-spinner" id="ai-all-spinner"></div>
          <span style="font-size:0.9rem;font-weight:600;">Validation Summary</span>
          <span id="ai-model-badge" style="font-size:0.68rem;color:var(--text-muted);display:none;"></span>
          <button class="btn btn-sm btn-secondary" id="btn-ai-expand" style="margin-left:auto;display:none;">Expand</button>
          <a href="#" id="btn-ai-reinterpret" style="margin-left:8px;font-size:0.8rem;display:none;">Re-interpret</a>
        </div>
        <div id="ai-all-text"></div>
      </div>
    </div>

    <div id="download-section" style="display:none;">
      <h3>Download Reports</h3>
      <a class="btn btn-secondary" id="dl-pdf" style="text-decoration:none;display:inline-block;">Download PDF Report</a>
      <a class="btn btn-secondary" id="dl-csv" style="text-decoration:none;display:inline-block;margin-left:8px;">Download CSV Summary</a>
    </div>

    ${renderGuideSection(GUIDE_STEPS)}

    <hr>

    <!-- FAQ -->
    <div class="faq-section">
      <h3>How to Use This App</h3>
      <details>
        <summary>What does this app do?</summary>
        <p>This app validates your CLIF 2.1 data against the official spec. It flags issues across conformance, completeness, and plausibility — but some "errors" might be totally fine for your site. Your data might not use a certain drug or device, and that's okay.</p>
      </details>
      <details>
        <summary>1. Review Each Table</summary>
        <p>Click through the tables on the Validation page. Check out the validation results and summary stats. See what's flagged.</p>
      </details>
      <details>
        <summary>2. Review & Provide Feedback</summary>
        <p>In the Validation tab, enable "Review Status-Affecting Errors". For each error, decide:</p>
        <ul>
          <li><strong>Accept</strong> — Legit issue, needs fixing</li>
          <li><strong>Reject</strong> — Not a problem for your site (add a reason why)</li>
          <li><strong>Pending</strong> — You'll deal with it later</li>
        </ul>
        <p>Hit "Save Feedback" when done — the table PDF and combined report are automatically regenerated with your feedback applied.</p>
      </details>
      <details>
        <summary>3. Fixed Something? Re-run Validation</summary>
        <p>After fixing an issue in your data, go to the Validation page and use "Run Validation" at the bottom to re-validate affected tables. Reports are automatically updated.</p>
      </details>
      <details>
        <summary>4. Check Out Table One</summary>
        <p>Click "Table One Results" above to explore cohorts, demographics, meds, ventilation, and outcomes — all in one place.</p>
      </details>
      <details>
        <summary>Questions?</summary>
        <p>Hit me up on CLIF Slack — Kaveri Chhikara.</p>
      </details>
    </div>
  `;

  // Hero card navigation
  document.getElementById('hero-tableone').addEventListener('click', () => navigate('tableone'));
  document.getElementById('hero-validation').addEventListener('click', () => navigate('validation'));

  // AI section — only show if LLM available
  initAiAll();

  // Check downloads
  checkDownloads();

}

async function initAiAll() {
  const section = document.getElementById('ai-all-section');
  if (!section) return;

  let modelName = '';
  try {
    const { available, model } = await api.getLlmStatus();
    if (!available) return;
    modelName = model || '';
  } catch (e) { return; }

  section.style.display = 'block';

  // Show model badge
  const badge = document.getElementById('ai-model-badge');
  if (badge && modelName) {
    badge.textContent = `via ${modelName} (Ollama)`;
    badge.style.display = 'inline';
  }

  // Restore cached interpretation or fetch fresh
  const cached = state.get('aiInterpretation');
  if (cached) {
    restoreAiResult(cached);
  } else {
    startAiStream();
  }

  // Re-interpret link — always fetches fresh
  document.getElementById('btn-ai-reinterpret').addEventListener('click', (e) => {
    e.preventDefault();
    startAiStream();
  });

  // Expand button — fullscreen overlay
  document.getElementById('btn-ai-expand').addEventListener('click', () => {
    const html = document.getElementById('ai-all-text').innerHTML;
    const overlay = document.createElement('div');
    overlay.className = 'ai-fullpage-overlay';
    overlay.innerHTML = `
      <button class="btn btn-secondary close-btn" id="ai-overlay-close">Close</button>
      <h2>Validation Summary — All Tables</h2>
      <div class="card ai-card"><div class="ai-prose">${html}</div></div>
    `;
    document.body.appendChild(overlay);
    overlay.querySelector('#ai-overlay-close').addEventListener('click', () => overlay.remove());
  });
}

function restoreAiResult(raw) {
  const textEl = document.getElementById('ai-all-text');
  const spinner = document.getElementById('ai-all-spinner');
  const expandBtn = document.getElementById('btn-ai-expand');
  const reinterpretBtn = document.getElementById('btn-ai-reinterpret');

  textEl.innerHTML = typeof marked !== 'undefined' ? marked.parse(raw) : raw;
  spinner.style.display = 'none';
  expandBtn.style.display = 'inline-flex';
  reinterpretBtn.style.display = 'inline';
}

function startAiStream() {
  const textEl = document.getElementById('ai-all-text');
  const spinner = document.getElementById('ai-all-spinner');
  const expandBtn = document.getElementById('btn-ai-expand');
  const reinterpretBtn = document.getElementById('btn-ai-reinterpret');

  textEl.innerHTML = '';
  spinner.style.display = '';
  expandBtn.style.display = 'none';
  reinterpretBtn.style.display = 'none';

  let raw = '';
  api.streamInterpretationAll({
    onChunk(text) {
      raw += text;
      textEl.innerHTML = typeof marked !== 'undefined' ? marked.parse(raw) : raw;
      textEl.scrollTop = textEl.scrollHeight;
    },
    onDone() {
      spinner.style.display = 'none';
      expandBtn.style.display = 'inline-flex';
      reinterpretBtn.style.display = 'inline';
      state.set('aiInterpretation', raw);
    },
    onError(msg) {
      spinner.style.display = 'none';
      if (msg && (msg.includes('404') || msg.toLowerCase().includes('no tables'))) {
        textEl.innerHTML = '<em>No tables analyzed yet. Run analysis first to generate a validation summary.</em>';
      } else {
        textEl.innerHTML = '<em>Error: ' + escapeHtml(msg) + '</em>';
      }
      reinterpretBtn.style.display = 'inline';
    },
  });
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function checkDownloads() {
  const section = document.getElementById('download-section');
  if (!section) return;
  fetch('/api/reports/download/pdf', { method: 'HEAD' }).then(r => {
    if (r.ok) {
      section.style.display = 'block';
      document.getElementById('dl-pdf').href = '/api/reports/download/pdf';
      document.getElementById('dl-csv').href = '/api/reports/download/csv';
    }
  }).catch(() => {});
}
