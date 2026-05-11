import { api } from '../api.js';
import { navigate } from '../router.js';
import { renderTabs } from '../components/tabs.js';

// Persist the user's cohort choice across navigation. Defaults to 'ci' so
// existing users land on the same view they had before the redesign.
const COHORT_STORAGE_KEY = 'clif_tableone_cohort';

function getCohort() {
  try {
    const v = localStorage.getItem(COHORT_STORAGE_KEY);
    if (v === 'ci' || v === 'ward') return v;
  } catch {}
  return 'ci';
}

function setCohort(cohort) {
  try { localStorage.setItem(COHORT_STORAGE_KEY, cohort); } catch {}
}

export async function renderTableone(el) {
  const cohort = getCohort();

  // Availability check for the chosen cohort. If results aren't generated,
  // fall back to a friendly message rather than rendering empty tabs.
  let available = false;
  try {
    const r = await api.getTableoneAvailable(cohort);
    available = r.available;
  } catch (e) {
    // fall through to "unavailable" message below
  }

  if (!available) {
    el.innerHTML = `
      <button class="btn btn-back" id="btn-back-home">&larr; Back to Home</button>
      <h1>Table One Results</h1>
      ${renderCohortToggle(cohort)}
      <p>Results not yet generated for the <strong>${cohort === 'ci' ? 'critical-illness' : 'ward'}</strong> cohort.
         Run: <code>uv run python run_project.py --no-summary --get-ecdf --ward</code></p>`;
    document.getElementById('btn-back-home').addEventListener('click', () => navigate('home'));
    wireCohortToggle(el);
    return;
  }

  el.innerHTML = `
    <button class="btn btn-back" id="btn-back-home">&larr; Back to Home</button>
    <h1>Table One Results</h1>
    ${renderCohortToggle(cohort)}
    <p style="opacity:0.6;">To regenerate, run: <code>uv run python run_project.py --no-summary --get-ecdf --ward</code></p>
    <div id="t1-tabs"></div>
    <div id="t1-content"></div>
  `;

  document.getElementById('btn-back-home').addEventListener('click', () => navigate('home'));
  wireCohortToggle(el);

  // Strata are CI-only; hide the tab in Ward mode rather than show "no data".
  const tabs = [
    { id: 'overview',     label: 'Overview' },
    { id: 'cohort',       label: 'Cohort flow' },
    { id: 'demographics', label: 'Demographics' },
    { id: 'medications',  label: 'Medications' },
    { id: 'imv',           label: 'Ventilation & PF/SF' },
    { id: 'sofa_cci',      label: 'SOFA & CCI' },
    { id: 'comorbidities', label: 'Comorbidities' },
    { id: 'outcomes',      label: 'Outcomes' },
    ...(cohort === 'ci' ? [{ id: 'strata', label: 'Strata' }] : []),
  ];

  renderTabs(
    document.getElementById('t1-tabs'),
    document.getElementById('t1-content'),
    tabs,
    async (tabId, panel) => {
      try {
        if (tabId === 'overview') {
          const data = await api.getTableoneStrobe(cohort);
          renderOverview(data, panel, cohort);
          return;
        }
        if (tabId === 'strata') {
          await renderStrataTab(panel, cohort);
          return;
        }
        const data = await api.getTableoneTabFor(cohort, tabId);
        renderTabContent(tabId, data, panel, cohort);
      } catch (e) {
        panel.innerHTML = `<p>No data for this tab in the <strong>${cohort}</strong> cohort.</p>`;
      }
    }
  );
}


// ── Cohort toggle ────────────────────────────────────────────────────

function renderCohortToggle(cohort) {
  return `
    <div class="cohort-toggle" role="radiogroup" aria-label="Cohort">
      <button class="cohort-btn ${cohort === 'ci' ? 'active' : ''}" data-cohort="ci" role="radio" aria-checked="${cohort === 'ci'}">
        Critical illness
      </button>
      <button class="cohort-btn ${cohort === 'ward' ? 'active' : ''}" data-cohort="ward" role="radio" aria-checked="${cohort === 'ward'}">
        Ward
      </button>
    </div>`;
}

function wireCohortToggle(el) {
  el.querySelectorAll('.cohort-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const next = btn.dataset.cohort;
      if (next === getCohort()) return;
      setCohort(next);
      // Cleanest reload: re-render the whole page. Tabs are lazy so this is cheap.
      renderTableone(el);
    });
  });
}


// ── Overview tab ─────────────────────────────────────────────────────

function renderOverview(data, panel, cohort) {
  const { strobe = {}, mortality = {}, kpi = {} } = data || {};

  // Curated KPI cards. Order matters: most-eyeballed metrics first.
  // Some entries only appear for one cohort — we render whichever exist.
  const cards = [];
  const push = (label, value) => {
    if (value !== null && value !== undefined && value !== '') cards.push({ label, value });
  };

  if (cohort === 'ci') {
    push('Critically ill encounters', kpi.n_encounters || fmt(strobe['5_all_critically_ill']));
    push('Hospital mortality',         kpi.hospital_mortality);
    push('Unique patients',            kpi.n_patients);
    push('Median age (Q1, Q3)',        kpi.age_median_iqr);
    push('ICU encounters',             fmt(strobe['1_icu_encounters']));
    push('Advanced respiratory',       fmt(strobe['2_advanced_resp_support_hospitalizations']));
    push('NIPPV / HFNC',               fmt(strobe['2b_nippv_hfnc_hospitalizations']));
    push('Vasoactive support',         fmt(strobe['3_vasoactive_hospitalizations']));
    push('Other critically ill',       fmt(strobe['4_other_critically_ill']));
    push('Sepsis events',              kpi.sepsis_events_total);
    push('Sepsis encounters',          kpi.sepsis_encounters);
    push('ICU episodes (total)',       kpi.icu_episodes_total);
  } else {
    push('Ward-touching encounters',   kpi.n_encounters || fmt(strobe['0_total_hospitalizations']));
    push('Hospital mortality',         kpi.hospital_mortality);
    push('Unique patients',            kpi.n_patients);
    push('Median age (Q1, Q3)',        kpi.age_median_iqr);
    push('All critically ill (in ward)', fmt(strobe['5_all_critically_ill']));
    push('Ward-only (no critical care)', fmt(strobe['6_ward_no_critical_care']));
    push('ICU encounters',             fmt(strobe['1_icu_encounters']));
    push('Advanced respiratory',       fmt(strobe['2_advanced_resp_support_hospitalizations']));
    push('NIPPV / HFNC',               fmt(strobe['2b_nippv_hfnc_hospitalizations']));
    push('Vasoactive support',         fmt(strobe['3_vasoactive_hospitalizations']));
  }

  let html = `<div class="kpi-grid">`;
  for (const c of cards) {
    html += `
      <div class="card metric-card">
        <div class="metric-label">${escapeHtml(c.label)}</div>
        <div class="metric-value">${escapeHtml(String(c.value))}</div>
      </div>`;
  }
  html += `</div>`;

  // Cohort flow diagram — boxes-and-arrows in SVG, mortality embedded
  // in each terminal box. Replaces the matplotlib CONSORT PNG and the
  // earlier mortality bar chart in one move.
  html += `<div class="card" style="margin-top:16px;">
    <h3 style="margin-top:0;">Cohort flow</h3>
    <div id="t1-cohort-flow"></div>
  </div>`;

  panel.innerHTML = html;

  const flowEl = panel.querySelector('#t1-cohort-flow');
  if (flowEl) flowEl.innerHTML = renderCohortFlowSVG(strobe, mortality, cohort);
}


// ── Existing tab content (preserved, threaded through cohort) ────────

function imgTag(filename, cohort) {
  return `<img src="/api/tableone/${cohort}/images/${filename}" style="width:100%;border-radius:8px;margin:8px 0;" loading="lazy">`;
}

function renderTabContent(tabId, data, panel, cohort) {
  let html = '';

  if (tabId === 'cohort') {
    // CONSORT PNG dropped — its successor (interactive SVG flow with
    // mortality embedded in each box) lives on the Overview tab.
    if (data.upset)   html += `<h3>Cohort Intersections (UpSet Plot)</h3>${imgTag(data.upset, cohort)}`;
    if (data.code_status) html += `<h3>Code Status Distribution</h3>${imgTag(data.code_status, cohort)}`;
    if (data.sankeys && data.sankeys.length > 0) {
      html += `<h3>Patient Flow Sankey Diagrams</h3><div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">`;
      for (const s of data.sankeys) {
        html += `<div><p style="font-weight:600;">${escapeHtml(s.label)}</p>${imgTag(s.filename, cohort)}</div>`;
      }
      html += `</div>`;
    }
  }

  else if (tabId === 'demographics') {
    if (data.tables) {
      for (const [key] of Object.entries(data.tables)) {
        html += `<h3>Table One — ${key === 'by_year' ? 'Stratified by Year' : 'Overall'}</h3>`;
        html += `<div id="demo-table-${key}"></div>`;
      }
    }
  }

  else if (tabId === 'medications') {
    if (data.html_plots) {
      for (const plot of data.html_plots) {
        html += `<h4>${escapeHtml(plot.label)}</h4><iframe src="/api/tableone/${cohort}/images/${plot.filename}" style="width:100%;height:520px;border:none;border-radius:8px;margin-bottom:16px;"></iframe>`;
      }
    }
    if (data.csv_files) {
      for (const f of data.csv_files) {
        html += `<h4>${escapeHtml(f.label)}</h4><div id="meds-table-${slug(f.label)}"></div>`;
      }
    }
  }

  else if (tabId === 'imv') {
    if (data.images) {
      for (const img of data.images) html += `<h3>${escapeHtml(img.label)}</h3>${imgTag(img.filename, cohort)}`;
    }
    if (data.csv_files) {
      for (const f of data.csv_files) html += `<h4>${escapeHtml(f.label)}</h4><div id="imv-table-${slug(f.label)}"></div>`;
    }
  }

  else if (tabId === 'sofa_cci') {
    if (data.images) {
      for (const img of data.images) html += `<h3>${escapeHtml(img.label)}</h3>${imgTag(img.filename, cohort)}`;
    }
    if (data.csv_files) {
      for (const f of data.csv_files) html += `<h4>${escapeHtml(f.label)}</h4><div id="sofa-table-${slug(f.label)}"></div>`;
    }
  }

  else if (tabId === 'comorbidities') {
    if (data.images) {
      for (const img of data.images) html += `<h3>${escapeHtml(img.label)}</h3>${imgTag(img.filename, cohort)}`;
    }
    if (data.csv_files) {
      for (const f of data.csv_files) html += `<h4>${escapeHtml(f.label)}</h4><div id="comorb-table-${slug(f.label)}"></div>`;
    }
  }

  else if (tabId === 'outcomes') {
    if (data.images) {
      for (const img of data.images) html += `<h3>${escapeHtml(img.label)}</h3>${imgTag(img.filename, cohort)}`;
    }
    if (data.csv_files) {
      for (const f of data.csv_files) html += `<h4>${escapeHtml(f.label)}</h4><div id="outcomes-table-${slug(f.label)}"></div>`;
    }
    const noData = (!data.images || data.images.length === 0) && (!data.csv_files || data.csv_files.length === 0);
    if (noData) html += '<p>No outcome data available for this cohort.</p>';
  }

  if (!html) html = '<p>No data available for this tab.</p>';
  panel.innerHTML = html;

  // Initialize DataTables for any table data.
  // Default sort is disabled (order: []) so the Variable column keeps its
  // hierarchical CSV order — section headers (e.g. "Encounter Types")
  // followed by indented sub-rows. Users can still click any column header
  // to sort interactively. paging:false on the demographics/strata-style
  // tables so reviewers see the whole Table One at a glance instead of
  // 25-row pages — the largest of these is ~140 rows, well within the
  // browser's comfort zone.
  if (typeof jQuery !== 'undefined' && jQuery.fn.DataTable) {
    if (data.tables) {
      for (const [key, tbl] of Object.entries(data.tables)) {
        const container = panel.querySelector(`#demo-table-${key}`);
        if (!container || !tbl.data || tbl.data.length === 0) continue;
        const tableEl = document.createElement('table');
        tableEl.className = 'display';
        tableEl.style.width = '100%';
        container.appendChild(tableEl);
        jQuery(tableEl).DataTable({
          data: tbl.data,
          columns: tbl.columns.map(c => ({ data: c, title: c })),
          paging: false,
          scrollX: true,
          order: [],
        });
      }
    }
    if (data.csv_files) {
      for (const f of data.csv_files) {
        if (!f.data || f.data.length === 0) continue;
        const container = panel.querySelector(`[id$="${slug(f.label)}"]`);
        if (!container) continue;
        const tableEl = document.createElement('table');
        tableEl.className = 'display';
        tableEl.style.width = '100%';
        container.appendChild(tableEl);
        jQuery(tableEl).DataTable({
          data: f.data,
          columns: f.columns.map(c => ({ data: c, title: c })),
          pageLength: 25,
          scrollX: true,
          order: [],
        });
      }
    }
  }
}


// ── Strata tab ───────────────────────────────────────────────────────
// Stratum dropdown picker; lazy per-stratum fetch. Each stratum renders
// up to three Table One tables: by-year, ICU vs non-ICU, and (vaso only)
// ED→ICU vs ED→Ward. CI cohort only.

async function renderStrataTab(panel, cohort) {
  panel.innerHTML = '<div class="loading-spinner"></div>';

  let resp;
  try {
    resp = await api.getTableoneStrataList(cohort);
  } catch (e) {
    panel.innerHTML = `<p>Could not load strata list.</p>`;
    return;
  }
  const strata = resp.strata || [];
  if (strata.length === 0) {
    panel.innerHTML = `<p>No strata available for the <strong>${cohort}</strong> cohort.</p>`;
    return;
  }

  // Restore last-picked stratum from localStorage (per cohort).
  const storeKey = `clif_tableone_stratum_${cohort}`;
  let initial = strata[0].key;
  try {
    const saved = localStorage.getItem(storeKey);
    if (saved && strata.some(s => s.key === saved)) initial = saved;
  } catch {}

  panel.innerHTML = `
    <div class="strata-controls">
      <label for="strata-picker" class="strata-label">Stratum:</label>
      <select id="strata-picker" class="strata-picker">
        ${strata.map(s => `<option value="${s.key}" ${s.key === initial ? 'selected' : ''}>${escapeHtml(s.label)}</option>`).join('')}
      </select>
      <span class="strata-hint">Subsets of the critical-illness cohort. See OUTPUT_REFERENCE.md for definitions.</span>
    </div>
    <div id="strata-content"></div>`;

  const sel = panel.querySelector('#strata-picker');
  const content = panel.querySelector('#strata-content');

  async function loadStratum(stratum) {
    try { localStorage.setItem(storeKey, stratum); } catch {}
    content.innerHTML = '<div class="loading-spinner"></div>';
    let data;
    try {
      data = await api.getTableoneStratum(cohort, stratum);
    } catch (e) {
      content.innerHTML = `<p>Could not load stratum: ${escapeHtml(stratum)}.</p>`;
      return;
    }
    renderStratumPayload(data, content);
  }

  sel.addEventListener('change', () => loadStratum(sel.value));
  loadStratum(initial);
}

function renderStratumPayload(data, content) {
  const tables = data && data.tables ? data.tables : {};
  const ids = Object.keys(tables);
  if (ids.length === 0) {
    content.innerHTML = `<p>No tables available for stratum <strong>${escapeHtml(data.stratum || '')}</strong>.</p>`;
    return;
  }

  // Render section heads + DataTables containers.
  let html = '';
  for (const id of ids) {
    const tbl = tables[id];
    html += `
      <div class="strata-section">
        <h3>${escapeHtml(tbl.label || id)}</h3>
        <p class="strata-source">Source: <code>${escapeHtml(tbl.filename || '')}</code> · ${tbl.data ? tbl.data.length : 0} rows</p>
        <div id="strata-table-${escapeHtml(id)}"></div>
      </div>`;
  }
  content.innerHTML = html;

  // Initialize DataTables once the DOM is set up.
  if (typeof jQuery !== 'undefined' && jQuery.fn.DataTable) {
    for (const id of ids) {
      const tbl = tables[id];
      if (!tbl.data || tbl.data.length === 0) continue;
      const container = content.querySelector(`#strata-table-${id}`);
      if (!container) continue;
      const tableEl = document.createElement('table');
      tableEl.className = 'display';
      tableEl.style.width = '100%';
      container.appendChild(tableEl);
      jQuery(tableEl).DataTable({
        data: tbl.data,
        columns: tbl.columns.map(c => ({ data: c, title: c })),
        paging: false,
        scrollX: true,
        // The Variable column is hierarchical (indented section headers).
        // Disable initial sort so the order matches the CSV; users can
        // still click a header to sort interactively.
        order: [],
      });
    }
  }
}


// ── SVG cohort flow diagram ──────────────────────────────────────────
// Pure-SVG, no library. Renders boxes-and-arrows from strobe + mortality.
// Mortality % is embedded in each terminal box. Replaces the matplotlib
// CONSORT PNG. CSS variables resolve via inline `style=` (not attributes).

function renderCohortFlowSVG(strobe, mortality, cohort) {
  const total       = strobe['0_total_hospitalizations'];
  const stitched    = strobe['1b_after_stitching'];
  const stitchedJoin = strobe['1b_stitched_hosp_ids'];

  const branches = [
    { label: 'ICU',                  n: strobe['1_icu_encounters'],                       mort: mortality['ICU Hospitalizations'],          sepsisPct: strobe['sepsis_icu_pct'] },
    { label: 'Advanced respiratory', n: strobe['2_advanced_resp_support_hospitalizations'], mort: mortality['Advanced Respiratory Support'],  sepsisPct: strobe['sepsis_advanced_resp_pct'] },
    { label: 'Vasoactive support',   n: strobe['3_vasoactive_hospitalizations'],          mort: mortality['Vasoactive Hospitalizations'],   sepsisPct: strobe['sepsis_vaso_pct'] },
    { label: 'Other critically ill', n: strobe['4_other_critically_ill'],                 mort: mortality['Other Critically Ill'],           sepsisPct: strobe['sepsis_other_ci_pct'] },
  ];
  if (cohort === 'ward') {
    branches.push({
      label: 'Ward only (no critical care)',
      n: strobe['6_ward_no_critical_care'],
      mort: null, // ward-only survivors → mortality not meaningful here
    });
  }

  // Sepsis incidence for the final box (if ASE was computed)
  const sepsisN   = strobe['sepsis_encounters'];
  const sepsisPct = strobe['sepsis_incidence_pct'];

  const bottom = cohort === 'ci'
    ? { label: 'All critically ill adults', n: strobe['5_all_critically_ill'], mort: mortality['All Critically Ill Adults'], sepsisN, sepsisPct }
    : { label: 'Ward cohort (all)',         n: strobe['0_total_hospitalizations'], mort: mortality['Ward Cohort'], sepsisN, sepsisPct };

  // Layout — viewBox so the SVG scales fluidly to its container.
  const W = 1100;
  const padX = 16;

  const branchN = branches.length;
  // Each branch box is the same width; gap = 12.
  const gap = 12;
  const branchBoxW = Math.floor((W - 2 * padX - gap * (branchN - 1)) / branchN);
  const branchBoxH = 110;  // taller to fit mortality + sepsis lines

  const topBoxW = 360;
  const topBoxH = 64;

  const yTop1   = 16;
  const yTop2   = yTop1 + topBoxH + 56;
  const yBranch = yTop2 + topBoxH + 70;
  const yBottom = yBranch + branchBoxH + 70;
  const H       = yBottom + branchBoxH + 48;  // extra room for sepsis line in bottom box

  const cx = W / 2;

  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;">
    <defs>
      <marker id="t1-arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--text-secondary);"/>
      </marker>
    </defs>`;

  // Two top boxes
  svg += flowBox(cx - topBoxW/2, yTop1,  topBoxW, topBoxH, 'All hospitalizations',                 total,    null, false);
  svg += flowBox(cx - topBoxW/2, yTop2,  topBoxW, topBoxH, `Adults after stitching${stitchedJoin != null ? ' (' + Number(stitchedJoin).toLocaleString() + ' linked IDs merged)' : ''}`, stitched, null, false);
  svg += flowArrow(cx, yTop1 + topBoxH, cx, yTop2);

  // Branch boxes
  const branchTotalW = branchBoxW * branchN + gap * (branchN - 1);
  const branchStartX = (W - branchTotalW) / 2;
  branches.forEach((b, i) => {
    const x = branchStartX + i * (branchBoxW + gap);
    svg += flowBox(x, yBranch, branchBoxW, branchBoxH, b.label, b.n, b.mort, false, null, b.sepsisPct);
    svg += flowArrow(cx, yTop2 + topBoxH, x + branchBoxW / 2, yBranch);
  });

  // Bottom summary box (accented) — includes sepsis incidence if available
  const bottomBoxH = (bottom.sepsisN && Number(bottom.sepsisN) > 0) ? branchBoxH + 20 : branchBoxH;
  svg += flowBox(cx - topBoxW/2, yBottom, topBoxW, bottomBoxH, bottom.label, bottom.n, bottom.mort, true, bottom.sepsisN, bottom.sepsisPct);

  // Arrows from each branch down to bottom.
  // In CI mode, skip the ward-only box (it doesn't exist).
  // In ward mode, ALL branches (including ward-only) flow to the final box.
  branches.forEach((b, i) => {
    if (cohort === 'ci' && b.label.startsWith('Ward only')) return;
    const x = branchStartX + i * (branchBoxW + gap);
    svg += flowArrow(x + branchBoxW / 2, yBranch + branchBoxH, cx, yBottom);
  });

  svg += `</svg>`;
  return svg;
}

function flowBox(x, y, w, h, label, n, mort, accent, sepsisN, sepsisPct) {
  const fmtN = (v) => (v == null ? '—' : Number(v).toLocaleString());
  const stroke = accent ? 'var(--primary)' : 'var(--border-strong)';
  const strokeW = accent ? 2 : 1;
  const fill = 'var(--bg-secondary)';

  const hasMort   = mort != null && Number.isFinite(Number(mort));
  // For branch boxes: sepsisPct is passed directly (no sepsisN).
  // For bottom box: both sepsisN and sepsisPct are passed.
  const hasSepsis = (sepsisN != null && Number(sepsisN) > 0) || (sepsisPct != null && Number(sepsisPct) > 0);
  const nLines    = 1 + (hasMort ? 1 : 0) + (hasSepsis ? 1 : 0);

  // Vertical centering depends on how many detail lines are present.
  const labelY = nLines >= 3 ? y + 18 : (hasMort ? y + 24 : y + h/2 - 6);
  const nY     = nLines >= 3 ? y + 44 : (hasMort ? y + 54 : y + h/2 + 18);
  const mortY  = nLines >= 3 ? y + 66 : y + 80;
  const sepsisY = nLines >= 3 ? y + 84 : y + 80;

  return `
    <g>
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" ry="6"
            style="fill:${fill};stroke:${stroke};stroke-width:${strokeW};"/>
      <text x="${x + w/2}" y="${labelY}" text-anchor="middle"
            style="fill:var(--text-secondary);font-size:13px;font-weight:600;font-family:var(--font-family);">${escapeXML(label)}</text>
      <text x="${x + w/2}" y="${nY}" text-anchor="middle"
            style="fill:var(--text);font-size:22px;font-weight:700;font-family:var(--font-family);">${escapeXML(fmtN(n))}</text>
      ${hasMort ? `<text x="${x + w/2}" y="${mortY}" text-anchor="middle"
            style="fill:var(--text-muted);font-size:12px;font-family:var(--font-family);">Mortality: ${Number(mort).toFixed(1)}%</text>` : ''}
      ${hasSepsis ? `<text x="${x + w/2}" y="${sepsisY}" text-anchor="middle"
            style="fill:var(--text-muted);font-size:12px;font-family:var(--font-family);">Sepsis: ${sepsisN != null && Number(sepsisN) > 0 ? Number(sepsisN).toLocaleString() + ' (' + Number(sepsisPct).toFixed(1) + '%)' : Number(sepsisPct).toFixed(1) + '%'}</text>` : ''}
    </g>`;
}

function flowArrow(x1, y1, x2, y2) {
  // Small offset on the destination side so the arrowhead doesn't overlap the box border.
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2 - 2}"
                style="stroke:var(--text-secondary);stroke-width:1.5;opacity:0.55;"
                marker-end="url(#t1-arrow)" />`;
}

function escapeXML(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


// ── tiny utilities ───────────────────────────────────────────────────

function fmt(n) {
  if (n === null || n === undefined) return null;
  const num = Number(n);
  if (!Number.isFinite(num)) return String(n);
  return num.toLocaleString();
}

function slug(s) { return String(s || '').replace(/\s+/g, '_'); }

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
