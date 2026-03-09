export function renderInstructions(el) {
  el.innerHTML = `
    <h1>How to Use This App</h1>
    <h3>What's This About?</h3>
    <p>This app validates your CLIF 2.1 data against the official spec.</p>
    <h3>Your Workflow:</h3>
    <ol>
      <li><strong>Review Each Table</strong> - Click through the tables in the grid on the Home page</li>
      <li><strong>Review & Provide Feedback</strong> - For each error, decide: Accept, Reject, or Pending. Saving feedback auto-regenerates reports.</li>
      <li><strong>Fixed Something? Re-run Validation</strong> - Select the affected table(s) and click "Run Validation" to re-validate. Reports update automatically.</li>
      <li><strong>Check Out Table One</strong> - Explore cohorts, demographics, meds, ventilation, outcomes</li>
    </ol>
  `;
}
