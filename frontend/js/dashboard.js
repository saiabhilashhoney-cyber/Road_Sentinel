const recordsBody = document.getElementById("recordsBody");
const searchInput = document.getElementById("searchInput");
const filterSelect = document.getElementById("filterSelect");
const exportBtn = document.getElementById("exportBtn");
const evidenceDialog = document.getElementById("evidenceDialog");
const evidenceContent = document.getElementById("evidenceContent");
const closeDialog = document.getElementById("closeDialog");
const themeToggle = document.getElementById("themeToggle");

function setThemeFromStorage() {
  if (localStorage.getItem("theme") === "light") {
    document.body.classList.add("light");
  }
}

themeToggle?.addEventListener("click", () => {
  document.body.classList.toggle("light");
  localStorage.setItem("theme", document.body.classList.contains("light") ? "light" : "dark");
});
setThemeFromStorage();

function statusClass(row) {
  return row.violation_status === "Violation" ? "status-violation" : "status-safe";
}

function riskBadge(level) {
  const l = level || "Low";
  return `<span class="badge ${l.toLowerCase()}">${l}</span>`;
}

async function loadAnalytics() {
  try {
    const res = await fetch("/api/analytics");
    const data = await res.json();
    const ctx = document.getElementById("analyticsChart");
    if (!ctx) return;
    
    // Check if chart instance exists
    let chartStatus = Chart.getChart("analyticsChart"); 
    if (chartStatus != undefined) {
      chartStatus.destroy();
    }

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: Array.from({length: 24}, (_, i) => `${i}:00`),
        datasets: [{
          label: "Violations per Hour",
          data: data.hourly_violations,
          backgroundColor: "#ef4444",
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, ticks: { precision: 0 } }
        }
      }
    });
  } catch (err) {
    console.error("Failed to load analytics", err);
  }
}

async function updateFine(recordId) {
  try {
    await fetch(`/api/records/${recordId}/fine`, { method: "PUT" });
    loadRecords(); // reload to reflect changes
  } catch(err) {
    alert("Could not update fine status.");
  }
}

function exportEvidence(recordId) {
  window.location.href = `/api/export/${recordId}`;
}

async function loadRecords() {
  try {
    const search = encodeURIComponent(searchInput.value || "");
    const violationsOnly = filterSelect.value === "violations";
    const res = await fetch(`/api/records?search=${search}&violations_only=${violationsOnly}`);
    const rows = await res.json();

    recordsBody.innerHTML = rows
      .map(
        (row) => `
        <tr>
          <td><strong>${row.case_id || '-'}</strong></td>
          <td>${row.plate_number}</td>
          <td>${new Date(row.created_at).toLocaleString()}</td>
          <td>${riskBadge(row.risk_level)}</td>
          <td class="${statusClass(row)}">${row.violation_status}</td>
          <td>
            <div style="font-size:13px; font-weight:600;">${row.fine_status}</div>
            <button class="btn secondary" style="padding: 4px 8px; font-size:12px; margin-top:4px;" onclick="updateFine(${row.id})">Toggle Paid</button>
          </td>
          <td>
            <button class="btn" style="padding: 6px 10px; font-size:13px;" onclick="openEvidence(${row.id})">View</button>
            <button class="btn secondary" style="padding: 6px 10px; font-size:13px; margin-left:4px;" onclick="exportEvidence(${row.id})">Pack</button>
          </td>
          <td><button class="btn btn-delete" style="padding: 6px 10px; font-size:13px;" data-delete-id="${row.id}">Del</button></td>
        </tr>
      `,
      )
      .join("");

    document.querySelectorAll("[data-delete-id]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to permanently delete this record and its evidence?")) return;
        const originalBtnText = btn.textContent;
        btn.textContent = "Del...";
        btn.disabled = true;
        try {
          await fetch(`/api/records/${btn.dataset.deleteId}`, { method: "DELETE" });
          btn.closest("tr").remove(); 
          loadAnalytics();
        } catch (error) {
          console.error("Failed to delete record:", error);
          btn.textContent = originalBtnText;
          btn.disabled = false;
        }
      });
    });
  } catch (error) {
    console.error("Failed to load records:", error);
  }
}

async function openEvidence(recordId) {
  const res = await fetch(`/api/evidence/${recordId}`);
  const row = await res.json();
  evidenceContent.innerHTML = `
    <div class="evidence-compare" style="grid-column: 1 / -1;">
      <div>
        <h4 style="margin-top:0;">Original Frame</h4>
        ${renderMedia(row.original_url, row.media_type)}
      </div>
      <div>
        <h4 style="margin-top:0;">Processed Annotations</h4>
        ${renderMedia(row.annotated_url, row.media_type)}
      </div>
    </div>
    <div style="grid-column: 1 / -1; display:flex; gap: 20px; align-items:center; background:var(--bg-soft); padding:16px; border-radius:12px; border:1px solid var(--border);">
      <div>
        <h4 style="margin-top:0; margin-bottom:8px;">Extracted Plate Crop</h4>
        ${row.plate_crop_url ? `<img src="${row.plate_crop_url}" style="max-width:200px;" />` : "<p class='muted'>Not available</p>"}
      </div>
      <div style="flex:1;">
        <h4 style="margin-top:0; margin-bottom:8px;">Detection Metadata</h4>
        <p><strong>Helmet Confidence:</strong> ${(row.helmet_conf * 100 || 0).toFixed(1)}%</p>
        <p><strong>Plate Confidence:</strong> ${(row.plate_conf * 100 || 0).toFixed(1)}%</p>
        <p><strong>Repeated Occurrences (Seen count):</strong> ${row.seen_count || 0}</p>
      </div>
    </div>
  `;
  evidenceDialog.showModal();
}

function renderMedia(url, mediaType) {
  if (mediaType === "video") {
    return `<video src="${url}" controls autoplay loop muted playsinline></video>`;
  }
  return `<img src="${url}" />`;
}

closeDialog.addEventListener("click", () => evidenceDialog.close());

exportBtn.addEventListener("click", () => {
  const tableRows = Array.from(document.querySelectorAll("#recordsBody tr")).map((tr) => {
    return Array.from(tr.querySelectorAll("td"))
      .slice(0, 6) 
      .map((td) => `"${td.innerText.replace(/\n/g, " ")}"`)
      .join(",");
  });
  const csvContent = "data:text/csv;charset=utf-8,Case ID,Plate Number,Date Time,Risk Level,Status,Fine Mgmt\n" + tableRows.join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "violation_records.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

searchInput.addEventListener("input", loadRecords);
filterSelect.addEventListener("change", loadRecords);

loadRecords();
loadAnalytics();
