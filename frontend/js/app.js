const state = {
  file: null,
};

const mediaInput = document.getElementById("mediaInput");
const dropZone = document.getElementById("dropZone");
const processBtn = document.getElementById("processBtn");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const originalContainer = document.getElementById("originalContainer");
const annotatedContainer = document.getElementById("annotatedContainer");
const plateContainer = document.getElementById("plateContainer");
const ocrText = document.getElementById("ocrText");
const violationCard = document.getElementById("violationCard");
const themeToggle = document.getElementById("themeToggle");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const helmetToggle = document.getElementById("helmetToggle");
const toastContainer = document.getElementById("toastContainer");

if (confSlider && confValue) {
  confSlider.addEventListener("input", (e) => {
    confValue.textContent = `${e.target.value}%`;
  });
}

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

let isAuthenticated = false;

function showLogin(onSuccess) {
  if (isAuthenticated) {
    onSuccess();
    return;
  }
  const overlay = document.getElementById("loginOverlay");
  const loginBtn = document.getElementById("loginBtn");
  const err = document.getElementById("loginError");
  overlay.classList.add("active");

  loginBtn.onclick = () => {
    const u = document.getElementById("username").value;
    const p = document.getElementById("password").value;
    if (u === "admin" && p === "admin") {
      isAuthenticated = true;
      overlay.classList.remove("active");
      err.style.display = "none";
      onSuccess();
    } else {
      err.style.display = "block";
    }
  };
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files?.[0];
  if (file) {
    showLogin(() => {
      state.file = file;
      dropZone.querySelector("span").textContent = `Authenticated! Selected: ${file.name}`;
    });
  }
});

mediaInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) {
    state.file = file;
    dropZone.querySelector("span").textContent = `Authenticated! Selected: ${file.name}`;
  }
});

dropZone.addEventListener("click", () => {
  showLogin(() => mediaInput.click());
});

function showToast(title, message) {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<strong>${title}</strong><br><span class="muted">${message}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = "slideIn 0.3s ease reverse forwards";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function renderMedia(container, url, mediaType) {
  container.innerHTML = "";
  if (mediaType === "video") {
    const v = document.createElement("video");
    v.src = url;
    v.controls = true;
    v.autoplay = true;
    v.loop = true;
    container.appendChild(v);
    return;
  }
  const img = document.createElement("img");
  img.src = url;
  container.appendChild(img);
}

function renderViolationCard(data) {
  const isViolation = data.violation_status === "Violation";
  const rClass = data.risk_level === "High" ? "high" : (data.risk_level === "Medium" ? "medium" : "low");
  violationCard.innerHTML = `
    <h3>Detection Result</h3>
    <div class="grid" style="grid-template-columns: 1fr 1fr; margin-bottom: 12px; gap:8px;">
      <p><strong>Case ID:</strong> ${data.case_id || 'N/A'}</p>
      <p><strong>Plate:</strong> ${data.plate_number}</p>
      <p><strong>Risk Level:</strong> <span class="badge ${rClass}">${data.risk_level} (${data.seen_count} seen)</span></p>
      <p><strong>Repeat Offender:</strong> ${data.repeat_offender ? "⚠️ Yes" : "No"}</p>
    </div>
    <div class="grid" style="grid-template-columns: 1fr 1fr; border-top: 1px solid var(--border); padding-top:12px; margin-bottom: 12px; gap:8px;">
      <p><strong>Helmet Status:</strong> ${data.helmet_status}</p>
      <p><strong>Violation:</strong> 
        <span class="${isViolation ? "status-violation" : "status-safe"}">
          ${isViolation ? "🚨 Yes" : "✅ No"}
        </span>
      </p>
      <p title="Algorithm confidence"><strong>Helmet Conf:</strong> ${(data.helmet_conf * 100).toFixed(1)}%</p>
      <p title="Algorithm confidence"><strong>Plate Conf:</strong> ${(data.plate_conf * 100).toFixed(1)}%</p>
    </div>
    <p style="border-top:1px solid var(--border); padding-top:12px;"><strong>Fine Status:</strong> ${data.fine_status}</p>
  `;
}

processBtn.addEventListener("click", async () => {
  if (!state.file) {
    alert("Please upload an image or video first.");
    return;
  }

  loading.classList.remove("hidden");
  results.classList.add("hidden");
  violationCard.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", state.file);
  if (confSlider && helmetToggle) {
    formData.append("confidence_threshold", (confSlider.value / 100.0).toString());
    formData.append("enable_helmet", helmetToggle.checked);
  }

  try {
    const res = await fetch("/api/process", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to process media");

    renderMedia(originalContainer, data.original_url, data.media_type);
    renderMedia(annotatedContainer, data.annotated_url, data.media_type);

    plateContainer.innerHTML = data.plate_crop_url
      ? `<img src="${data.plate_crop_url}" alt="Plate crop" />`
      : "<p class='muted'>Plate crop not available</p>";

    ocrText.textContent = data.plate_number;
    renderViolationCard(data);
    
    // UI enhancements
    if (data.violation_status === "Violation") {
      showToast("🚨 Violation Detected!", `Case: ${data.case_id} &bull; Plate: ${data.plate_number}`);
    } else {
      showToast("✅ Vehicle Cleared", `Plate: ${data.plate_number}`);
    }

    results.classList.remove("hidden");
    violationCard.classList.remove("hidden");
  } catch (err) {
    alert(err.message);
  } finally {
    loading.classList.add("hidden");
  }
});
