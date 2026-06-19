const researchButton = document.getElementById("research-button");
const heroResearchButton = document.getElementById("hero-research-button");
const navResearchButton = document.getElementById("nav-research-button");
const researchTarget = document.getElementById("research-target");
const researchProfile = document.getElementById("research-profile");
const researchStatus = document.getElementById("research-status");
let targetPayload = null;

function explainError(body) {
  try {
    return JSON.parse(body).detail || body;
  } catch {
    return body;
  }
}

function focusLauncher() {
  document.querySelector(".research-launch")?.scrollIntoView({ behavior: "smooth", block: "center" });
  window.setTimeout(() => researchTarget.focus(), 450);
}

function storeAndOpen(payload) {
  sessionStorage.setItem(`geochemad:research:${payload.sessionId}`, JSON.stringify(payload));
  window.location.href = `/web/analysis.html?research_id=${encodeURIComponent(payload.sessionId)}`;
}

async function loadTargets() {
  const profile = researchProfile.disabled || !["demo", "full"].includes(researchProfile.value)
    ? "demo"
    : researchProfile.value;
  try {
    const response = await fetch(`/api/research/targets?profile=${encodeURIComponent(profile)}`);
    if (!response.ok) throw new Error(explainError(await response.text()));
    const payload = await response.json();
    targetPayload = payload;
    if (researchProfile.disabled) {
      researchProfile.innerHTML = (payload.profiles || [])
        .map((item) =>
          `<option value="${item.key}" ${item.key === profile ? "selected" : ""} ${item.available ? "" : "disabled"}>` +
          `${item.label}${item.available ? "" : " (missing)"}</option>`
        )
        .join("");
      researchProfile.disabled = false;
    }
    researchTarget.innerHTML = payload.targets
      .map((target) => `<option value="${target.key}">${target.label} (${target.key}) · spatial AUC ${target.spatialAuc.toFixed(3)}</option>`)
      .join("");
    researchTarget.disabled = false;
    researchButton.disabled = false;
    const profileName = payload.dataProfile?.name || "Current data";
    researchStatus.textContent = `${payload.targets.length} supervisor targets available · ${profileName} · geochem / GSWA stream sediment.`;
  } catch (error) {
    researchStatus.textContent = `Unable to read supervisor targets: ${error.message || error}`;
  }
}

async function openResearch() {
  const target = researchTarget.value || "Cu";
  const profile = researchProfile.value || "demo";
  researchButton.disabled = true;
  researchTarget.disabled = true;
  researchProfile.disabled = true;
  const profileName = targetPayload?.dataProfile?.name || profile;
  researchStatus.textContent = profile === "full"
    ? `Loading the supervisor ${target} model with ${profileName}. Full geophysics/structure setup can take several minutes on first load...`
    : `Loading the supervisor ${target} model with ${profileName}. Initial fitting may take 10–25 seconds...`;
  try {
    const response = await fetch(`/api/research/start?${new URLSearchParams({ target, profile }).toString()}`, { method: "POST" });
    if (!response.ok) throw new Error(explainError(await response.text()));
    storeAndOpen(await response.json());
  } catch (error) {
    researchStatus.textContent = `Supervisor model failed to load: ${error.message || error}`;
    researchButton.disabled = false;
    researchTarget.disabled = false;
    researchProfile.disabled = false;
  }
}

researchButton.addEventListener("click", openResearch);
researchProfile.addEventListener("change", () => {
  researchTarget.disabled = true;
  researchButton.disabled = true;
  researchStatus.textContent = "Switching data profile...";
  loadTargets();
});
heroResearchButton.addEventListener("click", focusLauncher);
navResearchButton.addEventListener("click", focusLauncher);
loadTargets();
