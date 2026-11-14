const SWPC = "https://services.swpc.noaa.gov";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const C = { text: css("--text-2"), muted: css("--muted"), grid: css("--border"), s1: css("--s1"), s2: css("--s2"), s3: css("--s3"), obs: css("--text"),
  good: css("--good"), warn: css("--warn"), crit: css("--crit") };
Chart.defaults.color = C.text;
Chart.defaults.borderColor = C.grid;
Chart.defaults.font.family = "system-ui, sans-serif";
Chart.defaults.animation = false;

const getJSON = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); });
const utc = (s) => new Date(s.endsWith("Z") ? s : s + "Z");
const hhmm = (d) => d.toISOString().slice(5, 16).replace("T", " ");
const kpColor = (kp) => (kp >= 7 ? C.crit : kp >= 5 ? C.warn : C.good);
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

function lineChart(id, labels, datasets, yOpts = {}) {
  return new Chart(document.getElementById(id), {
    type: "line",
    data: { labels, datasets: datasets.map((d) => ({ borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, ...d })) },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: datasets.length > 1, labels: { boxWidth: 12, boxHeight: 2 } } },
      scales: { x: { ticks: { maxTicksLimit: 6, maxRotation: 0 }, grid: { display: false } }, y: yOpts },
    },
  });
}

// Same features as train.py:make_features, computed on hourly means of the 1-min real-time feed.
function features(wind, mag, kpPrev) {
  const now = new Date();
  const end = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), now.getUTCHours());
  const hours = [];
  for (let h = 3; h >= 1; h--) {
    const a = end - h * 3600e3, b = a + 3600e3;
    const inHour = (r) => { const t = utc(r.time_tag).getTime(); return t >= a && t < b; };
    const w = wind.filter(inHour), m = mag.filter(inHour);
    if (w.length < 10 || m.length < 10) return null; // data gap: refuse to forecast
    const v = mean(w.map((r) => r.proton_speed)), n = mean(w.map((r) => r.proton_density));
    const bt = mean(m.map((r) => r.bt)), by = mean(m.map((r) => r.by_gsm)), bz = mean(m.map((r) => r.bz_gsm));
    const theta = Math.atan2(by, bz);
    hours.push({ v, n, bt, bz, vbs: (v * Math.max(-bz, 0)) / 1000,
      newell: 1e-3 * v ** (4 / 3) * bt ** (2 / 3) * Math.abs(Math.sin(theta / 2)) ** (8 / 3) });
  }
  const avg = (k) => mean(hours.map((h) => h[k]));
  return { kp_prev: kpPrev, v: avg("v"), n: avg("n"), bt: avg("bt"), bz: avg("bz"),
    bz_min: Math.min(...hours.map((h) => h.bz)), vbs: avg("vbs"), newell: avg("newell") };
}

function predict(model, f) {
  const y = model.features.reduce((s, k, i) => s + (model.coef[i] * (f[k] - model.mean[i])) / model.scale[i], model.intercept);
  return Math.min(9, Math.max(0, y));
}

function tile(label, value, unit, color) {
  return `<div class="tile"><div class="label">${label}</div><div class="value"${color ? ` style="color:${color}"` : ""}>${value}</div><div class="unit">${unit}</div></div>`;
}

async function live() {
  const [windAll, magAll, kpFc, model] = await Promise.all([
    getJSON(`${SWPC}/json/rtsw/rtsw_wind_1m.json`),
    getJSON(`${SWPC}/json/rtsw/rtsw_mag_1m.json`),
    getJSON(`${SWPC}/products/noaa-planetary-k-index-forecast.json`),
    getJSON("model.json"),
  ]);
  // the feed mixes spacecraft; keep the active one and valid samples, oldest first
  const byTime = (a, b) => utc(a.time_tag) - utc(b.time_tag);
  const wind = windAll.filter((r) => r.active && r.proton_speed != null && r.proton_density != null).sort(byTime);
  const mag = magAll.filter((r) => r.active && r.bz_gsm != null && r.bt != null).sort(byTime);

  const observed = kpFc.filter((r) => r.observed !== "predicted");
  const lastKp = observed[observed.length - 1];
  const f = features(wind, mag, lastKp.kp);
  const fc = f && predict(model, f);
  const w = wind[wind.length - 1], m = mag[mag.length - 1];

  document.getElementById("tiles").innerHTML =
    tile("Solar wind speed", w.proton_speed.toFixed(0), "km/s") +
    tile("Density", w.proton_density.toFixed(1), "p/cm³") +
    tile("IMF Bz (GSM)", m.bz_gsm.toFixed(1), "nT", m.bz_gsm < -10 ? C.crit : undefined) +
    tile("Bt", m.bt.toFixed(1), "nT") +
    tile(`Kp ${hhmm(utc(lastKp.time_tag))} UTC`, lastKp.kp.toFixed(2), String(lastKp.observed).replace(/[^\w ]/g, ""), kpColor(lastKp.kp)) +
    tile("Our Kp forecast, next 3 h", fc == null ? "–" : fc.toFixed(1), fc == null ? "data gap" : "ridge model", fc == null ? undefined : kpColor(fc));
  document.getElementById("verdict").textContent =
    `Latest L1 sample ${hhmm(utc(w.time_tag))} UTC. Rule of thumb: aurora on the northern horizon needs about Kp ≥ 6 in Berlin/Warsaw and Kp ≥ 7 in Budapest/Vienna. ` +
    (fc != null && fc >= 6 ? "Forecast is in that range — look north tonight!" : "No Central European aurora expected in the next 3 h.");

  const day = Date.now() - 24 * 3600e3;
  const w24 = wind.filter((r) => utc(r.time_tag) >= day), m24 = mag.filter((r) => utc(r.time_tag) >= day);
  lineChart("cSpeed", w24.map((r) => hhmm(utc(r.time_tag))), [{ label: "Speed", data: w24.map((r) => r.proton_speed), borderColor: C.s1 }]);
  lineChart("cBz", m24.map((r) => hhmm(utc(r.time_tag))), [{ label: "Bz", data: m24.map((r) => r.bz_gsm), borderColor: C.s2 }]);

  new Chart(document.getElementById("cKp"), {
    type: "bar",
    data: {
      labels: kpFc.map((r) => hhmm(utc(r.time_tag))),
      datasets: [{
        label: "Kp", data: kpFc.map((r) => r.kp), borderRadius: 4, categoryPercentage: 0.9,
        backgroundColor: kpFc.map((r) => kpColor(r.kp) + (r.observed === "predicted" ? "55" : "")),
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `Kp ${c.raw} (${kpFc[c.dataIndex].observed})` } } },
      scales: { y: { min: 0, max: 9, ticks: { stepSize: 1 } }, x: { ticks: { maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } } },
    },
  });
}

async function aurora() {
  const map = L.map("map", { preferCanvas: true }).setView([58, 15], 3);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 8, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  const ov = await getJSON(`${SWPC}/json/ovation_aurora_latest.json`);
  const grid = new Map();
  for (const [lon0, lat, p] of ov.coordinates) {
    const lon = lon0 > 180 ? lon0 - 360 : lon0;
    grid.set(`${lon},${lat}`, p);
    if (p < 3 || lat < 30 || lat > 85 || lon < -70 || lon > 90) continue;
    // sequential single hue (aurora green), opacity encodes probability
    L.rectangle([[lat - 0.5, lon - 0.5], [lat + 0.5, lon + 0.5]], { stroke: false, fillColor: C.good, fillOpacity: Math.min(0.85, 0.15 + p / 40) })
      .bindTooltip(`${lat}°N ${lon}°E: ${p}%`).addTo(map);
  }
  const cities = { Budapest: [47.5, 19.04], Vienna: [48.21, 16.37], Prague: [50.08, 14.44], Berlin: [52.52, 13.4], Warsaw: [52.23, 21.01] };
  const local = Object.entries(cities).map(([name, [lat, lon]]) => {
    const p = grid.get(`${Math.round(lon)},${Math.round(lat)}`) ?? 0;
    L.circleMarker([lat, lon], { radius: 5, color: "#000", weight: 2, fillOpacity: 0 }).bindTooltip(`${name}: ${p}% overhead`).addTo(map);
    return `${name} ${p}%`;
  });
  document.getElementById("mapnote").textContent =
    `Probability of aurora overhead (green, darker = likelier), OVATION run ${ov["Observation Time"]}, valid ${ov["Forecast Time"]}. ` +
    `Overhead now: ${local.join(", ")}. Aurora can be seen on the horizon several hundred km south of the shaded oval.`;
}

async function backtest() {
  const bt = await getJSON("backtest.json");
  const models = [["persistence", "Persistence (last Kp)", C.s1], ["ridge", "Ridge (live model)", C.s2], ["gbm", "Gradient boosting", C.s3]];
  const row = (name, m) => `<tr><td>${name}</td><td>${m.rmse.toFixed(2)}</td><td>${m.mae.toFixed(2)}</td><td>${m.n_storm ? m.rmse_storm.toFixed(2) : "–"}</td><td>${m.pod.toFixed(2)}</td><td>${m.far.toFixed(2)}</td><td>${m.hss.toFixed(2)}</td></tr>`;
  const table = (id, metrics) => {
    document.getElementById(id).innerHTML = "<tr><th>Model</th><th>RMSE</th><th>MAE</th><th>RMSE (storm)</th><th>POD</th><th>FAR</th><th>HSS</th></tr>" +
      models.map(([k, name]) => row(name, metrics[k])).join("");
  };
  table("tAll", bt.overall);

  const sel = document.getElementById("storm");
  sel.innerHTML = Object.keys(bt.storms).map((s) => `<option>${s}</option>`).join("");
  sel.value = "May 2024 (Gannon)";
  let chart;
  const show = () => {
    const s = bt.storms[sel.value];
    chart?.destroy();
    chart = lineChart("cStorm", s.time.map((t) => hhmm(new Date(t))), [
      { label: "Observed (GFZ)", data: s.observed, borderColor: C.obs, stepped: "before", borderWidth: 3 },
      ...models.map(([k, name, color]) => ({ label: name, data: s[k], borderColor: color, stepped: "before" })),
    ], { min: 0, max: 9, ticks: { stepSize: 1 } });
    table("tStorm", s.metrics);
  };
  sel.onchange = show;
  show();
}

for (const [fn, el] of [[live, "tiles"], [aurora, "mapnote"], [backtest, "tAll"]]) {
  fn().catch((e) => { document.getElementById(el).innerHTML = `<span class="err">Failed to load: ${e.message}</span>`; });
}
