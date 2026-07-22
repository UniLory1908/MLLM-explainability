document.addEventListener("DOMContentLoaded", () => {
  function setQuery(urlString, params) {
    const url = new URL(urlString, window.location.href);
    Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
    return url.toString();
  }

  function mapUrl(caseId, word, layer, mode, threshold, norm) {
    const url = new URL(`/render/map/${caseId}/${word}/${layer}.png`, window.location.origin);
    url.searchParams.set("mode", mode);
    if (threshold !== undefined) url.searchParams.set("threshold", threshold);
    if (norm !== undefined) url.searchParams.set("norm", norm);
    return url.toString();
  }

  function scanpathUrl(caseId, mode, params) {
    const url = new URL(`/render/scanpath/${mode}/${caseId}.png`, window.location.origin);
    Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
    return url.toString();
  }

  function currentThreshold() {
    const threshold = document.getElementById("threshold");
    return threshold ? Number(threshold.value).toFixed(2) : "0.90";
  }

  function selectedLayer() {
    const layerSelect = document.getElementById("layer-select");
    return layerSelect ? layerSelect.value : "0";
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function fmtMetric(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-- unavailable";
    const n = Number(value);
    const a = Math.abs(n);
    if (a >= 100000) return n.toExponential(2);
    if (a >= 1000) return n.toFixed(1);
    if (a >= 1) return n.toFixed(3);
    return n.toFixed(4);
  }

  function updateFitAllMatrices() {
    document.querySelectorAll(".matrix-wrap.zoom-fit").forEach((wrap) => {
      const layerCount = Number(getComputedStyle(wrap).getPropertyValue("--matrix-layer-count")) || 29;
      const wordCol = Number.parseFloat(getComputedStyle(wrap).getPropertyValue("--matrix-word-col-width")) || 120;
      const available = wrap.clientWidth || window.innerWidth;
      const chrome = 28;
      const raw = Math.floor((available - wordCol - chrome) / Math.max(layerCount, 1)) - 3;
      const cellSize = clamp(raw, 28, 76);
      wrap.style.setProperty("--matrix-cell-size", `${cellSize}px`);
    });
  }

  function updateSelectedWord(word, label, detailUrl) {
    const panel = document.getElementById("selected-word-panel");
    if (!panel) return;
    const caseId = panel.dataset.case;
    const layer = selectedLayer();
    const threshold = currentThreshold();
    document.getElementById("selected-word-label").textContent = `${word} ${label}`;
    const selectedMapWord = document.getElementById("selected-map-word");
    if (selectedMapWord) selectedMapWord.textContent = String(word);
    const selectedMapLayer = document.getElementById("selected-map-layer");
    if (selectedMapLayer) selectedMapLayer.textContent = `L${layer}`;
    const detail = document.getElementById("open-word-detail");
    if (detail && detailUrl) detail.href = detailUrl;
    const scan = document.getElementById("selected-layer-scanpath");
    const overlay = document.getElementById("selected-overlay");
    const heatmap = document.getElementById("selected-heatmap");
    if (scan) {
      scan.src = scanpathUrl(caseId, "layer", {word});
      scan.dataset.caption = `Layer-wise scanpath for word ${word} ${label}`;
    }
    if (overlay) {
      overlay.src = mapUrl(caseId, word, layer, "overlay", threshold, "global");
      overlay.dataset.word = word;
      overlay.dataset.layer = layer;
      overlay.dataset.caption = `TAM overlay for word ${word} ${label}, layer L${layer}`;
    }
    if (heatmap) {
      heatmap.src = mapUrl(caseId, word, layer, "overlay", threshold, "local");
      heatmap.dataset.word = word;
      heatmap.dataset.layer = layer;
      heatmap.dataset.caption = `Locally normalized TAM overlay for word ${word} ${label}, layer L${layer}`;
    }
    document.querySelectorAll(".word-button").forEach((button) => button.classList.toggle("selected", button.dataset.word === String(word)));
    fetch(`/api/case/${caseId}/map_metrics?word_index=${word}&layer_index=${layer}`)
      .then((response) => response.json())
      .then((payload) => {
        const metrics = payload.metrics || {};
        document.querySelectorAll("[data-map-metric]").forEach((cell) => {
          cell.textContent = fmtMetric(metrics[cell.dataset.mapMetric]);
        });
      })
      .catch(() => {});
    fetch(`/api/case/${caseId}/word_metrics/${word}`)
      .then((response) => response.json())
      .then((payload) => {
        const metrics = payload.metrics || {};
        document.querySelectorAll("[data-layer-metric]").forEach((cell) => {
          cell.textContent = fmtMetric(metrics[cell.dataset.layerMetric]);
        });
      })
      .catch(() => {});
    fetch(`/api/case/${caseId}/regions?word_index=${word}&layer_index=${layer}`)
      .then((response) => response.json())
      .then((payload) => {
        const body = document.getElementById("region-candidates-body");
        if (!body) return;
        const rows = payload.regions || [];
        if (!rows.length) {
          body.innerHTML = '<tr><td colspan="8">No region candidates for selected word/layer.</td></tr>';
          return;
        }
        body.innerHTML = rows.slice(0, 120).map((region) => (
          `<tr>
            <td>${fmtMetric(region.threshold)}</td>
            <td>${region.rank ?? "-- unavailable"}</td>
            <td>${fmtMetric(region.mass)}</td>
            <td>${fmtMetric(region.ratio_to_primary)}</td>
            <td>${fmtMetric(region.centroid_x_norm)}, ${fmtMetric(region.centroid_y_norm)}</td>
            <td>[${fmtMetric(region.bbox_x0_norm)}, ${fmtMetric(region.bbox_y0_norm)}] -> [${fmtMetric(region.bbox_x1_norm)}, ${fmtMetric(region.bbox_y1_norm)}]</td>
            <td>${region.area ?? "-- unavailable"}</td>
            <td>${fmtMetric(region.peak_value)}</td>
          </tr>`
        )).join("");
      })
      .catch(() => {});
  }

  const threshold = document.getElementById("threshold");
  const thresholdValue = document.getElementById("threshold-value");
  if (threshold && thresholdValue) {
    const updateThreshold = () => {
      thresholdValue.textContent = Number(threshold.value).toFixed(2);
      const overlay = document.getElementById("selected-overlay");
      if (overlay) {
        overlay.src = setQuery(overlay.src, {threshold: Number(threshold.value).toFixed(2)});
      }
      const heatmap = document.getElementById("selected-heatmap");
      if (heatmap) heatmap.src = setQuery(heatmap.src, {threshold: Number(threshold.value).toFixed(2)});
      const scanpath = document.getElementById("word-scanpath");
      if (scanpath) {
        scanpath.src = setQuery(scanpath.src, {threshold: Number(threshold.value).toFixed(2)});
      }
      const selected = document.querySelector(".word-button.selected") || document.querySelector(".word-button");
      if (selected) updateSelectedWord(selected.dataset.word, selected.dataset.label, selected.dataset.detailUrl);
    };
    threshold.addEventListener("input", updateThreshold);
  }

  const layerSelect = document.getElementById("layer-select");
  const scanpath = document.getElementById("word-scanpath");
  if (layerSelect && scanpath) {
    layerSelect.addEventListener("change", () => {
      scanpath.src = setQuery(scanpath.src, {layer: layerSelect.value});
      const selected = document.querySelector(".word-button.selected") || document.querySelector(".word-button");
      if (selected) updateSelectedWord(selected.dataset.word, selected.dataset.label, selected.dataset.detailUrl);
    });
  }

  document.querySelectorAll(".word-button").forEach((button, index) => {
    button.addEventListener("click", () => updateSelectedWord(button.dataset.word, button.dataset.label, button.dataset.detailUrl));
    if (index === 0) button.classList.add("selected");
  });

  const modal = document.getElementById("image-modal");
  const modalImage = document.getElementById("modal-image");
  const modalCaption = document.getElementById("modal-caption");
  const modalClose = document.getElementById("modal-close");

  function openModal(src, caption) {
    if (!modal || !modalImage) return;
    modalImage.src = src;
    modalImage.alt = caption || "";
    if (modalCaption) modalCaption.textContent = caption || src;
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
  }

  function closeModal() {
    if (!modal || !modalImage) return;
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    modalImage.src = "";
  }

  document.querySelectorAll(".modal-image").forEach((img) => {
    img.addEventListener("click", () => openModal(img.src, img.dataset.caption || img.alt));
  });
  document.querySelectorAll(".modal-link").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      openModal(link.href, link.dataset.caption || link.textContent.trim());
    });
  });
  if (modalClose) modalClose.addEventListener("click", closeModal);
  if (modal) {
    modal.addEventListener("click", (event) => {
      if (event.target === modal) closeModal();
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeModal();
  });

  updateFitAllMatrices();
  window.addEventListener("resize", updateFitAllMatrices);
});
