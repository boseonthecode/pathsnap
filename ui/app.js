/* PathSnap demo UI: trips, Leaflet layers, controls, and server-side metrics. */
(function () {
  "use strict";

  var runtimeConfig = window.PATHSNAP_CONFIG || {};
  var API_BASE = (window.location.port === "8000" ? window.location.origin : (runtimeConfig.apiBaseUrl || "http://localhost:8000")).replace(/\/$/, "");
  var DEFAULT_ZOOM = runtimeConfig.defaultZoom || 14;

  var state = {
    map: null,
    trips: [],
    layers: {},
    requestSeq: 0,
    currentTripId: null,
    predictionController: null,
    benchmarkSeq: 0,
  };

  var elements = {};

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    cacheElements();
    bindResults();
    loadBenchmarks();
    if (typeof L === "undefined") {
      setStatus("Leaflet failed to load.", true);
      return;
    }
    setupMap();
    bindControls();
    loadHealth();
    loadTrips();
  }

  function cacheElements() {
    elements.health = document.getElementById("health");
    elements.tripSelect = document.getElementById("trip-select");
    elements.methodSelect = document.getElementById("method-select");
    elements.noiseSlider = document.getElementById("noise-slider");
    elements.noiseValue = document.getElementById("noise-value");
    elements.sampleSlider = document.getElementById("sample-slider");
    elements.sampleValue = document.getElementById("sample-value");
    elements.rerunButton = document.getElementById("rerun-button");
    elements.metrics = document.getElementById("metrics");
    elements.status = document.getElementById("status");
  }

  function setupMap() {
    state.map = L.map("map", { zoomControl: false, preferCanvas: true }).setView([49.85, 8.6], DEFAULT_ZOOM);
    L.control.zoom({position: "bottomright"}).addTo(state.map);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.map);
  }

  function bindControls() {
    document.getElementById("fit-button").addEventListener("click", fitJourney);
    ["route", "gps", "candidates"].forEach(function (name) {
      document.getElementById("layer-" + name).addEventListener("change", function (event) {
        var layer = state.layers[name];
        if (layer) { if (event.target.checked) layer.addTo(state.map); else state.map.removeLayer(layer); }
      });
    });
    elements.noiseSlider.addEventListener("input", function () {
      elements.noiseValue.textContent = elements.noiseSlider.value + " m";
      debouncedPredict();
    });
    elements.sampleSlider.addEventListener("input", function () {
      elements.sampleValue.textContent = elements.sampleSlider.value + " s";
      debouncedPredict();
    });
    elements.methodSelect.addEventListener("change", requestPrediction);
    elements.rerunButton.addEventListener("click", requestPrediction);
    elements.tripSelect.addEventListener("change", function () {
      state.currentTripId = elements.tripSelect.value;
      requestPrediction();
    });
  }

  var debouncedPredict = debounce(function () {
    requestPrediction();
  }, 250);

  function debounce(fn, delayMs) {
    var timer = null;
    return function () {
      var args = arguments;
      window.clearTimeout(timer);
      timer = window.setTimeout(function () {
        fn.apply(null, args);
      }, delayMs);
    };
  }

  function api(path, options) {
    var opts = options || {};
    opts.headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
    return fetch(API_BASE + path, opts).then(function (response) {
      if (!response.ok) {
        return response
          .json()
          .catch(function () {
            return {};
          })
          .then(function (body) {
            throw new Error(body.detail || "HTTP " + response.status);
          });
      }
      return response.json();
    });
  }

  function loadHealth() {
    api("/health")
      .then(function (health) {
        elements.health.textContent = health.model_loaded
          ? "API ready · " + (health.model_name || "model")
          : "API starting";
        elements.health.className = "health health--ok";
      })
      .catch(function () {
        elements.health.textContent = "API offline";
        elements.health.className = "health health--error";
      });
  }

  function loadTrips() {
    api("/trips")
      .then(function (payload) {
        state.trips = payload.trips || [];
        document.getElementById("trip-count").textContent = state.trips.length + " trips";
        elements.tripSelect.innerHTML = "";
        state.trips.forEach(function (trip) {
          var option = document.createElement("option");
          option.value = trip.trip_id;
          option.textContent =
            trip.trip_id + " · " + trip.point_count + " points";
          elements.tripSelect.appendChild(option);
        });
        if (state.trips.length === 0) {
          setStatus("No evaluation trips are mounted.", true);
          return;
        }
        state.currentTripId = state.trips[0].trip_id;
        elements.tripSelect.value = state.currentTripId;
        requestPrediction();
      })
      .catch(function (error) {
        elements.health.textContent = "API offline";
        elements.health.className = "health health--error";
        setStatus("Could not load trips: " + error.message, true);
      });
  }

  function requestPrediction() {
    if (!state.currentTripId) {
      return;
    }
    var sequence = ++state.requestSeq;
    if (state.predictionController) state.predictionController.abort();
    var controller = new AbortController();
    state.predictionController = controller;
    var timeout = window.setTimeout(function () { controller.abort(); }, 60000);
    var payload = {
      trip_id: state.currentTripId,
      method: elements.methodSelect.value,
      noise_sigma_m: Number(elements.noiseSlider.value),
      sample_every_s: Number(elements.sampleSlider.value),
      seed: 42,
    };
    setStatus("Running " + payload.method + " on " + payload.trip_id + "…", false);
    elements.rerunButton.disabled = true;
    document.getElementById("journey-card").setAttribute("aria-busy", "true");
    document.getElementById("journey-title").textContent = "Journey " + payload.trip_id;
    document.getElementById("journey-detail").textContent = "Matching GPS observations to the road network…";
    ["hero-accuracy", "hero-decoder", "hero-route"].forEach(function (id) { document.getElementById(id).textContent = "—"; });
    elements.metrics.textContent = "Recalculating this journey’s scores…";
    if (state.renderedTrip !== payload.trip_id) clearLayers();

    api("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
      signal: controller.signal,
    })
      .then(function (data) {
        if (sequence !== state.requestSeq) {
          return; /* A newer request has superseded this response. */
        }
        renderPrediction(data);
      })
      .catch(function (error) {
        if (sequence !== state.requestSeq) {
          return;
        }
        var message = error.name === "AbortError" ? "The request timed out. Press Run to retry." : "Prediction failed: " + error.message;
        document.getElementById("journey-detail").textContent = "No result for these settings. Press Run to retry.";
        elements.metrics.textContent = "Scores unavailable for this request.";
        setStatus(message, true);
      }).finally(function () {
        window.clearTimeout(timeout);
        if (sequence === state.requestSeq) {
          elements.rerunButton.disabled = false;
          document.getElementById("journey-card").setAttribute("aria-busy", "false");
        }
      });
  }

  function clearLayers() {
    Object.keys(state.layers).forEach(function (name) {
      state.map.removeLayer(state.layers[name]);
    });
    state.layers = {};
  }

  function renderPrediction(data) {
    state.renderedTrip = data.trip_id;
    clearLayers();

    var gpsLayer = L.layerGroup();
    (data.layers.gps_points || []).forEach(function (point) {
      L.circleMarker([point.lat, point.lon], {
        radius: 2.5,
        color: "#e87e51",
        weight: 1,
        fillOpacity: 0.85,
      }).addTo(gpsLayer);
    });
    if (document.getElementById("layer-gps").checked) gpsLayer.addTo(state.map);
    state.layers.gps = gpsLayer;

    var candidateLayer = L.layerGroup();
    (data.layers.candidate_points || []).forEach(function (point) {
      L.circleMarker([point.lat, point.lon], {
        radius: 1.6,
        color: "#e0a030",
        weight: 0,
        fillOpacity: 0.4,
      }).addTo(candidateLayer);
    });
    if (document.getElementById("layer-candidates").checked) candidateLayer.addTo(state.map);
    state.layers.candidates = candidateLayer;

    var routePoints = (data.route_coords || []).map(function (pair) {
      return [pair[1], pair[0]];
    });
    var routeLayer = L.layerGroup();
    L.polyline(routePoints, {color: "#fff", weight: 9, opacity: 0.95}).addTo(routeLayer);
    L.polyline(routePoints, {
      color: "#1a73e8",
      weight: 5,
      opacity: 0.95,
    }).addTo(routeLayer);
    if (routePoints.length) {
      L.marker(routePoints[0], {icon: L.divIcon({className: "endpoint-marker", iconSize: [15,15]})}).bindPopup("Journey start").addTo(routeLayer);
      L.marker(routePoints[routePoints.length-1], {icon: L.divIcon({className: "endpoint-marker end", iconSize: [15,15]})}).bindPopup("Journey end").addTo(routeLayer);
    }
    if (document.getElementById("layer-route").checked) routeLayer.addTo(state.map);
    state.layers.route = routeLayer;

    var boundsPoints = [];
    (data.layers.gps_points || []).forEach(function (point) {
      boundsPoints.push([point.lat, point.lon]);
    });
    routePoints.forEach(function (point) {
      boundsPoints.push(point);
    });
    if (boundsPoints.length > 0) {
      state.bounds = boundsPoints;
      fitJourney();
    }

    renderMetrics(data.metrics || {});
    document.getElementById("journey-title").textContent = "Journey " + data.trip_id;
    document.getElementById("journey-detail").textContent = data.point_predictions.length.toLocaleString() + " GPS points · " + data.predicted_arcs.length + " road segments · " + (data.inference_ms / 1000).toFixed(2) + " s";
    document.getElementById("hero-accuracy").textContent = formatValue(data.metrics.top1_accuracy, "percent");
    document.getElementById("hero-decoder").textContent = formatValue(data.metrics.decoder_macro_f1, "percent");
    document.getElementById("hero-route").textContent = formatValue(data.metrics.route_f1, "percent");
    setStatus(
      data.trip_id +
        " · " +
        data.method +
        " · " +
        Math.round(data.inference_ms) +
        " ms",
      false
    );
  }

  var METRIC_FIELDS = [
    ["top1_accuracy", "Top-1 accuracy", "percent"],
    ["macro_f1", "Classifier Macro F1", "percent"],
    ["decoder_macro_f1", "Decoder Macro F1", "percent"],
    ["route_f1", "Route sequence F1", "percent"],
    ["precision", "Classifier precision", "percent"],
    ["recall", "Classifier recall", "percent"],
    ["log_loss", "Classifier log loss ↓", "decimal"],
    ["candidate_miss_rate", "Candidate miss rate", "percent"],
    ["route_continuity", "Route continuity", "percent"],
    ["route_precision", "Route precision", "percent"],
    ["route_recall", "Route recall", "percent"],
    ["rmf", "Route mismatch fraction", "decimal"],
    ["alignment_coverage", "Alignment coverage", "percent"],
  ];

  function formatValue(value, kind) {
    if (typeof value !== "number" || !isFinite(value)) {
      return "N/A";
    }
    if (kind === "percent") {
      return (value * 100).toFixed(2) + "%";
    }
    return value.toFixed(4);
  }

  function renderMetrics(metrics) {
    elements.metrics.innerHTML = "";
    METRIC_FIELDS.forEach(function (field) {
      var key = field[0];
      if (!(key in metrics)) {
        return;
      }
      var row = document.createElement("div");
      row.className = "metrics__row";
      var label = document.createElement("span");
      label.className = "metrics__label";
      label.textContent = field[1];
      var value = document.createElement("span");
      value.className = "metrics__value";
      value.textContent = formatValue(metrics[key], field[2]);
      row.appendChild(label);
      row.appendChild(value);
      elements.metrics.appendChild(row);
    });
  }

  function setStatus(message, isError) {
    elements.status.textContent = message;
    elements.status.className = isError ? "status status--error" : "status";
  }

  function fitJourney() {
    if (!state.bounds) return;
    var mobile = window.innerWidth <= 700;
    state.map.fitBounds(state.bounds, {
      paddingTopLeft: mobile ? [25,60] : [410,90],
      paddingBottomRight: mobile ? [25,window.innerHeight * 0.44 + 110] : [70,175], maxZoom: 17
    });
  }

  function bindResults() {
    function show(open) {
      if (open) loadBenchmarks();
      document.getElementById("results-panel").hidden = !open;
      document.body.classList.toggle("results-open", open);
      [".panel", ".map-toolbar", "#journey-card", "#map"].forEach(function (selector) { document.querySelector(selector).inert = open; });
      document.getElementById("results-tab").classList.toggle("active", open);
      document.getElementById("map-tab").classList.toggle("active", !open);
      document.getElementById(open ? "close-results" : "results-tab").focus();
    }
    document.getElementById("results-tab").onclick = function () { show(true); };
    document.getElementById("map-tab").onclick = function () { show(false); };
    document.getElementById("close-results").onclick = function () { show(false); };
    document.addEventListener("keydown", function (event) {
      var panel = document.getElementById("results-panel");
      if (panel.hidden) return;
      if (event.key === "Escape") show(false);
      if (event.key === "Tab") {
        var focusable = Array.from(panel.querySelectorAll('button:not(:disabled), [tabindex="0"]'));
        var first = focusable[0], last = focusable[focusable.length-1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    });
    document.getElementById("download-metrics").onclick = function () {
      var lines = [["Metric", "5-fold CV", "Student test", "Evaluator test"]];
      document.querySelectorAll("#benchmark-table tr").forEach(function (row) {
        lines.push(Array.from(row.cells).map(function (cell) { return cell.textContent; }));
      });
      var csv = lines.map(function (row) { return row.map(function (value) { return '"' + value.replace(/"/g, '""') + '"'; }).join(","); }).join("\r\n");
      var url = URL.createObjectURL(new Blob(["\uFEFF" + csv], {type: "text/csv;charset=utf-8"}));
      var link = document.createElement("a"); link.href = url; link.download = "pathsnap-metrics.csv";
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    };
    window.addEventListener("resize", debounce(function () { if (state.map) { state.map.invalidateSize(); fitJourney(); } }, 200));
  }

  function loadBenchmarks() {
    var sequence = ++state.benchmarkSeq;
    fetch("metrics.json", {cache: "no-cache"}).then(function (response) {
      if (!response.ok) throw new Error("Metric report is not available yet");
      return response.json();
    }).then(function (data) {
      if (sequence !== state.benchmarkSeq) return;
      var cv = data.cv, student = data.partitions.student_test, evaluator = data.partitions.evaluator_test;
      document.getElementById("model-details").textContent = data.params.n_estimators + " trees · maximum depth " + data.params.max_depth + " · " + data.feature_count + " features · " + data.training_trips + " training trips · up to " + data.max_observations_per_trip + " observations per trip";
      var cards = document.getElementById("benchmark-cards");
      cards.replaceChildren();
      [["CV Macro F1",cv.macro_f1,"Mean of 5 held-out folds"],["Evaluator Top-1",evaluator.decoded_pooled_top1,"After decoding · pooled points"],["Decoder Macro F1",evaluator.decoded_mean_trip_macro_f1,"Evaluator · mean across trips"]].forEach(function (item) {
        var card = document.createElement("div"); card.className = "benchmark-card";
        var title = document.createElement("small"); title.textContent = item[0];
        var value = document.createElement("strong"); value.textContent = formatValue(item[1], "percent");
        var note = document.createElement("p"); note.textContent = item[2];
        card.append(title,value,note); cards.appendChild(card);
      });
      var rows = [
        ["Classifier Macro F1", "macro_f1", "percent"], ["Classifier Macro F1 · mean trip", "mean_trip_classifier_macro_f1", "percent"], ["Classifier Top-1", "top1_accuracy", "percent"],
        ["Classifier log loss ↓", "log_loss", "decimal"], ["Classifier precision", "precision", "percent"],
        ["Classifier recall", "recall", "percent"], ["Decoded Top-1 · mean trip", "decoded_mean_trip_top1", "percent"],
        ["Decoded Top-1 · pooled", "decoded_pooled_top1", "percent"], ["Decoder Macro F1 · mean trip", "decoded_mean_trip_macro_f1", "percent"],
        ["Decoder Macro F1 · pooled", "decoded_pooled_macro_f1", "percent"],
        ["Route sequence F1", "route_f1", "percent"], ["Route precision", "route_precision", "percent"],
        ["Route recall", "route_recall", "percent"], ["Route continuity", "route_continuity", "percent"],
        ["Route mismatch ↓", "rmf", "decimal"], ["Alignment coverage", "alignment_coverage", "percent"]
      ];
      var body = document.getElementById("benchmark-table"); body.replaceChildren();
      rows.forEach(function (row) {
        var tr = document.createElement("tr"), title = document.createElement("td"); title.textContent = row[0]; tr.appendChild(title);
        [cv,student,evaluator].forEach(function (split) { var td = document.createElement("td"); td.textContent = formatValue(split[row[1]],row[2]); tr.appendChild(td); });
        body.appendChild(tr);
      });
      document.getElementById("evaluation-protocol").textContent = data.protocol + " CV Macro F1 standard deviation: " + (cv.macro_f1_std*100).toFixed(2) + " percentage points. N/A means not evaluated in that protocol.";
      document.getElementById("metrics-updated").textContent = "Computed " + new Date(data.generated_at).toLocaleString() + ". The live journey panel reflects its selected noise and sampling settings; this table uses the original datasets.";
      document.getElementById("download-metrics").disabled = false;
    }).catch(function (error) {
      if (sequence !== state.benchmarkSeq) return;
      document.getElementById("evaluation-protocol").textContent = "Could not refresh saved evaluation results. Close and reopen Model results to retry.";
      if (!document.querySelector(".benchmark-card")) document.getElementById("benchmark-table").textContent = "Evaluation report unavailable.";
    });
  }
})();
