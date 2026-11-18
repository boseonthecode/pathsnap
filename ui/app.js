/* PathSnap demo UI: trips, Leaflet layers, controls, and server-side metrics. */
(function () {
  "use strict";

  var runtimeConfig = window.PATHSNAP_CONFIG || {};
  var API_BASE = (runtimeConfig.apiBaseUrl || "http://localhost:8000").replace(/\/$/, "");
  var DEFAULT_ZOOM = runtimeConfig.defaultZoom || 14;

  var state = {
    map: null,
    trips: [],
    layers: {},
    requestSeq: 0,
    currentTripId: null,
  };

  var elements = {};

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    cacheElements();
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
    state.map = L.map("map", { zoomControl: true }).setView([49.85, 8.6], DEFAULT_ZOOM);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.map);
  }

  function bindControls() {
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
    var payload = {
      trip_id: state.currentTripId,
      method: elements.methodSelect.value,
      noise_sigma_m: Number(elements.noiseSlider.value),
      sample_every_s: Number(elements.sampleSlider.value),
      seed: 42,
    };
    setStatus("Running " + payload.method + " on " + payload.trip_id + "…", false);

    api("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
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
        setStatus("Prediction failed: " + error.message, true);
      });
  }

  function clearLayers() {
    Object.keys(state.layers).forEach(function (name) {
      state.map.removeLayer(state.layers[name]);
    });
    state.layers = {};
  }

  function renderPrediction(data) {
    clearLayers();

    var gpsLayer = L.layerGroup();
    (data.layers.gps_points || []).forEach(function (point) {
      L.circleMarker([point.lat, point.lon], {
        radius: 3,
        color: "#d64545",
        weight: 1,
        fillOpacity: 0.85,
      }).addTo(gpsLayer);
    });
    gpsLayer.addTo(state.map);
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
    candidateLayer.addTo(state.map);
    state.layers.candidates = candidateLayer;

    var routePoints = (data.route_coords || []).map(function (pair) {
      return [pair[1], pair[0]];
    });
    var routeLayer = L.polyline(routePoints, {
      color: "#2fbf71",
      weight: 4,
      opacity: 0.95,
    }).addTo(state.map);
    state.layers.route = routeLayer;

    var boundsPoints = [];
    (data.layers.gps_points || []).forEach(function (point) {
      boundsPoints.push([point.lat, point.lon]);
    });
    routePoints.forEach(function (point) {
      boundsPoints.push(point);
    });
    if (boundsPoints.length > 0) {
      state.map.fitBounds(boundsPoints, { padding: [24, 24] });
    }

    renderMetrics(data.metrics || {});
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
    ["macro_f1", "Macro F1", "percent"],
    ["log_loss", "Log loss", "decimal"],
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
})();
