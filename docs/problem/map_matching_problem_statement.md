# Problem Statement: Map Matching (PathSnap)

## 1. Overview & Context: GPS Trajectories
Modern vehicles navigating through urban environments continuously record their locations using Global Positioning System (GPS) devices. As a vehicle moves, these devices capture a sequence of geographical coordinates (latitude and longitude) at regular time intervals. When ordered chronologically, these recorded points form a **GPS Trajectory**.

---

## 2. The Core Challenge: GPS Measurement Noise
While GPS technology provides valuable tracking data, raw GPS signals are inherently imperfect and rarely align cleanly with physical infrastructure. 

### Causes of Drift & Error
In dense urban environments, several factors degrade signal accuracy:
* **Urban Canyons & Buildings:** Tall structures block or reflect satellite signals (multipath interference).
* **Signal Interference:** Atmospheric conditions and electronic noise disrupt transmission.
* **Hardware Measurement Errors:** Inherent limitations in consumer-grade GPS receivers.

### Impact on Data
Because of these factors, the **observed GPS coordinates** frequently drift away from the **actual road** the vehicle is traveling on. Plotting raw GPS observations directly onto a map often places the vehicle inside buildings, across blocks, or on adjacent, incorrect streets.

---

## 3. Defining the Problem: Map Matching
Given a set of noisy, inaccurate GPS points overlaid on a connected road network, a fundamental question arises:

> *"Which roads did the vehicle actually travel on?"*

This challenge is known in spatial computing and transportation analytics as **Map Matching**. Simply snapping each individual GPS point to its nearest geometric road segment is insufficient, as it ignores the physical constraints of driving (e.g., connectivity, one-way streets, turn restrictions, and logical path continuity).

---

## 4. Objective: Recover the Route (PathSnap)
The primary objective of **PathSnap** (Map Matching) is to bridge the gap between noisy sensor observations and the physical road network.

### Inputs
1. **Observed GPS Trajectory:** A chronological sequence of noisy geographical coordinates $\{p_1, p_2, \dots, p_n\}$.
2. **Underlying Road Network:** A structured spatial graph representing valid roads, intersections, and connectivity.

### Goal
Using both the observed trajectory and the topology of the road network, the system must **identify the most likely sequence of road segments traveled**, effectively reconstructing a clean, continuous, and meaningful route from noisy location data.