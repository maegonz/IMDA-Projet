# NFL Prediction : Predict Player Movement During the Downfield Pass

## Overview

This project is realised with the friend [**blajox**](github.com/blajox), during our last year of master's degree. For this project we choose to achieve the problematic brought by the NFL league through their Kaggle competition. This competition focuses on the "crown jewel of American sports", the downfield pass and the critical uncertainty that keeps audiences engaged: will it be a touchdown, an interception, or an incomplete pass?

The objective is to leverage Next Gen Stats data to **predict player movement** during the entire phase when the football is in the air. This analysis is crucial for helping the **NFL** better understand player trajectories and movement patterns during contested catch situations.

---

### The Prediction Challenge

Participants are tasked with building models to predict the precise **location and movement of key players (the targeted receiver and converging defenders)** for every frame, starting the moment the quarterback releases the ball and ending when the ball lands.

#### **Input Information Provided (Pre-Pass Data):**

The data provided for prediction includes information available right up to the moment of ball release:

* **Pre-Pass Tracking Data:** Detailed NGS tracking data leading up to the moment the quarterback releases the ball.
* **Targeted Player:** Identification of the offensive player (the targeted receiver) who is the intended recipient of the pass.
* **Ball Landing Location:** The final $(\text{x}, \text{y})$ coordinates where the pass is expected to land.



#### **Data Details:**

* **Tracking Frequency:** The NFL tracking data is recorded at **10 frames per second (FPS)**.
* **Prediction Granularity:** If a pass is in the air for $T$ seconds, participants must predict **$10 \times T$ frames** of location data for each player.
* **Excluded Plays:** To ensure focus on relevant downfield pass analysis, the competition data **excludes** the following types of plays: quick passes (duration less than 0.5 seconds), deflected passes and throwaway passes.

#### **Prediction Task:**

Generate models that output predicted movement (location coordinates) for each relevant player across all frames while the **ball is traveling in the air**. The ultimate goal is to generate outputs that **most closely match the actual eventual player movement**.


---

### Getting Started

1.  **Analyze Trajectories:** Use the pre-pass movement data to determine initial velocities and intentions.
2.  **Model Player Intent:** Integrate the knowledge of the Targeted Player and Ball Landing Location, as these heavily influence player movement during the pass.
3.  **Time-Series Modeling:** Develop robust models capable of forecasting multi-step, multi-player time-series data accurately.


## Results

At this stage nothing has been developed yet.

-----
*Note: This Big Data Bowl 2026 has two competitions. This is the Prediction competition. Learn more about the Analytics competition [here](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics).*

## Author

Project created by **Antony Manuel** and **Florian Lemiere**, as part of the **IMDA** course.

IMT Nord Europe — **2025–2026**
