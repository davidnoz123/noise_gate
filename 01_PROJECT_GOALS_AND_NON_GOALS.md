# Project Goals and Non-Goals

## Goals

Build a reproducible test harness for comparing noise-gating methods using synthetic mixtures of clean voice and background noise.

The harness should support:

- repeatable sample discovery and selection
- cached source audio
- cached source listings/manifests
- deterministic scenario recipes
- on-the-fly audio mixture generation
- multiple gate implementations
- automatic scoring against ground truth
- failure plots and summaries
- later comparison against Vosk recognition outcomes

## Non-goals for the first milestone

Do not start by building:

- a polished GUI
- Android integration
- a database-backed asset system
- a final project folder architecture
- advanced ML training
- a general-purpose audio editor
- large-scale cloud processing
- permanent storage of all generated noisy WAVs

## Product context

The target use case is a speech-to-text app for a user who may speak at varying distances from an Android tablet. The noise gate is intended to help trigger and manage speech capture, especially for Vosk-style recognition.

The failure priorities are asymmetric:

1. Do not miss speech.
2. Do not clip the beginning of speech.
3. Do not close mid-utterance.
4. Avoid long false opens on background noise.
5. Avoid feeding too much silence/noise to the recognizer.

Opening slightly early is usually much less harmful than opening late.
Closing slightly late is usually less harmful than closing early.
