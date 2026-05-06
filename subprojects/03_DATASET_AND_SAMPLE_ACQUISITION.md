# Dataset and Sample Acquisition

## Initial source types

Use two kinds of source audio:

1. Clean speech
2. Background noise

The first milestone can use a small subset. The harness should be designed so the corpus can grow later.

## Candidate clean speech sources

- LibriSpeech test-clean
- LibriTTS subset
- Mozilla Common Voice subset later
- locally recorded clean speech
- other public speech datasets later

For the first milestone, LibriSpeech-style clean speech is attractive because it is already segmented into utterance files.

## Candidate background noise sources

- ESC-50
- UrbanSound8K
- FSD50K/Freesound-style sources
- MS-SNSD noise folder
- locally recorded noise
- device/tablet handling noise
- TV/radio/background speech-like noise

Speech-like background noise should be treated as a special hard category.

Examples:

- cafe chatter
- TV speech
- radio speech
- people talking in another room
- kitchen/restaurant noise
- traffic
- fan/aircon
- rain
- keyboard typing
- table bumps
- tablet handling noise

## Archive datasets vs individual-file repositories

There are two cases.

### Single archive datasets

Some datasets are distributed as one archive. For these:

1. download the archive once
2. extract/index locally
3. select deterministic subsets locally

Do not try to overcomplicate partial download unless the dataset source supports it cleanly.

### Individual-file repositories

Some repositories expose file-level listings or APIs. For these:

1. fetch/cache listing
2. deterministically select subset from listing
3. download only selected files

## Licensing metadata

Each source item or dataset manifest should record licence/source information.

At minimum:

- dataset name
- source URL
- licence name
- licence URL when known
- download time
- file hash
- local path
- original remote ID or filename

Do not assume non-commercial datasets are safe for commercial redistribution. For the harness, they are useful for development/testing. The source audio should remain external test material unless licensing is reviewed.
