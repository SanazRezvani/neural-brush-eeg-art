# Neural Brush

Real-time EEG-driven generative art using flow fields, particle systems, and neuro-reactive visual interaction.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![EEG](https://img.shields.io/badge/EEG-Generative_Art-purple)

---

## Overview

Neural Brush is an experimental neuro-art project that transforms EEG activity into evolving generative visuals in real time.

The system extracts EEG-derived features such as alpha, beta, theta, and blink activity from EEG recordings and maps them into a dynamic particle ecosystem driven by flow fields, trails, glow effects, and neuro-reactive colour dynamics.

Rather than functioning as a clinical brain-computer interface, Neural Brush explores EEG as a medium for artistic expression and human-computer co-creation.

---

## Features

- EEG-driven generative particle system
- Flow-field motion using noise-based vector fields
- Fading particle trails ("neural brushstrokes")
- Neuro-reactive colour dynamics
- Real-time animation in Streamlit
- Support for custom EEG CSV uploads
- Built-in demo EEG dataset

---

## EEG → Visual Mapping

| EEG Feature | Visual Behaviour |
|---|---|
| Alpha (8–12 Hz) | Calmness, smooth motion, cooler colours |
| Beta (13–30 Hz) | Energy, turbulence, warmer colours |
| Theta (4–7 Hz) | Dreamlike expansion, glow, atmosphere |
| Blink Events | Visual bursts and transient intensity changes |

---

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- SciPy

---

## Project Structure

```text
neural-brush-eeg-art/
├── app.py
├── requirements.txt
├── README.md
├── sample_data/
│   └── demo_eeg.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SanazRezvani/neural-brush-eeg-art.git
cd neural-brush-eeg-art
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## EEG Data

The application includes a small built-in demo EEG dataset for quick testing.

Users can also upload their own EEG CSV files through the Streamlit sidebar.

Expected EEG columns include:

```text
Time, C3, CZ, C4
```

---

## Artistic Direction

Neural Brush investigates the intersection of:

- Brain-computer interfaces (BCI)
- Generative art
- Computational creativity
- Neurofeedback
- Human-computer interaction
- Interactive media systems

The project explores how neural oscillations can shape immersive computational aesthetics and evolving visual environments.

---

## Future Extensions

Planned directions include:

- EEG-reactive audio synthesis
- Multimodal biosignal interaction
- Real-time EEG streaming
- Spatial EEG mapping
- AI-generated visual textures
- Projection-based installations
- VR/immersive environments

---

## Author

Sanaz Rezvani  
Data Scientist | BCI Researcher | Creative Technologist

---

## License

MIT License