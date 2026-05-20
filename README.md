# Neural Brush

Real-time EEG-driven generative art using neuro-reactive visual interaction, flow fields, and particle systems.

## Explore the live interactive version of the application [here](https://neural-brush.streamlit.app).

The application allows users to experiment with EEG-driven generative visuals in real time by either using the built-in demo EEG dataset or uploading their own EEG CSV recordings. As neural oscillatory dynamics change over time, the visual environment continuously adapts through flow-field motion, particle behaviour, atmospheric glow, and neuro-reactive colour transformations.

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

![demo](output.gif)

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

After cloning the repository and installing dependencies, run the app:
```bash
streamlit run app.py
```

---

## EEG Data

The application includes a small built-in demo EEG dataset for quick testing. The EEG is chosen from the following dataset: 

[Guttmann-Flury, E., Sheng, X. & Zhu, X. Dataset combining EEG, eye-tracking, and high-speed video for ocular activity analysis across BCI paradigms. Sci Data 12, 587 (2025).](https://arxiv.org/pdf/2506.07488)

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

- Real-time EEG streaming
- EEG-reactive audio synthesis
- VR/immersive environments

---

## Author

Sanaz Rezvani  
Data Scientist | BCI Researcher | Creative Technologist

---

## License

MIT License
