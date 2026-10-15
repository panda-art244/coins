# CoinAtlas
### A Digital Numismatic Catalogue and Image Analysis Platform

CoinAtlas is a Python-based application for the documentation, exploration, and visual analysis of historical coins. The project combines digital cataloguing practices with image processing techniques to support the structured study and comparison of numismatic objects.

Designed as a lightweight research tool, CoinAtlas brings together descriptive metadata, visual documentation, and computational image analysis in a single interface.

## Research Objectives

The project explores how digital tools can support the systematic documentation and comparative analysis of historical coins. It focuses on three complementary areas:

- **Digital cataloguing** — organising numismatic objects through structured metadata, including provenance-related information, chronology, mint, material, weight, and bibliographic references.
- **Image processing** — applying computational techniques to enhance visual features and support the examination of coin imagery.
- **Visual comparison** — investigating image similarity methods as a preliminary aid to identifying visual correspondences between numismatic objects.

## Key Features

- Structured catalogue of historical coins and their metadata.
- Image-based documentation and individual object records.
- Catalogue management and coin entry functionality.
- Adaptive image binarisation and edge detection.
- Visual similarity comparison using Structural Similarity Index (SSIM) and colour histogram correlation.

## Technical Stack

| Component | Technology |
|---|---|
| Application interface | Streamlit |
| Data management | Pandas |
| Numerical operations | NumPy |
| Image processing | OpenCV |
| Image similarity analysis | scikit-image |
| Image handling | Pillow |

## Methodological Approach

CoinAtlas combines descriptive object data with computational image analysis. Image preprocessing methods can help reveal visual features, while similarity metrics provide quantitative comparisons between images.

These methods are intended as exploratory analytical aids rather than definitive identification systems. Similarity scores may be affected by image quality, lighting, orientation, scale, wear, and differences in photographic conditions. Results therefore require interpretation in conjunction with numismatic expertise and contextual evidence.

## Project Structure

```text
CoinAtlas/
├── app.py
├── data/
│   └── coins.csv
├── catalog/
│   └── images/
├── .gitignore
└── README.md
```

## Getting Started

### Requirements

- Python 3.10 or later
- pip

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone YOUR_REPOSITORY_URL
cd CoinAtlas
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install streamlit pandas numpy opencv-python-headless scikit-image pillow
```

### Launch

Run the application locally:

```bash
streamlit run app.py
```

Streamlit will provide a local URL through which the application can be accessed in a web browser.

## Scope and Limitations

CoinAtlas is a developing project. Image similarity measurements do not establish that two coins share the same die, origin, date, or historical attribution. Such conclusions require additional evidence and specialist assessment.

Future development may explore more advanced computer vision techniques, improved image normalisation, and methods for evaluating the reliability of visual comparisons.

## Project Status

In development.

## Licence

A licence has not yet been specified.