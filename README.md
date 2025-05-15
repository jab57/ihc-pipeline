# IHC Analysis Streamlit App

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

A Streamlit-based application for analyzing Immunohistochemistry (IHC) stained images to provide automated scoring, malignancy detection, and treatment recommendations. Powered by the Groq API and orchestrated via a LangGraph workflow, this tool supports pathologists in evaluating protein markers and assessing cervical tissue.

## ✨ Features
- **IHC Scoring**: Evaluates staining intensity, distribution, and percentage to assign scores (0, 1+, 2+, 3+).
- **Malignancy Detection**: Identifies malignancy signs like nuclear pleomorphism and tissue invasion.
- **Treatment Recommendations**: Generates tailored treatment plans based on patient data and analysis.
- **User-Friendly Interface**: Upload images and input patient info via a clean Streamlit UI.
- **Robust Workflow**: Uses LangGraph for reliable, state-based analysis.
- **Type-Safe**: Employs Pydantic for data validation.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Groq API key](https://x.ai/api)
- Git and a GitHub account

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ihc-analysis-app.git
   cd ihc-analysis-app