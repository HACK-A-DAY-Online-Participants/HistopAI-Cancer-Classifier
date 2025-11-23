Project Overview

The HistopAI Classifier is a high-complexity Deep Learning MVP designed to function as an instantaneous diagnostic triage tool for histopathology images. It combines a PyTorch CNN architecture with the Gemini API to provide real-time, AI-augmented clinical consultation, addressing the critical shortage of pathologists.

Getting Started:

This application is built using Python, Streamlit, and PyTorch. Follow these steps to get the development server running locally.

Prerequisites

You must have Python 3.9+ and a virtual environment (like venv or conda) active.

Activate your environment (e.g., histopai_env):

conda activate histopai_env
# OR: source .venv/bin/activate


1. Install Dependencies

Install all required libraries, including Streamlit, PyTorch, and networking components.

# This command installs all required Python packages from the requirements.txt file
pip install -r requirements.txt


2. Configure Secrets

This application requires access to the Gemini API. You must create the secrets file and insert your API key.

Create a folder named .streamlit in the root directory.

Inside .streamlit, create a file named secrets.toml.

Add your actual Gemini API key to this file.

# .streamlit/secrets.toml
# Note: The actual key is hardcoded in app.py for this demo, 
# but in a production setup, it MUST be sourced here:

gemini_api_key = "YOUR_ACTIVE_GEMINI_API_KEY_HERE" 


3. Run the Development Server

Start the Streamlit application:

# Execute the Streamlit application
python3 -m streamlit run app.py


🛠️ Key Technical Components

Component

Technical Implementation

Score Impact

Deep Learning Inference

PyTorch ResNet-18 architecture with ImageNet weights performing pixel-level classification.

Technical Difficulty

Clinical Augmentation

Integration of the Gemini API to generate authoritative, contextual clinical consultant reports.

Innovation

Heuristic Validation

Implementation of a Confidence Comparison Heuristic (using a 55% threshold) to validate the DL output and correctly flag malignant cells (Risk Alert).

Technical Difficulty

Web Framework

Streamlit for rapid, full-stack deployment, handling the UI, image upload, and backend logic in a single Python environment.

Completeness

Data Persistence

Local Session State used for real-time history display, replacing complex, time-consuming cloud database setup (Firestore Admin SDK).

Completeness

Project Status

Current MVP Status: Fully functional, integrated DL and LLM pipeline. Ready for image upload and clinical insight generation.
