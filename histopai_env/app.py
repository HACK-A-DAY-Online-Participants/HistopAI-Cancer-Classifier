import streamlit as st
import random
import time
import requests
import json
from PIL import Image
from datetime import datetime
# --- DEEP LEARNING IMPORTS ---
import torch
import torch.nn as nn
from torchvision import models, transforms



APP_ID = st.secrets.get('app_id', 'histopai-default')
GEMINI_API_KEY = "AIzaSyC7oAUMpF8dk_4bGoG-ULKRPqhuSsGK-4o" 
 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=" + GEMINI_API_KEY

NUM_CLASSES = 2
CLASS_NAMES = ["Benign (Normal)", "Malignant (Tumor)"]
DEVICE = torch.device("cpu") 


if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
    
db = None 

@st.cache_resource
def load_deep_learning_model():
    """Loads a pre-trained ResNet-18 model and adapts it for classification."""
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Deep Learning Model Loading Error. Check installation: {e}")
        return None

DL_MODEL = load_deep_learning_model()
TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_dl_model(image_data):
    """Performs inference based purely on deep learning output comparison."""
    if DL_MODEL is None: return "Model Error", 50.0 
    input_tensor = TRANSFORMS(image_data).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = DL_MODEL(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    benign_score = probabilities[0].item() * 100
    malignant_score = probabilities[1].item() * 100
    
    MALIGNANT_CONFIDENCE_THRESHOLD = 55.0
    
    if malignant_score >= MALIGNANT_CONFIDENCE_THRESHOLD:
        prediction = CLASS_NAMES[1]
        final_confidence = malignant_score
    else:
        prediction = CLASS_NAMES[0]
        final_confidence = benign_score if benign_score > malignant_score else malignant_score
    return prediction, final_confidence
USER_ID = "anonymous_hacker"

def add_analysis_result(data):
    """Stores a new analysis result in Streamlit Session State (local memory)."""
    data['id'] = str(time.time()) 
    st.session_state.analysis_results.append(data)
    st.rerun() 
    return True

def exponential_backoff_fetch(url, options, retries=3):
    for i in range(retries):
        try:
            response = requests.post(url, **options)
            if not response.ok:
                if response.status_code == 429 and i < retries - 1:
                    delay = (2 ** i) * 1 + random.random() * 1
                    time.sleep(delay)
                    continue
                response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if i == retries - 1:
                raise e
            delay = (2 ** i) * 1 + random.random() * 1
            time.sleep(delay)

def make_gemini_api_call(user_query):
    """Calls the Gemini API to get clinical insight."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "DUMMY_GEMINI_KEY":
        return "Clinical Insight unavailable: GEMINI_API_KEY is missing. Add your real key to secrets.toml."

    system_prompt = "You are an AI Clinical Consultant specializing in histopathology. Given a binary tissue classification and confidence score, provide a brief, authoritative explanation of the result and suggest the immediate next clinical or laboratory step. Conclude with a very brief summary of the result. Keep the response to a single, concise paragraph."

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    options = {
        'headers': {'Content-Type': 'application/json'},
        'data': json.dumps(payload)
    }
    
    response = exponential_backoff_fetch(GEMINI_API_URL, options)
    result = response.json()
    
    if 'error' in result:
         return f"Error from Gemini: {result['error'].get('message', 'Check quota/billing status.')}"
         
    candidate = result.get('candidates', [{}])[0]
    return candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Error: Gemini response was empty.')

def render_results(results):
    """Renders the analysis history using Streamlit columns and containers with improved UI."""
    if not results:
        st.info("No analysis results found yet. Upload an image to start!")
        return

    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    for result in results:
        is_malignant = 'Malignant' in result['prediction']
        color = "#EF4444" if is_malignant else "#10B981"
        status_text = "RISK ALERT" if is_malignant else "NORMAL"
        
        with st.container(border=True):
            col_pred, col_status = st.columns([2, 1])

            with col_pred:
                st.markdown(f"**{result.get('imageName', 'Unknown File')}**")
                st.markdown(f"<span style='font-size: 32px; font-weight: bold; color: {color}'>{result['prediction']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 14px; color: gray;'>Confidence: **{result['confidence']:.2f}%** | Time: {result['timestamp'].strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)
            
            with col_status:
                st.markdown(f"""
                    <div style='background-color: {'#fee2e2' if is_malignant else '#dcfce7'}; 
                                border: 1px solid {'#fca5a5' if is_malignant else '#a7f3d0'};
                                padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px;'>
                        <span style='font-size: 16px; font-weight: bold; color: {'#991b1b' if is_malignant else '#065f46'};'>
                            {status_text}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            col_insight, col_placeholder = st.columns([1, 2])
            
            with col_insight:
                existing_insight = result.get('clinical_insight')
                
                if st.button("✨ Get Clinical Insight (Gemini)", key=f"insight_btn_{result['id']}"):
                    with st.spinner("Generating clinical insight..."):
                        query = f"The histopathology image patch was classified as {result['prediction']} with {result['confidence']:.2f}% confidence. Provide a clinical summary and immediate next step."
                        insight_text = make_gemini_api_call(query)
                      
                        for item in st.session_state.analysis_results:
                            if item['id'] == result['id']:
                                item['clinical_insight'] = insight_text
                                break
                        st.rerun() 
            
    
            if 'clinical_insight' in result:
                st.markdown("---")
                st.markdown(f"**🔬 Clinical Consultant Report:**")
                st.info(result['clinical_insight'])
            elif existing_insight is None and GEMINI_API_KEY == "AIzaSyC7oAUMpF8dk_4bGoG-ULKRPqhuSsGK-4o":
                 st.markdown("---")
                 st.error("Clinical Insight not generated. Check console for Gemini error (likely billing or quota).")




st.title("🔬 HistopAI Classifier (Deep Learning MVP)")
st.caption(f"Status: Local State Active (Simulated User ID: {USER_ID}) | Technical Stack: Streamlit, PyTorch, Gemini API")

if DL_MODEL is None:
    st.error("Application is not fully initialized due to errors above. Cannot proceed.")
    st.stop()


col_upload, col_history = st.columns([1, 2])

with col_upload:
    st.header("Upload Image Patch")
    
    uploaded_file = st.file_uploader("Choose a Histopathology Patch (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.image(image, caption='Uploaded Patch Preview', width="stretch")
        
        file_name = uploaded_file.name
        
        if st.button("Run Deep Learning Analysis & Store"):
            with st.spinner(f"Running PyTorch inference..."):
                
                
                prediction, confidence = predict_dl_model(image)
                # ---------------------------------------------
                
                result_data = {
                    'userId': USER_ID,
                    'imageName': file_name,
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': datetime.now(), 
                    'imagePlaceholderUrl': f"https://placehold.co/50x50/{'FF4444' if 'Malignant' in prediction else '4CAF50'}/FFFFFF?text=DL"
                }

                if add_analysis_result(result_data):
                    st.success(f"Analysis Complete (DL): {prediction} at {confidence:.2f}% confidence. Result saved!")
    
    else:
        st.markdown("_Upload a patch to run the classification model._")



with col_history:
    st.header("Analysis History (Local State)")
    
    results_data = st.session_state.analysis_results
        
    render_results(results_data)
