import re
import os
from dotenv import load_dotenv
import streamlit as st
import requests
import base64
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field
from PIL import Image
import io
import logging
from langgraph.graph import StateGraph, END
import traceback
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting Streamlit app")

# Set random seeds for reproducibility
np.random.seed(42)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
logger.info(f"GROQ_API_KEY loaded: {'Yes' if GROQ_API_KEY else 'No'}")

# Groq API settings
GROQ_API_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEXT_MODEL = "deepseek-r1-distill-llama-70b"

# Define state types with Pydantic for type safety
class PatientInfo(BaseModel):
    age: int = Field(description="Patient age in years")
    symptoms: str = Field(description="Clinical symptoms and history")
    protein_type: str = Field(description="Type of protein targeted in IHC staining")

class IHCImage(BaseModel):
    base64: str = Field(description="Base64-encoded image data")
    raw_bytes: bytes = Field(description="Raw image bytes")

class IHCScore(BaseModel):
    score: str = Field(description="IHC score (0, 1+, 2+, or 3+)")
    intensity: Optional[str] = Field(default=None, description="Staining intensity (Negative, Weak, Moderate, Strong)")
    distribution: Optional[str] = Field(default=None, description="Staining distribution (Focal, Diffuse)")
    percentage: Optional[str] = Field(default=None, description="Percentage of positive cells (0-25%, 26-50%, 51-75%, 76-100%)")
    final_score: Optional[int] = Field(default=None, description="Numerical final score (0-7)")
    status: Optional[str] = Field(default=None, description="Staining status (Negative, Positive)")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the score")

class MalignancyAssessment(BaseModel):
    assessment: str = Field(description="Detailed malignancy assessment")

class TreatmentRecommendation(BaseModel):
    recommendation: str = Field(description="Treatment recommendation")

class IHCAnalysisState(BaseModel):
    patient_info: Optional[PatientInfo] = Field(default=None, description="Patient information")
    image: Optional[IHCImage] = Field(default=None, description="IHC image data")
    ihc_score: Optional[IHCScore] = Field(default=None, description="IHC scoring result")
    malignancy: Optional[MalignancyAssessment] = Field(default=None, description="Malignancy assessment")
    treatment: Optional[TreatmentRecommendation] = Field(default=None, description="Treatment recommendation")
    error: Optional[Union[str, List[str]]] = Field(default=None, description="Error messages if any")

# API interaction functions
def call_vision_model(prompt: str, image_base64: Optional[str] = None, temperature: float = 0.0) -> str:
    """Call the Groq API for vision tasks"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    if image_base64:
        image_url = f"data:image/jpeg;base64,{image_base64}"
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 512,
        "seed": 42
    }

    try:
        response = requests.post(GROQ_API_BASE, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        return f"Error: {str(e)}"

def call_text_model(prompt: str, temperature: float = 0.0) -> str:
    """Call the Groq API for text-only tasks"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
        "temperature": temperature,
        "max_tokens": 512,
        "seed": 42
    }

    try:
        response = requests.post(GROQ_API_BASE, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        return f"Error: {str(e)}"

# Workflow processing nodes
def process_ihc_scoring(state: IHCAnalysisState) -> Dict[str, Any]:
    """Node for IHC scoring analysis"""
    try:
        if not state.image or not state.patient_info:
            raise ValueError("Missing image or patient information for analysis")

        ihc_prompt = f"""Analyze the IHC-stained image for {state.patient_info.protein_type} based on visible staining. Ignore any external metadata.

        Instructions:
        - Evaluate brown staining (DAB) for:
          - Intensity: Negative (0), Weak (1), Moderate (2), Strong (3).
          - Distribution: Focal (<50% cells) or Diffuse (≥50% cells).
          - Percentage: 0-25%, 26-50%, 51-75%, 76-100%.
        - Calculate score:
          - Intensity score: 0-3.
          - Percentage score: 0 (0%), 1 (1-25%), 2 (26-50%), 3 (51-75%), 4 (76-100%).
          - Final score: Intensity + Percentage (0-7, capped at 7).
          - Categorical score: 0 (score=0), 1+ (score=1-2), 2+ (score=3-4), 3+ (score=5-7).
          - Status: Negative (score<2), Positive (score≥2).
        - If ambiguous, favor moderate-to-strong intensity only if clearly visible.

        Output format:
        **Intensity**: [Negative/Weak/Moderate/Strong] (Score: [0-3])
        **Distribution**: [Focal/Diffuse]
        **Percentage**: [0-25%/26-50%/51-75%/76-100%] (Score: [0-4])
        **Final Score**: [0-7]
        **Categorical Score**: [0/1+/2+/3+]
        **Status**: [Negative/Positive]
        **Reasoning**: [Explain scoring]
        """

        logger.info("Sending IHC scoring prompt to Groq API")
        ihc_response = call_vision_model(ihc_prompt, state.image.base64, temperature=0.0)
        logger.info(f"Full IHC scoring response: {ihc_response}")

        # Parse response
        intensity = distribution = percentage = reasoning = None
        intensity_score = percentage_score = final_score = None
        categorical_score = status = None

        intensity_match = re.search(r"\*\*Intensity\*\*: (\w+) \(Score: (\d)\)", ihc_response)
        distribution_match = re.search(r"\*\*Distribution\*\*: (Focal|Diffuse)", ihc_response)
        percentage_match = re.search(r"\*\*Percentage\*\*: (\d+-\d+%|\d+%) \(Score: (\d)\)", ihc_response)
        final_score_match = re.search(r"\*\*Final Score\*\*: (\d)", ihc_response)
        categorical_score_match = re.search(r"\*\*Categorical Score\*\*: (0|1\+|2\+|3\+)", ihc_response)
        status_match = re.search(r"\*\*Status\*\*: (Negative|Positive)", ihc_response)
        reasoning_match = re.search(r"\*\*Reasoning\*\*: (.*?)(?:\n|$)", ihc_response, re.DOTALL)

        if intensity_match:
            intensity = intensity_match.group(1)
            intensity_score = int(intensity_match.group(2))
        if distribution_match:
            distribution = distribution_match.group(1)
        if percentage_match:
            percentage = percentage_match.group(1)
            percentage_score = int(percentage_match.group(2))
        if final_score_match:
            final_score = int(final_score_match.group(1))
        if categorical_score_match:
            categorical_score = categorical_score_match.group(1)
        if status_match:
            status = status_match.group(1)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Validate scoring
        if all([intensity_score, percentage_score, final_score, categorical_score, status]):
            calculated_score = min(intensity_score + percentage_score, 7)
            if calculated_score != final_score:
                logger.warning(f"Score mismatch: Calculated {calculated_score}, Reported {final_score}")
                final_score = calculated_score

            expected_categorical = "0" if final_score == 0 else "1+" if final_score <= 2 else "2+" if final_score <= 4 else "3+"
            if expected_categorical != categorical_score:
                logger.warning(f"Categorical score mismatch: Expected {expected_categorical}, Reported {categorical_score}")
                categorical_score = expected_categorical

            expected_status = "Negative" if final_score < 2 else "Positive"
            if expected_status != status:
                logger.warning(f"Status mismatch: Expected {expected_status}, Reported {status}")
                status = expected_status
        else:
            logger.error("Incomplete IHC scoring response")
            if intensity_score is not None and percentage_score is not None:
                final_score = min(intensity_score + percentage_score, 7)
                categorical_score = "0" if final_score == 0 else "1+" if final_score <= 2 else "2+" if final_score <= 4 else "3+"
                status = "Negative" if final_score < 2 else "Positive"
            else:
                categorical_score = "0"
                intensity = "Negative"
                distribution = "None"
                percentage = "0%"
                final_score = 0
                status = "Negative"
                reasoning = "No valid staining data detected"

        logger.info(f"Validated IHC Score: Final={final_score}, Categorical={categorical_score}, Status={status}")

        return {
            "ihc_score": IHCScore(
                score=categorical_score,
                intensity=intensity,
                distribution=distribution,
                percentage=percentage,
                final_score=final_score,
                status=status,
                reasoning=reasoning
            )
        }
    except Exception as e:
        error_msg = f"Error in IHC scoring: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def process_malignancy_detection(state: IHCAnalysisState) -> Dict[str, Any]:
    """Node for malignancy detection"""
    try:
        if not state.image or not state.patient_info:
            raise ValueError("Missing image or patient information for analysis")

        malignancy_prompt = f"""Analyze the IHC-stained cervical tissue image for {state.patient_info.protein_type} based on visible features. 
        Ignore external metadata. Take any symptom data provided in {state.patient_info.symptoms} into consideration. 

        Evaluate malignancy signs:
        1. Nuclear pleomorphism
        2. Mitotic figures
        3. Tissue disorganization
        4. Tissue invasion
        5. Cellular overcrowding

        Evaluate normal tissue signs:
        1. Uniform nuclei
        2. Organized layers
        3. Intact basement membrane
        4. Normal cell density

        Classification Rule:
        - CANCER: Any malignancy signs present.
        - NORMAL: No malignancy signs detected.

        Output format:
        **Classification**: [CANCER/NORMAL]
        **Evidence**:
        - Malignancy signs: [list or "None"]
        - Normal signs: [list or "None"]
        - Reasoning: [explain classification]
        """

        logger.info("Sending malignancy detection prompt to Groq API")
        malignancy_response = call_vision_model(malignancy_prompt, state.image.base64, temperature=0.0)
        logger.info(f"Malignancy detection response: {malignancy_response[:100]}...")

        malignancy_response_upper = malignancy_response.upper()
        classification = "CANCER" if "**CLASSIFICATION**: CANCER" in malignancy_response_upper else "NORMAL"

        formatted_response = f"CLASSIFICATION: {classification}\n\n{malignancy_response}"

        return {"malignancy": MalignancyAssessment(assessment=formatted_response)}
    except Exception as e:
        error_msg = f"Error in malignancy detection: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def generate_treatment_recommendation(state: IHCAnalysisState) -> Dict[str, Any]:
    """Node for generating treatment recommendations"""
    try:
        if not state.ihc_score or not state.malignancy or not state.patient_info:
            raise ValueError("Missing required information for treatment recommendation")

        recommendation_prompt = f"""Based on the following, recommend a treatment plan:

        Patient:
        - Age: {state.patient_info.age}
        - Symptoms: {state.patient_info.symptoms}

        Findings:
        - Protein: {state.patient_info.protein_type}
        - IHC Score: {state.ihc_score.score}
        - Malignancy: {state.malignancy.assessment}

        Provide a concise recommendation with rationale."""

        recommendation_response = call_text_model(recommendation_prompt)

        return {"treatment": TreatmentRecommendation(recommendation=recommendation_response)}
    except Exception as e:
        error_msg = f"Error in treatment recommendation: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def router(state: IHCAnalysisState) -> Dict[str, Any]:
    """Determine the next step"""
    return state.model_dump()

def validate_image(file):
    """Validate and process the uploaded image"""
    try:
        img = Image.open(file)
        img.verify()
        img = Image.open(file)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG")
        return img_buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

def create_ihc_analysis_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(IHCAnalysisState)

    workflow.add_node("ihc_scoring", process_ihc_scoring)
    workflow.add_node("malignancy_detection", process_malignancy_detection)
    workflow.add_node("treatment_recommendation", generate_treatment_recommendation)
    workflow.add_node("router", router)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: (
            "treatment_recommendation"
            if state.ihc_score is not None and state.malignancy is not None
            else [
                node for node in [
                    "ihc_scoring" if state.ihc_score is None else None,
                    "malignancy_detection" if state.malignancy is None else None
                ] if node is not None
            ]
        ),
        {
            "treatment_recommendation": "treatment_recommendation",
            "ihc_scoring": "ihc_scoring",
            "malignancy_detection": "malignancy_detection"
        }
    )

    workflow.add_edge("ihc_scoring", "router")
    workflow.add_edge("malignancy_detection", "router")
    workflow.add_edge("treatment_recommendation", END)

    return workflow.compile()

def run_ihc_analysis_workflow(image_bytes: bytes, patient_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run the LangGraph workflow"""
    try:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        patient = PatientInfo(
            age=patient_info.get("age", 0),
            symptoms=patient_info.get("symptoms", ""),
            protein_type=patient_info.get("protein_type", "")
        )

        initial_state = IHCAnalysisState(
            patient_info=patient,
            image=IHCImage(base64=image_base64, raw_bytes=image_bytes),
            ihc_score=None,
            malignancy=None,
            treatment=None,
            error=None
        )

        workflow = create_ihc_analysis_workflow()
        logger.info("Starting workflow execution")
        final_state_dict = workflow.invoke(initial_state.model_dump())
        logger.info("Workflow execution completed")

        if final_state_dict.get("error"):
            error_data = final_state_dict.get("error")
            error_message = "; ".join(error_data) if isinstance(error_data, list) else str(error_data)
            return {"error": error_message}

        final_state = IHCAnalysisState.model_validate(final_state_dict)

        return {
            "protein_type": final_state.patient_info.protein_type if final_state.patient_info else None,
            "ihc_score": final_state.ihc_score if final_state.ihc_score else None,
            "malignant_features": final_state.malignancy.assessment if final_state.malignancy else None,
            "recommendation": final_state.treatment.recommendation if final_state.treatment else None
        }

    except Exception as e:
        error_msg = f"Error in workflow execution: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"error": error_msg}

# Common IHC protein markers
COMMON_IHC_MARKERS = [
    "FKBP6", "INTS1", "ZNF516", "GGTLA4", "HER2", "ER", "PR", "Ki-67", "p53", "CD20", "CD3", "CD4", "CD8", "CD45",
    "PD-L1", "EGFR", "ALK", "ROS1", "BRAF", "KRAS", "CD31", "Cytokeratin",
    "E-cadherin", "MMR proteins", "Other"
]

def show_workflow_diagram():
    """Render the workflow diagram and protein marker info in the sidebar"""
    logger.info("Starting show_workflow_diagram")
    try:
        st.sidebar.subheader("IHC Analysis Workflow")
        st.sidebar.markdown("""
        ```mermaid
        graph TD
            A[Router] --> B[IHC Scoring]
            A --> C[Malignancy Detection]
            B --> A
            C --> A
            A --> E[Treatment Recommendation]
            E --> F[End]
        ```
        """)
        st.sidebar.subheader("About IHC Protein Markers")
        st.sidebar.markdown("""
        Common protein markers and their significance:
        - **HER2**: Human epidermal growth factor receptor 2 (breast, gastric cancers)
        - **ER/PR**: Hormone receptors (breast cancer)
        - **Ki-67**: Proliferation marker (multiple cancers)
        - **PD-L1**: Immune checkpoint (immunotherapy response)
        - **p53**: Tumor suppressor (multiple cancers)
        """)
        logger.info("show_workflow_diagram rendered successfully")
    except Exception as e:
        logger.error(f"Error in show_workflow_diagram: {str(e)}\n{traceback.format_exc()}")
        st.sidebar.error(f"Failed to render diagram: {str(e)}")

# Streamlit UI
def main():
    logger.info("Main function called")
    try:
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found. Please set it in the .env file.")
            logger.error("Missing GROQ_API_KEY")
            return

        st.title("IHC Analysis & Treatment Recommendation")
        logger.info("Title rendered")
        st.write("Upload an immunohistochemistry (IHC) stained tissue image for analysis")
        logger.info("Initial UI elements rendered")

        with st.expander("Patient Information", expanded=True):
            age = st.number_input("Patient Age", min_value=0, max_value=120, value=50)
            symptoms = st.text_area("Clinical Symptoms and History")
            protein_selection = st.selectbox("Select Protein Target", options=COMMON_IHC_MARKERS)
            protein_type = protein_selection
            if protein_selection == "Other":
                protein_type = st.text_input("Specify Protein Target", "")
            logger.info("Patient information form rendered")

        uploaded_file = st.file_uploader("Upload IHC Image", type=["jpg", "jpeg", "png"])
        logger.info("File uploader rendered")

        if st.button("Run Analysis") and uploaded_file:
            logger.info("Run Analysis button clicked")
            try:
                with st.spinner("Processing image..."):
                    ihc_image = validate_image(uploaded_file)
                    patient_info = {
                        "age": age,
                        "symptoms": symptoms,
                        "protein_type": protein_type
                    }
                    logger.info("Starting workflow execution")
                    results = run_ihc_analysis_workflow(ihc_image, patient_info)

                    if "error" in results:
                        st.error(f"Analysis failed: {results['error']}")
                        logger.error(f"Workflow error: {results['error']}")
                        return

                    # Display results
                    st.image(ihc_image, caption=f"Uploaded IHC Image ({results['protein_type']})", use_column_width=True)
                    with st.container():
                        st.subheader("Analysis Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Protein Target", results['protein_type'])
                        with col2:
                            st.metric("IHC Score", results['ihc_score'].score if results['ihc_score'] else "N/A")
                        
                        st.subheader("IHC Scoring Details")
                        if results['ihc_score']:
                            st.info(
                                f"**Intensity**: {results['ihc_score'].intensity or 'N/A'}\n"
                                f"**Distribution**: {results['ihc_score'].distribution or 'N/A'}\n"
                                f"**Percentage**: {results['ihc_score'].percentage or 'N/A'}\n"
                                f"**Final Score**: {results['ihc_score'].final_score if results['ihc_score'].final_score is not None else 'N/A'}\n"
                                f"**Status**: {results['ihc_score'].status or 'N/A'}\n"
                                f"**Reasoning**: {results['ihc_score'].reasoning or 'No reasoning provided'}"
                            )
                        else:
                            st.info("No IHC scoring details available")
                        
                        st.subheader("Malignancy Assessment")
                        st.info(results['malignant_features'] or "No malignancy assessment available")
                        
                        st.subheader("Treatment Recommendation")
                        st.warning(results['recommendation'] or "No treatment recommendation available")
                        
                        st.caption("Note: This is an AI-generated recommendation. Always consult with a qualified healthcare professional.")
                        logger.info("Results displayed")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Error in analysis: {str(e)}\n{traceback.format_exc()}")
        elif not uploaded_file:
            st.warning("Please upload an image to analyze.")
            logger.info("No file uploaded")
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        logger.error(f"Main function error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("Entering __main__ block")
    try:
        show_workflow_diagram()
        logger.info("show_workflow_diagram completed")
        main()
        logger.info("main completed")
    except Exception as e:
        logger.error(f"Error in __main__: {str(e)}\n{traceback.format_exc()}")