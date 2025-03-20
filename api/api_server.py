from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from .motion_generator import MotionGenerator
from .motion_to_glb import motion_to_glb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Motion Diffusion Model API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MotionRequest(BaseModel):
    text_prompt: str
    motion_length: float = 6.0
    num_samples: int = 1
    num_repetitions: int = 1
    output_format: str = "numpy"  # "numpy" or "glb"

class MotionResponse(BaseModel):
    motion_data: list = None
    glb_url: str = None
    success: bool
    error: str = None

# Initialize the motion generator
MODEL_PATH = "./save/humanml_trans_enc_512/model000750000.pt"
generator = None

# Create output directory for GLB files
GLB_OUTPUT_DIR = Path("outputs/glb")
GLB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global generator
    try:
        generator = MotionGenerator(MODEL_PATH)
        # Test imports on startup
        generator.test_imports()
        logger.info("Motion generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize motion generator: {str(e)}")
        raise

def convert_to_serializable(obj):
    """Convert numpy arrays and other types to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

@app.post("/generate_motion")
async def generate_motion(request: MotionRequest):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Request {request_id} received - Prompt: '{request.text_prompt}'")
    logger.info(f"Request {request_id} parameters: {request.dict()}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Request {request_id} - Created temp directory: {temp_dir}")
            
            # Generate motion using our wrapper
            motion_data = generator.generate_motion(
                text_prompt=request.text_prompt,
                output_dir=temp_dir,
                motion_length=request.motion_length,
                num_samples=request.num_samples,
                num_repetitions=request.num_repetitions
            )
            
            # Log motion data structure
            logger.info(f"Request {request_id} - Motion data type: {type(motion_data)}")
            if isinstance(motion_data, dict):
                logger.info(f"Request {request_id} - Motion data keys: {list(motion_data.keys())}")
            elif isinstance(motion_data, (list, tuple)):
                logger.info(f"Request {request_id} - Motion data length: {len(motion_data)}")
            if isinstance(motion_data, np.ndarray):
                logger.info(f"Request {request_id} - Motion data shape: {motion_data.shape}")
            
            if request.output_format.lower() == "glb":
                try:
                    # Generate unique filename for GLB
                    glb_filename = f"motion_{request_id}.glb"
                    glb_path = GLB_OUTPUT_DIR / glb_filename
                    
                    logger.info(f"Request {request_id} - Attempting GLB conversion to: {glb_path}")
                    
                    # Convert motion data to GLB
                    if isinstance(motion_data, dict) and 'motion' in motion_data:
                        motion_array = motion_data['motion']
                        logger.info(f"Request {request_id} - Motion array type: {type(motion_array)}")
                        if isinstance(motion_array, np.ndarray):
                            logger.info(f"Request {request_id} - Motion array shape: {motion_array.shape}")
                        logger.info(f"Request {request_id} - Motion array content type: {type(motion_array[0]) if motion_array is not None else 'None'}")
                    else:
                        raise ValueError("Motion data must be a dictionary containing 'motion' key")
                        
                    motion_to_glb(motion_array, str(glb_path))
                    
                    # Verify the file was created
                    if not glb_path.exists():
                        raise ValueError("GLB file was not created")
                    
                    logger.info(f"Request {request_id} - GLB file created successfully at: {glb_path}")
                    
                    response = MotionResponse(
                        glb_url=f"/glb/{glb_filename}",
                        success=True
                    )
                    logger.info(f"Request {request_id} - Sending GLB response: {response.dict()}")
                    return response
                except Exception as e:
                    error_msg = f"GLB conversion failed: {str(e)}"
                    logger.error(f"Request {request_id} - {error_msg}")
                    response = MotionResponse(
                        success=False,
                        error=error_msg
                    )
                    logger.info(f"Request {request_id} - Sending error response: {response.dict()}")
                    return response
            else:
                # Extract just the motion array for numpy output
                if isinstance(motion_data, dict) and 'motion' in motion_data:
                    motion_array = motion_data['motion']
                else:
                    motion_array = motion_data
                    
                # Convert the data to serializable format
                motion_data_list = convert_to_serializable(motion_array)
                
                response = MotionResponse(
                    motion_data=motion_data_list,  # Send just the motion array
                    success=True
                )
                logger.info(f"Request {request_id} - Sending numpy response structure: {str(type(motion_data_list))}, length: {len(motion_data_list) if isinstance(motion_data_list, list) else 'N/A'}")
                return response
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Request {request_id} - Error: {error_msg}")
        response = MotionResponse(
            motion_data=[],
            success=False,
            error=error_msg
        )
        logger.info(f"Request {request_id} - Sending error response: {response.dict()}")
        return response

@app.get("/glb/{filename}")
async def get_glb_file(filename: str):
    file_path = GLB_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="GLB file not found")
    return FileResponse(str(file_path), media_type="model/gltf-binary")

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 