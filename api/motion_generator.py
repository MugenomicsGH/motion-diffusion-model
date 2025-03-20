import os
import sys
import subprocess
import logging
import numpy as np
from pathlib import Path

class MotionGenerator:
    def __init__(self, model_path, conda_prefix=None):
        self.model_path = os.path.abspath(model_path)
        self.conda_prefix = conda_prefix or os.environ.get('CONDA_PREFIX')
        if not self.conda_prefix:
            raise RuntimeError("No conda environment detected")
        
        self.logger = logging.getLogger(__name__)
        self._setup_environment()
        
    def _setup_environment(self):
        """Set up the environment variables needed for motion generation"""
        self.env = os.environ.copy()
        self.current_dir = os.getcwd()
        
        # Set up Python path
        python_path = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = f"{self.current_dir}{os.pathsep}{python_path}" if python_path else self.current_dir
        
        # Set up PATH
        if os.name == 'nt':
            self.env["PATH"] = f"{self.conda_prefix};{self.conda_prefix}\\Library\\bin;{self.conda_prefix}\\Scripts;{self.env['PATH']}"
            self.python_exe = os.path.join(self.conda_prefix, 'python.exe')
        else:
            self.env["PATH"] = f"{self.conda_prefix}/bin:{self.env['PATH']}"
            self.python_exe = os.path.join(self.conda_prefix, 'bin', 'python')
        
        # Additional environment variables
        self.env["CONDA_PREFIX"] = self.conda_prefix
        self.env["PYTHONUNBUFFERED"] = "1"
        self.env["CUDA_VISIBLE_DEVICES"] = "0"
        self.env["PYTHONIOENCODING"] = "utf-8"
        # Add environment variable to skip video generation
        self.env["MDM_SKIP_VIDEO"] = "1"
        # Force CLIP to use float32
        self.env["FORCE_CLIP_FP32"] = "1"
    
    def _run_subprocess(self, cmd, **kwargs):
        """Run a subprocess with proper encoding handling"""
        try:
            # Use Popen instead of run to handle output encoding properly
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                cwd=self.current_dir,
                encoding='utf-8',
                errors='replace',
                **kwargs
            )
            
            stdout, stderr = process.communicate(timeout=300)
            return process.returncode, stdout, stderr
            
        except subprocess.TimeoutExpired as e:
            process.kill()
            raise RuntimeError("Process timed out after 300 seconds")
        except Exception as e:
            raise RuntimeError(f"Subprocess error: {str(e)}")
    
    def test_imports(self):
        """Test if all required modules can be imported"""
        self.logger.info("Testing dependencies import...")
        
        # Test dependencies
        deps_cmd = [
            self.python_exe,
            "-c",
            "import torch; import numpy; import clip; print('Dependencies imported successfully')"
        ]
        
        returncode, stdout, stderr = self._run_subprocess(deps_cmd)
        if returncode != 0:
            raise RuntimeError(f"Failed to import dependencies: {stderr}")
        
        # Test motion generation module
        test_cmd = [
            self.python_exe,
            "-c",
            "import sys; sys.path.insert(0, '.'); import sample.generate; print('Module import successful')"
        ]
        
        returncode, stdout, stderr = self._run_subprocess(test_cmd)
        if returncode != 0:
            raise RuntimeError(f"Failed to import generation module: {stderr}")
        
        return True
    
    def generate_motion(self, text_prompt, output_dir, motion_length=6.0, num_samples=1, num_repetitions=1, seed=42):
        """Generate motion from text using the original repo's functionality"""
        self.logger.info(f"Generating motion for prompt: {text_prompt}")
        
        # Ensure output directory exists
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct command using command-line arguments
        cmd = [
            self.python_exe,
            "-m", "sample.generate",
            "--model_path", self.model_path,
            "--text_prompt", text_prompt,
            "--motion_length", str(motion_length),
            "--num_samples", str(num_samples),
            "--num_repetitions", str(num_repetitions),
            "--output_dir", output_dir,
            "--dataset", "humanml",
            "--seed", str(seed),
            "--device", "-1"
        ]
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            returncode, stdout, stderr = self._run_subprocess(cmd)
            
            if returncode != 0:
                raise RuntimeError(f"Generation failed:\nStdout: {stdout}\nStderr: {stderr}")
            
            # Load and return the results
            result_file = Path(output_dir) / "results.npy"
            if not result_file.exists():
                raise RuntimeError("Results file not found")
            
            try:
                motion_data = np.load(str(result_file), allow_pickle=True)
                if isinstance(motion_data, np.ndarray):
                    motion_data = motion_data.item()  # Convert from 0d array if needed
                return motion_data
            except Exception as e:
                raise RuntimeError(f"Failed to load motion data: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Motion generation failed: {str(e)}") 