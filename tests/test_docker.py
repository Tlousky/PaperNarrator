"""
Task 14: Docker Support Tests

Validates Dockerfiles, docker-compose.yml, and .dockerignore configuration.
"""
import pytest
import yaml
import os
from pathlib import Path


class TestDockerConfiguration:
    """Test Docker configuration files."""
    
    @pytest.fixture
    def docker_dir(self):
        """Return the docker directory path."""
        return Path(__file__).parent.parent / "docker"
    
    @pytest.fixture
    def docker_compose(self, docker_dir):
        """Load docker-compose.yml."""
        with open(docker_dir / "docker-compose.yml", "r") as f:
            return yaml.safe_load(f)
    
    def test_docker_files_exist(self, docker_dir):
        """Test that all required Docker files exist."""
        required_files = [
            "Dockerfile.cpu",
            "Dockerfile.gpu",
            "docker-compose.yml",
            ".dockerignore"
        ]
        for file in required_files:
            assert (docker_dir / file).exists(), f"Missing {file}"
    
    def test_docker_compose_syntax(self, docker_compose):
        """Test docker-compose.yml has valid YAML syntax."""
        assert docker_compose is not None
        assert "services" in docker_compose
    
    def test_docker_compose_services(self, docker_compose):
        """Test docker-compose.yml has required services."""
        services = docker_compose.get("services", {})
        assert "paper-narrator-cpu" in services
        assert "paper-narrator-gpu" in services
        assert "ollama" in services
    
    def test_cpu_service_configuration(self, docker_compose):
        """Test CPU service configuration."""
        service = docker_compose["services"]["paper-narrator-cpu"]
        
        # Check build configuration
        assert service["build"]["dockerfile"] == "docker/Dockerfile.cpu"
        
        # Check port mapping
        assert "7860:7860" in service["ports"]
        
        # Check environment variables (can be list or dict)
        env = service["environment"]
        if isinstance(env, dict):
            assert env["LLM_PROVIDER"] == "openai"
        elif isinstance(env, list):
            assert any("LLM_PROVIDER" in e for e in env)
        
        # Check profile
        assert "cpu" in service["profiles"]
    
    def test_gpu_service_configuration(self, docker_compose):
        """Test GPU service configuration."""
        service = docker_compose["services"]["paper-narrator-gpu"]
        
        # Check build configuration
        assert service["build"]["dockerfile"] == "docker/Dockerfile.gpu"
        
        # Check GPU deployment configuration
        assert "deploy" in service
        assert "resources" in service["deploy"]
        assert "reservations" in service["deploy"]["resources"]
        devices_list = service["deploy"]["resources"]["reservations"]["devices"]
        assert isinstance(devices_list, list)
        devices = devices_list[0]  # First device in the list
        assert devices["driver"] == "nvidia"
        assert devices["count"] == 1
        
        # Check profile
        assert "gpu" in service["profiles"]
    
    def test_ollama_service_configuration(self, docker_compose):
        """Test Ollama service configuration."""
        service = docker_compose["services"]["ollama"]
        
        # Check image
        assert "ollama" in service["image"]
        
        # Check port mapping
        assert "11434:11434" in service["ports"]
        
        # Check profile
        assert "ollama" in service["profiles"]
    
    def test_dockerignore_excludes_large_files(self, docker_dir):
        """Test .dockerignore excludes large and unnecessary files."""
        with open(docker_dir / ".dockerignore", "r") as f:
            content = f.read()
        
        # Check that models directory is excluded
        assert "models/" in content or "models" in content
        
        # Check that Python cache is excluded
        assert "__pycache__" in content
        
        # Check that venv is excluded
        assert "venv" in content or ".venv" in content
        
        # Check that output files are excluded
        assert "*.epub" in content or "*.mp3" in content or "*.wav" in content


class TestDockerfileContent:
    """Test Dockerfile content and structure."""
    
    @pytest.fixture
    def docker_dir(self):
        return Path(__file__).parent.parent / "docker"
    
    def test_cpu_dockerfile_multistage(self, docker_dir):
        """Test CPU Dockerfile uses multi-stage build."""
        with open(docker_dir / "Dockerfile.cpu", "r") as f:
            content = f.read()
        
        # Check for multi-stage build (FROM ... AS builder)
        assert "AS builder" in content or "as builder" in content
        assert "AS runtime" in content or "as runtime" in content
        
        # Check for UV installation
        assert "uv" in content.lower()
        
        # Check for non-root user
        assert "useradd" in content or "USER" in content
    
    def test_gpu_dockerfile_cuda(self, docker_dir):
        """Test GPU Dockerfile uses CUDA base image."""
        with open(docker_dir / "Dockerfile.gpu", "r") as f:
            content = f.read()
        
        # Check for CUDA base image
        assert "nvidia/cuda" in content
        
        # Check for multi-stage build
        assert "AS builder" in content or "as builder" in content
        
        # Check for CUDA environment variables
        assert "CUDA_VISIBLE_DEVICES" in content
    
    def test_dockerfiles_have_healthcheck(self, docker_dir):
        """Test both Dockerfiles have health checks."""
        for dockerfile in ["Dockerfile.cpu", "Dockerfile.gpu"]:
            with open(docker_dir / dockerfile, "r") as f:
                content = f.read()
            assert "HEALTHCHECK" in content, f"{dockerfile} missing HEALTHCHECK"
    
    def test_dockerfiles_expose_port_7860(self, docker_dir):
        """Test both Dockerfiles expose port 7860."""
        for dockerfile in ["Dockerfile.cpu", "Dockerfile.gpu"]:
            with open(docker_dir / dockerfile, "r") as f:
                content = f.read()
            assert "EXPOSE 7860" in content, f"{dockerfile} missing EXPOSE 7860"
    
    def test_dockerfiles_install_ffmpeg(self, docker_dir):
        """Test both Dockerfiles install FFmpeg."""
        for dockerfile in ["Dockerfile.cpu", "Dockerfile.gpu"]:
            with open(docker_dir / dockerfile, "r") as f:
                content = f.read()
            assert "ffmpeg" in content.lower(), f"{dockerfile} missing ffmpeg installation"
    
    def test_dockerfiles_use_libgomp(self, docker_dir):
        """Test both Dockerfiles install libgomp (required by vibevoice)."""
        for dockerfile in ["Dockerfile.cpu", "Dockerfile.gpu"]:
            with open(docker_dir / dockerfile, "r") as f:
                content = f.read()
            assert "libgomp" in content, f"{dockerfile} missing libgomp installation"
    
    def test_cpu_dockerfile_uses_audio_processing(self, docker_dir):
        """Test CPU Dockerfile includes audio processing capabilities."""
        with open(docker_dir / "Dockerfile.cpu", "r") as f:
            content = f.read()
        
        # Check for ffmpeg (used for MP3 encoding in pydub)
        assert "ffmpeg" in content.lower(), "Dockerfile should install ffmpeg for audio processing"
        
        # Check for libgomp (required by vibevoice for audio synthesis)
        assert "libgomp" in content, "Dockerfile should install libgomp for vibevoice dependency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
