# OpenCV Python Project Makefile

# Variables
PROJECT_NAME=opencv-python-project
DOCKER_IMAGE=opencv-python
CONTAINER_NAME=opencv_dev
PYTHON_VERSION=3.11

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(GREEN)OpenCV Python Project Makefile$(NC)"
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup and Installation
.PHONY: setup
setup: ## Set up the development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	python -m pip install --upgrade pip
	pip install -r pysrc/requirements.txt
	@echo "$(GREEN)Setup complete!$(NC)"

.PHONY: setup-dev
setup-dev: setup ## Set up development environment with additional tools
	@echo "$(GREEN)Installing development tools...$(NC)"
	pip install jupyter black flake8 pytest pytest-cov
	@echo "$(GREEN)Development setup complete!$(NC)"

# Docker commands
.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):latest .
	@echo "$(GREEN)Docker image built successfully!$(NC)"

.PHONY: docker-build-dev
docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(NC)"
	docker build --target development -t $(DOCKER_IMAGE):dev .
	@echo "$(GREEN)Development Docker image built successfully!$(NC)"

.PHONY: docker-run
docker-run: ## Run Docker container with camera access
	@echo "$(GREEN)Starting Docker container...$(NC)"
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		-v $(PWD)/pysrc:/app \
		-v $(PWD)/outputs:/app/outputs \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		--device=/dev/video0:/dev/video0 \
		--privileged \
		$(DOCKER_IMAGE):latest

.PHONY: docker-dev
docker-dev: ## Run development Docker container with shell access
	@echo "$(GREEN)Starting development container...$(NC)"
	docker run -it --rm \
		--name $(CONTAINER_NAME)_dev \
		-v $(PWD)/pysrc:/app \
		-v $(PWD)/outputs:/app/outputs \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		--device=/dev/video0:/dev/video0 \
		--privileged \
		$(DOCKER_IMAGE):dev bash

.PHONY: docker-compose-up
docker-compose-up: ## Start services with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(NC)"
	docker-compose up -d

.PHONY: docker-compose-dev
docker-compose-dev: ## Start development environment with docker-compose
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker-compose up opencv-dev

.PHONY: docker-compose-down
docker-compose-down: ## Stop docker-compose services
	@echo "$(GREEN)Stopping docker-compose services...$(NC)"
	docker-compose down

.PHONY: jupyter
jupyter: ## Start Jupyter notebook server
	@echo "$(GREEN)Starting Jupyter notebook server...$(NC)"
	docker-compose up jupyter

# Application commands
.PHONY: run
run: ## Run the main object tracking application
	@echo "$(GREEN)Running object tracking application...$(NC)"
	cd pysrc && python object_tracking.py

.PHONY: run-face-detection
run-face-detection: ## Run face detection application
	@echo "$(GREEN)Running face detection application...$(NC)"
	cd pysrc && python task3/face_detection.py

.PHONY: run-edge-detection
run-edge-detection: ## Run Canny edge detection application
	@echo "$(GREEN)Running edge detection application...$(NC)"
	cd pysrc && python task4/canny_edge_detection.py

.PHONY: run-shape-detection
run-shape-detection: ## Run shape detection application
	@echo "$(GREEN)Running shape detection application...$(NC)"
	cd pysrc && python task4/square_detection.py

.PHONY: run-histogram
run-histogram: ## Run real-time histogram application
	@echo "$(GREEN)Running histogram application...$(NC)"
	cd pysrc && python task2/histogram_realtime.py

# Testing and Quality
.PHONY: test
test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	cd pysrc && python -m pytest -v

.PHONY: lint
lint: ## Run code linting
	@echo "$(GREEN)Running code linting...$(NC)"
	cd pysrc && python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

.PHONY: format
format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	cd pysrc && python -m black .

.PHONY: check
check: lint test ## Run all quality checks

# Utility commands
.PHONY: clean
clean: ## Clean up generated files
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf outputs/*.jpg outputs/*.png outputs/*.avi
	@echo "$(GREEN)Cleanup complete!$(NC)"

.PHONY: clean-docker
clean-docker: ## Clean up Docker images and containers
	@echo "$(GREEN)Cleaning up Docker resources...$(NC)"
	docker system prune -f
	docker image prune -f
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

.PHONY: create-dirs
create-dirs: ## Create necessary directories
	@echo "$(GREEN)Creating directories...$(NC)"
	mkdir -p outputs logs notebooks archive
	@echo "$(GREEN)Directories created!$(NC)"

# Information commands
.PHONY: info
info: ## Show project information
	@echo "$(GREEN)Project Information:$(NC)"
	@echo "Project Name: $(PROJECT_NAME)"
	@echo "Docker Image: $(DOCKER_IMAGE)"
	@echo "Python Version: $(PYTHON_VERSION)"
	@echo "OpenCV Version: $$(cd pysrc && python -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'Not installed')"

.PHONY: list-apps
list-apps: ## List available applications
	@echo "$(GREEN)Available Applications:$(NC)"
	@echo "  $(BLUE)object_tracking.py$(NC)          - Main object tracking"
	@echo "  $(BLUE)task1/$(NC)"
	@echo "    $(BLUE)change_color.py$(NC)           - HSV color manipulation"
	@echo "    $(BLUE)click_change.py$(NC)           - Interactive color segmentation"
	@echo "    $(BLUE)circle_detection.py$(NC)       - Hough circle detection"
	@echo "    $(BLUE)template_matching.py$(NC)      - Template matching"
	@echo "  $(BLUE)task2/$(NC)"
	@echo "    $(BLUE)histogram_realtime.py$(NC)     - Real-time histograms"
	@echo "  $(BLUE)task3/$(NC)"
	@echo "    $(BLUE)face_detection.py$(NC)         - Face and eye detection"
	@echo "    $(BLUE)background_subtraction_*.py$(NC) - Motion detection"
	@echo "  $(BLUE)task4/$(NC)"
	@echo "    $(BLUE)canny_edge_detection.py$(NC)   - Edge detection"
	@echo "    $(BLUE)square_detection.py$(NC)       - Shape detection"
	@echo "    $(BLUE)hough_line_detection.py$(NC)   - Line detection"

# Quick start commands
.PHONY: quick-start
quick-start: create-dirs docker-build ## Quick start with Docker
	@echo "$(GREEN)Quick start complete! Use 'make docker-run' to start the application.$(NC)"

.PHONY: dev-start
dev-start: create-dirs setup-dev ## Quick start for development
	@echo "$(GREEN)Development environment ready! Use 'make run' to start applications.$(NC)"

# Camera test
.PHONY: test-camera
test-camera: ## Test camera access
	@echo "$(GREEN)Testing camera access...$(NC)"
	cd pysrc && python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened()); cap.release()"

# X11 setup (for Linux GUI)
.PHONY: setup-x11
setup-x11: ## Set up X11 forwarding for GUI applications
	@echo "$(GREEN)Setting up X11 forwarding...$(NC)"
	xhost +local:docker
	@echo "$(GREEN)X11 forwarding enabled for Docker!$(NC)"

# All-in-one commands
.PHONY: install
install: setup create-dirs ## Install everything needed for local development

.PHONY: docker-install
docker-install: docker-build create-dirs setup-x11 ## Install everything needed for Docker development

.PHONY: demo
demo: ## Run a quick demo of the main features
	@echo "$(GREEN)Running demo sequence...$(NC)"
	@echo "$(YELLOW)1. Testing camera...$(NC)"
	$(MAKE) test-camera
	@echo "$(YELLOW)2. Running object tracking for 10 seconds...$(NC)"
	timeout 10s $(MAKE) run || true
	@echo "$(GREEN)Demo complete!$(NC)"