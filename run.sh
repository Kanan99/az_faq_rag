#!/bin/bash

# run.sh - Deployment script for Azerbaijani FAQ RAG System
# Usage: ./run.sh [command]
# Commands:
#   start       - Start the full system (Docker + Ollama)
#   stop        - Stop all components
#   restart     - Restart all components
#   ollama      - Manage Ollama (start|stop|restart|status)
#   app         - Manage the RAG application (start|stop|restart|status)
#   build       - Build or rebuild Docker images
#   logs        - Show logs
#   pull        - Pull an Ollama model
#   status      - Show system status
#   help        - Show this help message

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
    local level=$1
    local message=$2
    local color=$NC
    
    case $level in
        "INFO") color=$GREEN ;;
        "WARN") color=$YELLOW ;;
        "ERROR") color=$RED ;;
        "STEP") color=$BLUE ;;
    esac
    
    echo -e "${color}[$level] $message${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log "ERROR" "Docker is not installed. Please install Docker first."
        log "INFO" "Visit https://docs.docker.com/get-docker/ for installation instructions."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log "ERROR" "Docker is not running or you don't have permission to access it."
        log "INFO" "Start Docker service and ensure you have proper permissions."
        exit 1
    fi
}

# Check if Docker Compose is installed
check_docker_compose() {
    # First, try the new docker compose command
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
        return 0
    fi
    
    # If that fails, try the old docker-compose command
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
        return 0
    fi
    
    log "ERROR" "Docker Compose is not installed. Please install Docker Compose first."
    log "INFO" "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
}

# Check if Ollama is available
check_ollama() {
    # Check if Ollama is installed on the host
    if command -v ollama &> /dev/null; then
        OLLAMA_INSTALLED=true
        log "INFO" "Ollama is installed on the host system."
    else
        OLLAMA_INSTALLED=false
        log "WARN" "Ollama is not installed on the host system. Only Docker mode will be available."
    fi
    
    # Check if Ollama container is running
    if docker ps --format '{{.Names}}' | grep -q "ollama"; then
        OLLAMA_CONTAINER_RUNNING=true
        log "INFO" "Ollama container is running."
    else
        OLLAMA_CONTAINER_RUNNING=false
        log "INFO" "Ollama container is not running."
    fi
}

# Start the system
start_system() {
    log "STEP" "Starting the Azerbaijani FAQ RAG system..."
    
    # Create necessary directories
    mkdir -p data models uploads
    
    # Pull Ollama model if specified
    if [ -n "$OLLAMA_MODEL" ]; then
        pull_ollama_model "$OLLAMA_MODEL"
    fi
    
    # Start all services with Docker Compose
    log "STEP" "Starting Docker containers..."
    $DOCKER_COMPOSE up -d
    
    if [ $? -eq 0 ]; then
        log "INFO" "All containers started successfully."
        log "INFO" "The application is now available at http://localhost:8501"
    else
        log "ERROR" "Failed to start containers."
        exit 1
    fi
}

# Stop the system
stop_system() {
    log "STEP" "Stopping the Azerbaijani FAQ RAG system..."
    
    # Stop all services with Docker Compose
    $DOCKER_COMPOSE down
    
    if [ $? -eq 0 ]; then
        log "INFO" "All containers stopped successfully."
    else
        log "ERROR" "Failed to stop containers."
        exit 1
    fi
}

# Restart the system
restart_system() {
    log "STEP" "Restarting the Azerbaijani FAQ RAG system..."
    stop_system
    start_system
}

# Manage Ollama
manage_ollama() {
    local action=$1
    
    case $action in
        "start")
            log "STEP" "Starting Ollama container..."
            $DOCKER_COMPOSE up -d ollama
            if [ $? -eq 0 ]; then
                log "INFO" "Ollama container started successfully."
            else
                log "ERROR" "Failed to start Ollama container."
                exit 1
            fi
            ;;
        "stop")
            log "STEP" "Stopping Ollama container..."
            $DOCKER_COMPOSE stop ollama
            if [ $? -eq 0 ]; then
                log "INFO" "Ollama container stopped successfully."
            else
                log "ERROR" "Failed to stop Ollama container."
                exit 1
            fi
            ;;
        "restart")
            log "STEP" "Restarting Ollama container..."
            $DOCKER_COMPOSE restart ollama
            if [ $? -eq 0 ]; then
                log "INFO" "Ollama container restarted successfully."
            else
                log "ERROR" "Failed to restart Ollama container."
                exit 1
            fi
            ;;
        "status")
            if docker ps --format '{{.Names}}' | grep -q "ollama"; then
                log "INFO" "Ollama container is running."
            else
                log "INFO" "Ollama container is not running."
            fi
            ;;
        *)
            log "ERROR" "Unknown Ollama action: $action"
            log "INFO" "Available actions: start, stop, restart, status"
            exit 1
            ;;
    esac
}

# Manage the RAG application
manage_app() {
    local action=$1
    
    case $action in
        "start")
            log "STEP" "Starting RAG application container..."
            $DOCKER_COMPOSE up -d rag_app
            if [ $? -eq 0 ]; then
                log "INFO" "RAG application container started successfully."
                log "INFO" "The application is now available at http://localhost:8501"
            else
                log "ERROR" "Failed to start RAG application container."
                exit 1
            fi
            ;;
        "stop")
            log "STEP" "Stopping RAG application container..."
            $DOCKER_COMPOSE stop rag_app
            if [ $? -eq 0 ]; then
                log "INFO" "RAG application container stopped successfully."
            else
                log "ERROR" "Failed to stop RAG application container."
                exit 1
            fi
            ;;
        "restart")
            log "STEP" "Restarting RAG application container..."
            $DOCKER_COMPOSE restart rag_app
            if [ $? -eq 0 ]; then
                log "INFO" "RAG application container restarted successfully."
                log "INFO" "The application is now available at http://localhost:8501"
            else
                log "ERROR" "Failed to restart RAG application container."
                exit 1
            fi
            ;;
        "status")
            if docker ps --format '{{.Names}}' | grep -q "az_faq_rag"; then
                log "INFO" "RAG application container is running."
                log "INFO" "The application is available at http://localhost:8501"
            else
                log "INFO" "RAG application container is not running."
            fi
            ;;
        *)
            log "ERROR" "Unknown application action: $action"
            log "INFO" "Available actions: start, stop, restart, status"
            exit 1
            ;;
    esac
}

# Build Docker images
build_images() {
    log "STEP" "Building Docker images..."
    $DOCKER_COMPOSE build
    
    if [ $? -eq 0 ]; then
        log "INFO" "Docker images built successfully."
    else
        log "ERROR" "Failed to build Docker images."
        exit 1
    fi
}

# Show logs
show_logs() {
    local service=$1
    local lines=${2:-100}
    
    if [ -z "$service" ]; then
        log "STEP" "Showing logs for all services (last $lines lines)..."
        $DOCKER_COMPOSE logs --tail="$lines"
    else
        log "STEP" "Showing logs for $service (last $lines lines)..."
        $DOCKER_COMPOSE logs --tail="$lines" "$service"
    fi
}

# Pull an Ollama model
pull_ollama_model() {
    local model=$1
    
    if [ -z "$model" ]; then
        log "ERROR" "No model specified."
        log "INFO" "Usage: ./run.sh pull <model_name>"
        exit 1
    fi
    
    log "STEP" "Pulling Ollama model: $model..."
    
    # If Ollama container is running, use it
    if docker ps --format '{{.Names}}' | grep -q "ollama"; then
        docker exec ollama ollama pull "$model"
    # If Ollama is installed on host, use it
    elif command -v ollama &> /dev/null; then
        ollama pull "$model"
    else
        log "ERROR" "Ollama is not available. Start the Ollama container first or install Ollama."
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        log "INFO" "Model $model pulled successfully."
    else
        log "ERROR" "Failed to pull model $model."
        exit 1
    fi
}

# Show system status
show_status() {
    log "STEP" "Checking system status..."
    
    # Check if Docker and Docker Compose are available
    if command -v docker &> /dev/null; then
        log "INFO" "Docker is installed."
    else
        log "WARN" "Docker is not installed."
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        log "INFO" "Docker Compose (new) is installed."
    elif command -v docker-compose &> /dev/null; then
        log "INFO" "Docker Compose (old) is installed."
    else
        log "WARN" "Docker Compose is not installed."
    fi
    
    # Check Ollama
    check_ollama
    
    # Check container status
    log "STEP" "Checking container status..."
    docker ps --filter "name=az_faq_rag|ollama"
    
    # List available Ollama models
    log "STEP" "Available Ollama models:"
    if $OLLAMA_CONTAINER_RUNNING; then
        docker exec ollama ollama list
    elif $OLLAMA_INSTALLED; then
        ollama list
    else
        log "WARN" "Ollama is not available."
    fi
}

# Show help
show_help() {
    echo -e "${BLUE}Azerbaijani FAQ RAG System - Management Script${NC}"
    echo -e "Usage: ./run.sh [command] [options]"
    echo -e ""
    echo -e "Commands:"
    echo -e "  ${GREEN}start${NC}                   Start the full system (Docker + Ollama)"
    echo -e "  ${GREEN}stop${NC}                    Stop all components"
    echo -e "  ${GREEN}restart${NC}                 Restart all components"
    echo -e "  ${GREEN}ollama${NC} [action]         Manage Ollama (start|stop|restart|status)"
    echo -e "  ${GREEN}app${NC} [action]            Manage the RAG application (start|stop|restart|status)"
    echo -e "  ${GREEN}build${NC}                   Build or rebuild Docker images"
    echo -e "  ${GREEN}logs${NC} [service] [lines]  Show logs (default: all services, 100 lines)"
    echo -e "  ${GREEN}pull${NC} [model]            Pull an Ollama model (e.g., mistral:latest)"
    echo -e "  ${GREEN}status${NC}                  Show system status"
    echo -e "  ${GREEN}help${NC}                    Show this help message"
    echo -e ""
    echo -e "Examples:"
    echo -e "  ./run.sh start                  # Start all components"
    echo -e "  ./run.sh pull mistral:latest    # Pull the Mistral model"
    echo -e "  ./run.sh logs rag_app 50        # Show last 50 lines of RAG app logs"
    echo -e "  ./run.sh ollama restart         # Restart just the Ollama container"
}

# Main script execution
# Check for required commands
check_docker
check_docker_compose
check_ollama

# Process command line arguments
command=$1
shift

case $command in
    "start")
        OLLAMA_MODEL=$1
        start_system
        ;;
    "stop")
        stop_system
        ;;
    "restart")
        restart_system
        ;;
    "ollama")
        action=$1
        manage_ollama "$action"
        ;;
    "app")
        action=$1
        manage_app "$action"
        ;;
    "build")
        build_images
        ;;
    "logs")
        service=$1
        lines=$2
        show_logs "$service" "$lines"
        ;;
    "pull")
        model=$1
        pull_ollama_model "$model"
        ;;
    "status")
        show_status
        ;;
    "help"|"")
        show_help
        ;;
    *)
        log "ERROR" "Unknown command: $command"
        show_help
        exit 1
        ;;
esac

exit 0
