#!/usr/bin/env bash
set -euo pipefail

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
RESET='\033[0m'

CONFIG_FILE="config/config.yaml"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    printf "${RED}✗ Error: ${CONFIG_FILE} not found${RESET}\n"
    exit 1
fi

# Parse config to build mapping of endpoint -> model -> agents
# This awk script tracks the current agent context and extracts api_base and model pairs
# Format: "endpoint|model|agent1, agent2, ..."
MODEL_AGENT_MAP=$(awk '
    /^[[:space:]]*agent_llm:/ { agent="Topic Segmentation Agent"; in_block=1; next }
    /^[[:space:]]*critic_llm:/ { agent="Topic Segmentation Critic"; in_block=1; next }
    /^[[:space:]]*writer_llm:/ { agent="Article Writer"; in_block=1; next }

    in_block && /^[[:space:]]*model:/ { model=$2; has_model=1 }
    in_block && /^[[:space:]]*api_base:/ { api_base=$2; has_api_base=1 }

    in_block && has_model && has_api_base {
        print api_base "|" model "|" agent
        has_model=0
        has_api_base=0
        in_block=0
    }

    /^[a-z]/ && in_block { in_block=0; has_model=0; has_api_base=0 }
' "$CONFIG_FILE")

# Function to get agents for a given model (ignoring endpoint)
get_agents_for_model() {
    local model="$1"
    local agents=""

    while IFS='|' read -r map_endpoint map_model map_agent; do
        # Strip "openai/" prefix from config model for comparison
        local stripped_model="${map_model#openai/}"

        if [[ "$stripped_model" == "$model" || "$map_model" == "$model" ]]; then
            if [[ -z "$agents" ]]; then
                agents="$map_agent"
            else
                agents="$agents, $map_agent"
            fi
        fi
    done <<< "$MODEL_AGENT_MAP"

    echo "$agents"
}

# Extract all api_base URLs using grep and deduplicate
ENDPOINTS=$(grep 'api_base:' "$CONFIG_FILE" | awk '{print $2}' | sort -u)

if [[ -z "$ENDPOINTS" ]]; then
    printf "${RED}✗ Error: No api_base fields found in ${CONFIG_FILE}${RESET}\n"
    exit 1
fi

# Count unique endpoints
ENDPOINT_COUNT=$(echo "$ENDPOINTS" | wc -l | tr -d ' ')
TOTAL_MODELS=0
EXIT_CODE=0

# Display configured models first
printf "\n${BLUE}Configured Models:${RESET}\n"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

if [[ -n "$MODEL_AGENT_MAP" ]]; then
    while IFS='|' read -r map_endpoint map_model map_agent; do
        # Strip "openai/" prefix for display
        display_model="${map_model#openai/}"
        printf "  • ${YELLOW}%s${RESET}\n" "$display_model"
        printf "    ${BLUE}Agent:${RESET} %s\n" "$map_agent"
        printf "    ${BLUE}Endpoint:${RESET} %s\n" "$map_endpoint"
        printf "\n"
    done <<< "$MODEL_AGENT_MAP"
else
    printf "  ${RED}No models configured${RESET}\n"
fi

# Query each endpoint
while IFS= read -r endpoint; do
    printf "\n${BLUE}API Endpoint:${RESET} %s\n" "$endpoint"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    # Query the /models endpoint with timeout
    RESPONSE=$(curl -s -f --connect-timeout 5 --max-time 10 "${endpoint}/models" 2>/dev/null) || {
        printf "${RED}✗ Error: Failed to connect to %s${RESET}\n" "$endpoint"
        EXIT_CODE=1
        continue
    }

    # Parse JSON and extract model IDs
    MODELS=$(echo "$RESPONSE" | jq -r '.data[].id' 2>/dev/null) || {
        printf "${RED}✗ Error: Invalid JSON response from %s${RESET}\n" "$endpoint"
        EXIT_CODE=1
        continue
    }

    if [[ -z "$MODELS" ]]; then
        printf "  ${RED}No models available${RESET}\n"
        continue
    fi

    # Display models
    MODEL_COUNT=0
    while IFS= read -r model; do
        # Check if this model is used by any agent (regardless of endpoint)
        agents=$(get_agents_for_model "$model")
        if [[ -n "$agents" ]]; then
            printf "  • %s    ${YELLOW}<-- used by: %s${RESET}\n" "$model" "$agents"
        else
            printf "  • %s\n" "$model"
        fi
        ((MODEL_COUNT++))
        ((TOTAL_MODELS++))
    done <<< "$MODELS"

    printf "\n${GREEN}Total: %d models${RESET}\n" "$MODEL_COUNT"
done <<< "$ENDPOINTS"

# Display summary
printf "\n"
if [[ $EXIT_CODE -eq 0 ]]; then
    printf "${GREEN}✓ Found %d unique endpoint(s) with %d total models${RESET}\n" "$ENDPOINT_COUNT" "$TOTAL_MODELS"
else
    printf "${RED}⚠ Found %d unique endpoint(s), but some failed${RESET}\n" "$ENDPOINT_COUNT"
fi

exit $EXIT_CODE
