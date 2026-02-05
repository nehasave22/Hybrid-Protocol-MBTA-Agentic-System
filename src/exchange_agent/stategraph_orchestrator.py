"""
StateGraph-based Orchestrator for MBTA Agntcy
SEMANTIC AGENT DISCOVERY: LLM matches queries to agent descriptions
NO HARDCODED CAPABILITIES - Purely natural language matching
"""
import os
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from dataclasses import dataclass
import asyncio
import httpx
from opentelemetry import trace
import logging
from urllib.parse import urlparse
from datetime import datetime, timedelta
from openai import OpenAI
import json

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Registry configuration
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://23.92.17.180:6900")

# Discovery cache
_agent_catalog_cache = None
_catalog_cache_time = None
_catalog_cache_ttl = timedelta(minutes=5)  # Refresh agent catalog every 5 minutes


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """The state that flows through the StateGraph"""
    # Input
    user_message: str
    conversation_id: str
    intent: str
    confidence: float
    
    # Agent selection (discovered semantically)
    matched_agents: list[str]  # Agent IDs discovered by semantic matching
    
    # Agent execution tracking
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agents_called: list[str]
    agent_responses: list[dict]
    
    # Final output
    final_response: str
    should_end: bool
    
    # LLM decision metadata
    llm_matching_decision: dict | None


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

@dataclass
class AgentConfig:
    name: str
    url: str
    port: int
    description: str
    capabilities: list[str]  # Optional, for backward compatibility
    discovered_from_registry: bool = True


async def validate_registry_connection() -> bool:
    """Validate that registry is accessible."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{REGISTRY_URL}/health")
            return response.status_code == 200
    except Exception as e:
        logger.error(f"❌ Registry health check failed: {e}")
        return False


async def get_agent_catalog_from_registry() -> list[dict]:
    """
    Get ALL registered agents with their descriptions from registry.
    This is cached for 5 minutes to reduce registry load.
    
    Returns:
        List of agent info dicts with descriptions:
        [
            {
                "agent_id": "mbta-alerts",
                "agent_url": "http://96.126.111.107:8001",
                "description": "Provides real-time service alerts...",
                "capabilities": ["alerts"],  # Optional
                "alive": true
            },
            ...
        ]
    """
    global _agent_catalog_cache, _catalog_cache_time
    
    # Check cache
    if _agent_catalog_cache and _catalog_cache_time:
        cache_age = (datetime.now() - _catalog_cache_time).total_seconds()
        if datetime.now() - _catalog_cache_time < _catalog_cache_ttl:
            logger.info(f"💾 Agent catalog cache hit ({len(_agent_catalog_cache)} agents, age: {cache_age:.1f}s)")
            return _agent_catalog_cache
        else:
            logger.info(f"🔄 Agent catalog cache expired, refreshing...")
    
    logger.info("🔍 Fetching agent catalog from registry...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get all agent IDs
            response = await client.get(f"{REGISTRY_URL}/list")
            response.raise_for_status()
            agent_list = response.json()
            
            # Get detailed info for each agent
            agents_info = []
            for agent_id in agent_list.keys():
                if agent_id == 'agent_status':
                    continue
                
                try:
                    # Get full agent details
                    agent_response = await client.get(f"{REGISTRY_URL}/agents/{agent_id}")
                    if agent_response.status_code != 200:
                        continue
                    
                    agent_data = agent_response.json()
                    
                    # Only include alive agents
                    if not agent_data.get("alive"):
                        logger.debug(f"Skipping {agent_id} (not alive)")
                        continue
                    
                    agents_info.append({
                        "agent_id": agent_data.get("agent_id"),
                        "agent_url": agent_data.get("agent_url"),
                        "description": agent_data.get("description", "No description available"),
                        "capabilities": agent_data.get("capabilities", []),
                        "alive": agent_data.get("alive", False)
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️  Could not get details for {agent_id}: {e}")
                    continue
            
            # Cache the catalog
            _agent_catalog_cache = agents_info
            _catalog_cache_time = datetime.now()
            
            logger.info(f"✅ Agent catalog loaded: {len(agents_info)} alive agents")
            for agent in agents_info:
                logger.info(f"   • {agent['agent_id']}: {agent['description'][:60]}...")
            
            return agents_info
            
    except Exception as e:
        logger.error(f"❌ Failed to fetch agent catalog: {e}")
        return []


async def semantic_agent_discovery(query: str) -> list[AgentConfig]:
    """
    SEMANTIC DISCOVERY: LLM matches query to agent descriptions.
    
    
    Args:
        query: User's natural language query
        
    Returns:
        List of matched agents (ordered by relevance)
    """
    
    with tracer.start_as_current_span("semantic_agent_discovery") as span:
        span.set_attribute("query", query)
        
        # Get all available agents with descriptions
        agent_catalog = await get_agent_catalog_from_registry()
        
        if not agent_catalog:
            logger.warning("⚠️  No agents available in registry")
            return []
        
        # Build agent catalog text for LLM
        agent_descriptions = []
        for agent in agent_catalog:
            agent_descriptions.append(
                f"• {agent['agent_id']}: {agent['description']}"
            )
        
        catalog_text = "\n".join(agent_descriptions)
        
        # Ask LLM to semantically match query to descriptions
        prompt = f"""You are an intelligent agent router that matches user queries to available agents using semantic understanding.

User Query: "{query}"

Available Agents (each with natural language description):
{catalog_text}

Your Task:
1. Read and understand what the user is asking for
2. Read ALL agent descriptions carefully
3. Identify which agent(s) can best handle this query based on their descriptions
4. Consider semantic meaning, not just keyword matching
5. You can select multiple agents if the query needs multiple capabilities
6. Return agent IDs in order of relevance (most relevant first)

Examples of good semantic matching:
- Query: "Are there delays?" → Agent describing "real-time service alerts" ✓
- Query: "Find stops near me" → Agent describing "location-based stop search" ✓
- Query: "Bike route avoiding hills" → Agent mentioning "elevation-aware cycling routes" ✓
- Query: "How do I get to Harvard?" → Agent describing "route planning and directions" ✓

Return ONLY valid JSON:
{{
  "matched_agents": ["agent_id_1", "agent_id_2"],
  "reasoning": "Brief explanation of semantic match",
  "confidence": 0.9
}}

If NO agents match the query semantically, return empty list.
"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at semantic matching. Carefully read descriptions and match based on meaning, not just keywords. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            matched_agent_ids = result.get("matched_agents", [])
            reasoning = result.get("reasoning", "No reasoning provided")
            confidence = result.get("confidence", 0.5)
            
            logger.info(f"🤖 Semantic Agent Matching:")
            logger.info(f"   Query: {query[:60]}...")
            logger.info(f"   Matched: {', '.join(matched_agent_ids) if matched_agent_ids else 'None'}")
            logger.info(f"   Reasoning: {reasoning}")
            logger.info(f"   Confidence: {confidence:.2f}")
            
            span.set_attribute("matched_agents", ",".join(matched_agent_ids))
            span.set_attribute("match_confidence", confidence)
            
            # Convert matched IDs to AgentConfig objects
            matched_configs = []
            for agent_id in matched_agent_ids:
                # Find agent in catalog
                agent_info = next((a for a in agent_catalog if a['agent_id'] == agent_id), None)
                
                if not agent_info:
                    logger.warning(f"⚠️  Matched agent {agent_id} not found in catalog")
                    continue
                
                # Parse URL
                parsed = urlparse(agent_info['agent_url'])
                
                config = AgentConfig(
                    name=agent_info['agent_id'],
                    url=f"{parsed.scheme or 'http'}://{parsed.hostname}",
                    port=parsed.port or 80,
                    description=agent_info['description'],
                    capabilities=agent_info.get('capabilities', []),
                    discovered_from_registry=True
                )
                
                matched_configs.append(config)
                logger.info(f"✅ Matched: {config.name} → {config.url}:{config.port}")
            
            return matched_configs
            
        except Exception as e:
            logger.error(f"❌ Semantic matching failed: {e}")
            return []


async def call_agent_api(agent_config: AgentConfig, message: str, conversation_id: str) -> dict:
    """
    Call an agent via A2A protocol.
    
    Args:
        agent_config: Agent configuration (discovered semantically)
        message: Message to send to agent
        conversation_id: Conversation ID for tracking
        
    Returns:
        Agent response or error dict
    """
    
    url = f"{agent_config.url}:{agent_config.port}/a2a/message"
    
    payload = {
        "type": "request",
        "payload": {
            "message": message,
            "conversation_id": conversation_id
        },
        "metadata": {
            "source": "stategraph-orchestrator",
            "agent_name": agent_config.name,
            "discovered_via": "semantic-description-matching"
        }
    }
    
    # Add HTTP call span
    with tracer.start_as_current_span(
        f"http_post: {agent_config.name}",
        attributes={
            "http.method": "POST",
            "http.url": url,
            "http.target": "/a2a/message"
        }
    ) as http_span:
        logger.info(f"📞 Calling {agent_config.name} at {url}")
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, json=payload)
                
                http_span.set_attribute("http.status_code", response.status_code)
                
                response.raise_for_status()
                result = response.json()
                
                # Extract response from A2A envelope
                if result.get("type") == "response" and "payload" in result:
                    response_text = result["payload"].get("text", "")
                    logger.info(f"📥 Response from {agent_config.name}: {len(response_text)} chars")
                    
                    http_span.set_attribute("response.text.length", len(response_text))
                    
                    return {
                        "response": response_text,
                        "payload": result["payload"],
                        "agent_used": agent_config.name,
                        "agent_description": agent_config.description
                    }
                
                return result
                
        except httpx.HTTPError as e:
            error_msg = f"Agent API call failed for {agent_config.name}: {e}"
            logger.error(f"❌ {error_msg}")
            
            http_span.set_attribute("error", True)
            http_span.set_attribute("error.message", str(e))
            http_span.record_exception(e)
            
            return {
                "response": f"Error calling agent {agent_config.name}: {e}",
                "error": True,
                "agent_attempted": agent_config.name
            }


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

async def semantic_discovery_node(state: AgentState) -> AgentState:
    """
    Use LLM to semantically match query to agent descriptions.
    This is the CORE INNOVATION of the system!
    """
    with tracer.start_as_current_span("semantic_discovery_node") as span:
        span.set_attribute("user_message", state["user_message"])
        
        # Semantic agent discovery
        matched_agents = await semantic_agent_discovery(state["user_message"])
        
        # Extract just the agent IDs
        matched_agent_ids = [agent.name for agent in matched_agents]
        
        # Determine intent based on what was matched
        intent = "general"
        confidence = 0.5
        
        if matched_agent_ids:
            # Infer intent from first matched agent's description
            first_agent = matched_agents[0]
            desc_lower = first_agent.description.lower()
            
            if any(word in desc_lower for word in ["alert", "delay", "disruption"]):
                intent = "alerts"
                confidence = 0.85
            elif any(word in desc_lower for word in ["stop", "station", "location"]):
                intent = "stops"
                confidence = 0.85
            elif any(word in desc_lower for word in ["route", "planning", "direction", "trip"]):
                intent = "trip_planning"
                confidence = 0.85
            else:
                intent = "transit_info"
                confidence = 0.75
        
        logger.info(f"🎯 Semantic Discovery Result:")
        logger.info(f"   Matched: {', '.join(matched_agent_ids) if matched_agent_ids else 'None'}")
        logger.info(f"   Inferred Intent: {intent} (confidence: {confidence:.2f})")
        
        span.set_attribute("matched_agents", ",".join(matched_agent_ids))
        span.set_attribute("intent", intent)
        span.set_attribute("confidence", confidence)
        
        return {
            **state,
            "matched_agents": matched_agent_ids,
            "intent": intent,
            "confidence": confidence,
            "agent_responses": [],
            "agents_called": [],
            "messages": [HumanMessage(content=state["user_message"])],
            "llm_matching_decision": {
                "matched_agents": matched_agent_ids,
                "agent_configs": [
                    {"name": a.name, "description": a.description} 
                    for a in matched_agents
                ]
            }
        }


async def execute_agents_node(state: AgentState) -> AgentState:
    """
    Execute all matched agents sequentially.
    Collects responses from each agent.
    """
    with tracer.start_as_current_span("execute_agents_node") as span:
        matched_agent_ids = state.get("matched_agents", [])
        
        if not matched_agent_ids:
            logger.info("ℹ️  No agents matched - will handle as general query")
            return {
                **state,
                "agents_called": [],
                "agent_responses": []
            }
        
        logger.info(f"🔄 Executing {len(matched_agent_ids)} matched agent(s)")
        span.set_attribute("agents.to_execute", len(matched_agent_ids))
        span.set_attribute("agent_ids", json.dumps(matched_agent_ids))
        
        # Get fresh agent catalog to get AgentConfig objects
        agent_catalog = await get_agent_catalog_from_registry()
        
        responses = []
        agents_called = []
        
        for agent_id in matched_agent_ids:
            # Find agent in catalog
            agent_info = next((a for a in agent_catalog if a['agent_id'] == agent_id), None)
            
            if not agent_info:
                logger.warning(f"⚠️  Agent {agent_id} not found in catalog")
                continue
            
            # Parse URL and create config
            parsed = urlparse(agent_info['agent_url'])
            agent_config = AgentConfig(
                name=agent_info['agent_id'],
                url=f"{parsed.scheme or 'http'}://{parsed.hostname}",
                port=parsed.port or 80,
                description=agent_info['description'],
                capabilities=agent_info.get('capabilities', []),
                discovered_from_registry=True
            )
            
            # CREATE A DEDICATED SPAN FOR THIS SPECIFIC AGENT
            with tracer.start_as_current_span(
                f"agent: {agent_id}",  # This will show clearly in Jaeger!
                attributes={
                    "agent.id": agent_id,
                    "agent.name": agent_config.name,
                    "agent.url": f"{agent_config.url}:{agent_config.port}",
                    "agent.description": agent_info['description'][:100],
                    "agent.type": "a2a",
                    "query": state["user_message"]
                }
            ) as agent_span:
                try:
                    logger.info(f"📞 Calling agent {agent_id} ({agent_config.description[:40]}...)")
                    
                    # Call the agent
                    result = await call_agent_api(
                        agent_config=agent_config,
                        message=state["user_message"],
                        conversation_id=state["conversation_id"]
                    )
                    
                    # Mark success/failure
                    if result.get("error"):
                        agent_span.set_attribute("agent.status", "error")
                        agent_span.set_attribute("agent.error", str(result.get("response", "")))
                        agent_label = f"{agent_id} (failed)"
                        logger.error(f"❌ Agent {agent_id} failed: {result.get('response')}")
                    else:
                        agent_span.set_attribute("agent.status", "success")
                        agent_span.set_attribute("response.length", len(result.get("response", "")))
                        agent_label = agent_id
                        logger.info(f"✅ Agent {agent_id} responded successfully")
                    
                    responses.append(result)
                    agents_called.append(agent_label)
                    
                except Exception as e:
                    agent_span.set_attribute("agent.status", "exception")
                    agent_span.set_attribute("agent.error", str(e))
                    agent_span.record_exception(e)
                    logger.error(f"❌ Exception calling agent {agent_id}: {e}")
                    
                    responses.append({
                        "response": f"Error calling agent {agent_id}: {str(e)}",
                        "error": True,
                        "agent_attempted": agent_id
                    })
                    agents_called.append(f"{agent_id} (error)")
        
        span.set_attribute("agents.executed", len(agents_called))
        span.set_attribute("agents.called", json.dumps(agents_called))
        
        return {
            **state,
            "agent_responses": responses,
            "agents_called": agents_called,
            "messages": [
                AIMessage(content=f"{r.get('agent_used', 'agent')}: {r.get('response', '')[:100]}...", 
                         name=r.get('agent_used', 'agent'))
                for r in responses if not r.get('error')
            ]
        }


async def synthesize_response_node(state: AgentState) -> AgentState:
    """Synthesize all agent responses into final answer"""
    with tracer.start_as_current_span("synthesize_response_node"):
        
        # Handle general queries (no agents matched)
        matched_agents = state.get("matched_agents", [])
        if not matched_agents:
            message = state["user_message"].lower()
            
            if any(word in message for word in ["hi", "hello", "hey", "good morning", "whats up"]):
                return {
                    **state,
                    "final_response": "Hello! I'm MBTA Agntcy, your Boston transit assistant. I use semantic agent discovery to connect you with the right transit information. What would you like to know?",
                    "should_end": True
                }
            else:
                return {
                    **state,
                    "final_response": "I'm specialized in Boston MBTA transit information. I couldn't find agents that match your query - could you try asking about:\n• Service alerts and delays\n• Finding stops and stations\n• Planning routes and trips",
                    "should_end": True
                }
        
        # Collect agent responses
        responses = []
        had_errors = False
        
        for result in state.get("agent_responses", []):
            if result.get("error"):
                had_errors = True
                continue
            
            response_text = result.get("response", "")
            if response_text and response_text.strip():
                responses.append(response_text)
        
        # Build final response
        if responses:
            final_response = "\n\n".join(filter(None, responses))
            
            if had_errors:
                final_response += "\n\n⚠️ Note: Some agents were unavailable. Response may be incomplete."
        else:
            if had_errors:
                final_response = "I'm sorry, but I'm having trouble connecting to the necessary agents right now. Please try again in a moment."
            else:
                final_response = "I received your request but couldn't generate a complete response. Please try rephrasing your question."
        
        return {
            **state,
            "final_response": final_response,
            "should_end": True
        }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_discovery(state: AgentState) -> Literal["execute_agents", "synthesize"]:
    """Route based on whether agents were matched"""
    matched_agents = state.get("matched_agents", [])
    
    if matched_agents:
        logger.info(f"🤖 Routing to execute {len(matched_agents)} matched agent(s)")
        return "execute_agents"
    else:
        logger.info(f"🤖 No agents matched - routing to general handler")
        return "synthesize"


def route_after_execution(state: AgentState) -> Literal["synthesize"]:
    """Always synthesize after execution"""
    return "synthesize"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_mbta_graph() -> StateGraph:
    """Build StateGraph with semantic discovery"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("semantic_discovery", semantic_discovery_node)
    workflow.add_node("execute_agents", execute_agents_node)
    workflow.add_node("synthesize", synthesize_response_node)
    
    # Set entry point
    workflow.set_entry_point("semantic_discovery")
    
    # Conditional edges
    workflow.add_conditional_edges(
        "semantic_discovery",
        route_after_discovery,
        {
            "execute_agents": "execute_agents",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_agents",
        route_after_execution,
        {
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class StateGraphOrchestrator:
    """
    LLM-powered multi-agent orchestrator with SEMANTIC DESCRIPTION-BASED discovery.
    
    Zero hardcoded capabilities!
    Zero hardcoded agent names!
    Pure semantic matching of queries to agent descriptions!
    """
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("🚀 Initializing StateGraph Orchestrator (SEMANTIC DISCOVERY MODE)")
        logger.info(f"   Registry URL: {REGISTRY_URL}")
        logger.info(f"   Catalog Cache TTL: {_catalog_cache_ttl.total_seconds():.0f}s")
        logger.info(f"   Discovery Method: LLM semantic matching on agent descriptions")
        logger.info("=" * 80)
        
        self.graph = build_mbta_graph()
        logger.info("✅ StateGraph initialized (SEMANTIC + REGISTRY-BASED)")
    
    async def startup_validation(self):
        """
        Validate registry connectivity and load initial agent catalog.
        """
        logger.info("🔍 Running startup validation...")
        
        # Check registry is accessible
        registry_ok = await validate_registry_connection()
        if not registry_ok:
            raise RuntimeError(
                f"Registry at {REGISTRY_URL} is not accessible. "
                "Cannot start without registry connectivity."
            )
        logger.info(f"✅ Registry is accessible at {REGISTRY_URL}")
        
        # Load agent catalog (will be cached)
        agent_catalog = await get_agent_catalog_from_registry()
        
        if not agent_catalog:
            logger.warning("⚠️  Warning: No agents currently registered in registry")
            logger.warning("   System will start but queries will fail until agents are registered")
        else:
            logger.info(f"📚 Agent Catalog Loaded:")
            for agent in agent_catalog:
                logger.info(f"   ✅ {agent['agent_id']}")
                logger.info(f"      Description: {agent['description'][:70]}...")
                logger.info(f"      Endpoint: {agent['agent_url']}")
        
        logger.info("=" * 80)
        logger.info("✅ Startup validation complete")
        logger.info("🎯 System ready - agents will be discovered via semantic matching")
        logger.info("=" * 80)
    
    async def process_message(self, user_message: str, conversation_id: str) -> dict:
        """Process message through semantic discovery StateGraph"""
        with tracer.start_as_current_span("stategraph_orchestrator") as span:
            span.set_attribute("conversation_id", conversation_id)
            span.set_attribute("discovery_mode", "semantic-description-based")
            
            initial_state: AgentState = {
                "user_message": user_message,
                "conversation_id": conversation_id,
                "intent": "",
                "confidence": 0.0,
                "matched_agents": [],
                "messages": [],
                "agents_called": [],
                "agent_responses": [],
                "final_response": "",
                "should_end": False,
                "llm_matching_decision": None
            }
            
            final_state = await self.graph.ainvoke(initial_state)
            
            span.set_attribute("intent", final_state["intent"])
            span.set_attribute("matched_agents", ",".join(final_state.get("matched_agents", [])))
            span.set_attribute("agents_called", ",".join(final_state["agents_called"]))
            
            return {
                "response": final_state["final_response"],
                "intent": final_state["intent"],
                "confidence": final_state["confidence"],
                "matched_agents": final_state.get("matched_agents", []),
                "agents_called": final_state["agents_called"],
                "metadata": {
                    "conversation_id": conversation_id,
                    "graph_execution": "completed",
                    "llm_matching": final_state.get("llm_matching_decision"),
                    "discovery": "semantic-description-based",
                    "registry_url": REGISTRY_URL
                }
            }


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

async def main():
    """Test semantic discovery orchestrator"""
    
    logger.info("🧪 Testing MBTA StateGraph Orchestrator (Semantic Discovery)")
    
    try:
        orchestrator = StateGraphOrchestrator()
        
        # Run startup validation
        await orchestrator.startup_validation()
        
        # Test queries with semantic matching
        test_queries = [
            "Red Line delays?",  # Should match agent describing "alerts"
            "How do I get to Harvard?",  # Should match agent describing "route planning"
            "Find stops near me",  # Should match agent describing "stop search"
            "Bike route avoiding steep hills",  # Would match bike agent if registered
        ]
        
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print('='*70)
            
            result = await orchestrator.process_message(query, f"test-{hash(query)}")
            
            print(f"\n✅ Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            print(f"🎯 Matched Agents: {', '.join(result['matched_agents'])}")
            print(f"📞 Agents Called: {', '.join(result['agents_called'])}")
            print(f"🔍 Discovery: {result['metadata'].get('discovery')}")
            print(f"\n💬 Response:\n{result['response'][:200]}...")
    
    except RuntimeError as e:
        logger.error(f"❌ Startup failed: {e}")


if __name__ == "__main__":
    # Set up basic logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
