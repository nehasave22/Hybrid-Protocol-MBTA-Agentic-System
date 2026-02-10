"""
SLIM-enabled StopFinder Agent Server
Uses A2A SDK + SLIM transport from agntcy-app-sdk
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

sys.path.insert(0, '/opt/mbta-agents')

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, Message
from dotenv import load_dotenv
import uvicorn
import httpx

logger = logging.getLogger(__name__)


class StopFinderAgentExecutor(AgentExecutor):
    """Executor that handles stop finding requests"""
    
    def __init__(self, mbta_api_key: str):
        self.mbta_api_key = mbta_api_key
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Handle incoming stop search requests"""
        try:
            # Get message text - handle Part(root=TextPart) structure
            message_text = ""
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    message_text = part.root.text
                    break
                elif hasattr(part, 'text'):
                    message_text = part.text
                    break
            
            logger.info(f"📨 StopFinder Agent received: {message_text}")
            
            # Fetch stops from MBTA API (no location_type filter to get all)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api-v3.mbta.com/stops",
                    params={
                        "api_key": self.mbta_api_key,
                        # Removed filter[location_type] to get all stop types
                        "page[limit]": "100"  # Increased limit
                    },
                    timeout=10.0
                )
                stops_data = response.json().get("data", [])
            
            # Extract search terms from query (remove common words and punctuation)
            import string
            
            # Remove punctuation and split
            message_clean = message_text.translate(str.maketrans('', '', string.punctuation))
            
            common_words = {'find', 'the', 'nearest', 'station', 'to', 'near', 'search', 'for', 'show', 'me', 'where', 'is'}
            query_words = set(message_clean.lower().split()) - common_words
            
            logger.info(f"🔍 Search terms: {query_words}")
            
            # Filter by query - bidirectional matching
            matching_stops = []
            for stop in stops_data:
                stop_name = stop.get("attributes", {}).get("name", "").lower()
                
                # Check if ANY query word appears in stop name OR stop name word in query
                for query_word in query_words:
                    if len(query_word) >= 3:  # Ignore very short words
                        if query_word in stop_name or any(stop_word in query_word for stop_word in stop_name.split()):
                            matching_stops.append(stop)
                            break
            
            # Remove duplicates while preserving order
            seen = set()
            unique_stops = []
            for stop in matching_stops:
                stop_id = stop.get("id")
                if stop_id not in seen:
                    seen.add(stop_id)
                    unique_stops.append(stop)
            
            # Format response
            if not unique_stops:
                text = f"❌ No stops found matching '{message_text}'"
            else:
                text = f"🚉 Found {len(unique_stops)} stops:\n"
                for i, stop in enumerate(unique_stops[:5], 1):
                    name = stop.get("attributes", {}).get("name", "Unknown")
                    text += f"{i}. {name}\n"
            
            # Send response
            from a2a.types import TextPart
            from uuid import uuid4
            response_message = Message(
                message_id=str(uuid4()),
                parts=[TextPart(text=text)],
                role="agent"
            )
            await event_queue.enqueue_event(response_message)
            
            logger.info(f"✅ Stop list sent via SLIM")
            
        except Exception as e:
            logger.error(f"❌ Error in stopfinder executor: {e}", exc_info=True)
            from a2a.types import TextPart
            from uuid import uuid4
            error_message = Message(
                message_id=str(uuid4()),
                parts=[TextPart(text=f"Error: {str(e)}")],
                role="agent"
            )
            await event_queue.enqueue_event(error_message)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle cancellation"""
        raise NotImplementedError("Cancellation not supported for stopfinder agent")


def main():
    """Start StopFinder A2A server with SLIM support"""
    load_dotenv()
    
    mbta_api_key = os.getenv("MBTA_API_KEY", "")
    
    # Define skill
    skill = AgentSkill(
        id="mbta_stop_finder",
        name="MBTA Stop Finder",
        description="Finds MBTA stations and stops by name, location, or proximity",
        tags=["stops", "stations", "search", "mbta"],
        examples=["Find Park Street station", "Stops near Harvard", "Where is Kendall?"]
    )
    
    # Define agent card
    agent_card = AgentCard(
        name="mbta-stops",
        description="Finds MBTA stations and stops by name, location, or proximity. Provides stop details and accessibility info.",
        url="http://96.126.111.107:50053/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[skill],
        capabilities=AgentCapabilities(streaming=True)
    )
    
    # Create agent executor
    agent_executor = StopFinderAgentExecutor(mbta_api_key)
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )
    
    # Create A2A server
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    # Build ASGI app
    app = server.build()
    
    logger.info("🚀 Starting StopFinder Agent with A2A+SLIM on port 50053")
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=50053,
        log_level="info"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()