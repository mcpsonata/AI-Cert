import os
import sys
import json
import logging
import inspect
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from azure.identity import AzureCliCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.server import SQLEndpoint, Fabric, TabularEditor, CopilotDataEvaluator, AuthenticationManager
except ImportError as e:
    print(f"Error importing classes from src.server: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from root folder
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# PowerBI MCP Server wrapper
class PowerBIMCPServer:
    """Wrapper for the MCP Server specific to Power BI operations."""
    def __init__(self):
        try:
            logger.info("Initializing PowerBIMCPServer...")
            from mcp.server import Server
            from mcp.server.models import InitializationOptions
            
            # Configure the server
            options = InitializationOptions(
                tool_namespace="mcp",
                notification_options=None,
                server_name="Power BI MCP Server",  # Added required fields
                server_version="1.0.0",
                capabilities=[]
            )
            
            # Create the server instance
            self.server = Server(options)
            logger.info("PowerBIMCPServer initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import MCP modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PowerBIMCPServer: {e}")
            raise

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Azure OpenAI client using AzureCliCredential
from azure.identity import AzureCliCredential
from openai import AzureOpenAI

# Get Azure OpenAI credentials from environment variables
azure_endpoint = os.getenv("PROJECT_ENDPOINT")
azure_deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Initialize Azure OpenAI client with manual token acquisition
try:
    # Get the token manually from AzureCliCredential with the correct scope
    azure_credential = AzureCliCredential()
    # Get token for Azure OpenAI scope - using the correct audience for Azure AI services
    token = azure_credential.get_token("https://ai.azure.com/.default")
    
    # Use the token directly
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token=token.token,
        api_version=api_version
    )
    logger.info(f"Successfully initialized Azure OpenAI client with Azure CLI token for endpoint {azure_endpoint}")
except Exception as e:
    logger.error(f"Error initializing Azure OpenAI client: {e}")
    client = None
    raise RuntimeError(f"Failed to initialize Azure OpenAI client: {e}")

class MCPToolManager:
    """Manager for MCP tools that dynamically loads methods from different classes."""
    
    def __init__(self):
        """Initialize the tool manager with all necessary components."""
        self.available_tools = {}
        self.auth_manager = None
        self.sql_endpoint = None
        self.fabric = None
        self.tabular_editor = None
        self.copilot_evaluator = None
        self.mcp_server = None
        
        # Track connection state
        self.connected_dataset = None
        self.connected_workspace = None
        
        try:
            # Initialize core components
            self.auth_manager = AuthenticationManager()
            self.sql_endpoint = SQLEndpoint()
            self.fabric = Fabric()
            self.tabular_editor = TabularEditor()
            
            # Initialize evaluator after tabular_editor
            if self.tabular_editor and self.sql_endpoint:
                self.copilot_evaluator = CopilotDataEvaluator(self.tabular_editor, self.sql_endpoint)
            
            try:
                # Initialize MCP server connection
                from mcp.server import Server
                from mcp.types import ServerResult
                self.mcp_server = PowerBIMCPServer()
                logger.info("Successfully initialized MCP server connection")
            except Exception as mcp_err:
                logger.warning(f"Failed to initialize MCP server: {mcp_err}")
                self.mcp_server = None
            
            # Load all available tools
            self._initialize_tools()
            logger.info(f"MCP Tool Manager initialized with {len(self.available_tools)} tools")
            
        except Exception as e:
            logger.error(f"Error initializing MCP Tool Manager: {e}")
    
    def _initialize_tools(self) -> Dict[str, Dict]:
        """Initialize available MCP tools by fetching them dynamically from the MCP server."""
        try:
            # Get tools directly from TabularEditor server
            tools_from_server = self._fetch_tools_from_mcp_server()
            if tools_from_server:
                logger.info(f"Successfully loaded {len(tools_from_server)} tools from MCP server")
                return tools_from_server
            else:
                # Fallback to class method extraction if server method fails
                logger.warning("Failed to fetch tools from MCP server, falling back to class method extraction")
                return self._extract_tools_from_classes()
        except Exception as e:
            logger.error(f"Error initializing tools from server: {e}")
            # Fallback to class method extraction
            logger.info("Falling back to class method extraction")
            return self._extract_tools_from_classes()
    
    def _fetch_tools_from_mcp_server(self) -> Dict[str, Dict]:
        """Fetch available tools directly from the MCP server."""
        try:
            logger.info("Attempting to fetch tools from MCP server...")
            
            # Create a PowerBIMCPServer instance to access MCP tools
            try:
                if not self.mcp_server:
                    self.mcp_server = PowerBIMCPServer()
                
                # Access the MCP server's request handlers
                if hasattr(self.mcp_server, 'server') and hasattr(self.mcp_server.server, 'request_handlers'):
                    request_handlers = self.mcp_server.server.request_handlers
                    
                    # Look for ListToolsRequest handler
                    from mcp.types import ListToolsRequest
                    if ListToolsRequest in request_handlers:
                        # Get the handler function
                        handler = request_handlers[ListToolsRequest]
                        logger.info("Found ListToolsRequest handler")
                        
                        try:
                            # Call the handler with an empty ListToolsRequest
                            import asyncio
                            from mcp.types import ListToolsRequest
                            
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                # Create a proper request object with required fields
                                request = ListToolsRequest(method="tools/list")
                                
                                # Call the handler
                                result = loop.run_until_complete(handler(request))
                                
                                # Extract tools from the result
                                # Get the actual ListToolsResult from ServerResult
                                actual_result = result.root if hasattr(result, 'root') else result
                                
                                if hasattr(actual_result, 'tools'):
                                    tools_list = actual_result.tools
                                    
                                    # Convert MCP Tool objects to our dictionary format
                                    tools_dict = {}
                                    for tool in tools_list:
                                        tools_dict[tool.name] = {
                                            "description": tool.description,
                                            "parameters": self._convert_schema_to_params(tool.inputSchema),
                                            "required_params": tool.inputSchema.get("required", []) if tool.inputSchema else [],
                                            "optional_params": [param for param in tool.inputSchema.get("properties", {}) 
                                                              if param not in tool.inputSchema.get("required", [])] if tool.inputSchema else [],
                                            "source_class": "MCP_Server"
                                        }
                                    
                                    if tools_dict:
                                        logger.info(f"Successfully loaded {len(tools_dict)} tools from MCP server")
                                        self.available_tools = tools_dict
                                        return tools_dict
                                    else:
                                        logger.warning("MCP server returned empty tools list")
                                        return {}
                                else:
                                    logger.warning(f"Handler result doesn't have tools attribute: {actual_result}")
                                    return {}
                                    
                            finally:
                                if not loop.is_closed():
                                    loop.close()
                                    
                        except Exception as e:
                            logger.warning(f"Failed to call ListToolsRequest handler: {e}")
                            return {}
                    else:
                        logger.warning("MCP server doesn't have ListToolsRequest handler")
                        return {}
                else:
                    logger.warning("PowerBIMCPServer doesn't have expected request_handlers")
                    return {}
                    
            except Exception as e:
                logger.warning(f"Failed to create PowerBIMCPServer instance: {e}")
                return {}
            
        except Exception as e:
            logger.warning(f"Could not fetch tools from MCP server: {e}")
            return {}
    
    def _convert_schema_to_params(self, input_schema: Dict) -> Dict[str, str]:
        """Convert JSON schema to parameter descriptions."""
        if not input_schema or "properties" not in input_schema:
            return {}
        
        params = {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", f"{param_type} parameter")
            
            # Add required indicator
            if param_name in required:
                params[param_name] = f"{param_type} - {param_desc} (required)"
            else:
                params[param_name] = f"{param_type} - {param_desc} (optional)"
        
        return params
    
    def _extract_tools_from_classes(self) -> Dict[str, Dict]:
        """Extract tools from class methods as a fallback when server method fails."""
        try:
            # Collect tools from all available classes
            logger.info("Initializing tools from all available classes...")
            
            # Load tools from SQLEndpoint
            if self.sql_endpoint:
                self._extract_class_methods(
                    self.sql_endpoint, 
                    "SQLEndpoint", 
                    ["initialize_sql_connection", "get_sql_tables", "execute_sql_query", "get_sql_table_schema"]
                )
            
            # Load tools from Fabric
            if self.fabric:
                self._extract_class_methods(
                    self.fabric,
                    "Fabric",
                    ["get_workspace_info", "get_lakehouse_info", "refresh_sql_endpoint"]
                )
            
            # Load tools from TabularEditor
            if self.tabular_editor:
                self._extract_class_methods(
                    self.tabular_editor,
                    "TabularEditor",
                    [
                        "connect_dataset", "disconnect_dataset", "list_tables", 
                        "list_table_columns", "get_multiple_sql_tables_schema",
                        "generate_tmsl_columns", "select_tables_with_schema", 
                        "execute_dax_query", "check_date_table_exists", 
                        "get_column_properties", "get_measure_properties",
                        "get_table_properties", "list_all_relationships",
                        "_generate_erd_diagram"
                    ]
                )
            
            # Load tools from CopilotDataEvaluator
            if self.copilot_evaluator:
                self._extract_class_methods(
                    self.copilot_evaluator,
                    "CopilotDataEvaluator",
                    [
                        "dataset_columns_analysis", "evaluate_security_roles",
                        "evaluate_fact_and_dimension_analysis", "evaluate_table_linking"
                    ]
                )
            
            logger.info(f"Successfully loaded {len(self.available_tools)} tools")
            return self.available_tools
            
        except Exception as e:
            logger.error(f"Error initializing tools from classes: {e}")
            return {}
    
    def _extract_class_methods(self, obj: object, class_name: str, method_names: List[str]) -> None:
        """Extract methods from a class instance and convert them to tools."""
        for method_name in method_names:
            if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
                method = getattr(obj, method_name)
                
                # Skip if it starts with underscore (except explicitly included)
                if method_name.startswith('_') and method_name not in [
                    "_generate_erd_diagram"
                ]:
                    continue
                
                # Get method signature
                try:
                    signature = inspect.signature(method)
                    
                    # Get parameter info
                    parameters = {}
                    required_params = []
                    optional_params = []
                    
                    for param_name, param in signature.parameters.items():
                        # Skip 'self' parameter
                        if param_name == 'self':
                            continue
                            
                        param_info = {
                            "type": str(param.annotation).replace("<class '", "").replace("'>", ""),
                            "default": None if param.default is inspect.Parameter.empty else param.default,
                        }
                        
                        parameters[param_name] = param_info
                        
                        # Determine if parameter is required
                        if param.default is inspect.Parameter.empty:
                            required_params.append(param_name)
                        else:
                            optional_params.append(param_name)
                    
                    # Get method docstring
                    docstring = inspect.getdoc(method) or f"Execute {method_name} on {class_name}"
                    
                    # Add method as a tool
                    tool_name = f"{class_name.lower()}_{method_name}"
                    self.available_tools[tool_name] = {
                        "description": docstring,
                        "method": method,
                        "parameters": parameters,
                        "required_params": required_params,
                        "optional_params": optional_params,
                        "source_class": class_name,
                        "function": self._create_wrapper_function(obj, method)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error extracting method {method_name} from {class_name}: {e}")
    
    def _create_wrapper_function(self, obj: object, method: Callable) -> Callable:
        """Create a wrapper function for a method that handles arguments properly."""
        def wrapper(**kwargs):
            try:
                return method(**kwargs)
            except Exception as e:
                logger.error(f"Error executing {method.__name__}: {e}")
                raise e
        return wrapper
    
    def refresh_available_tools(self) -> bool:
        """Refresh the available tools from the MCP server."""
        try:
            logger.info("Refreshing available tools from MCP server...")
            new_tools = self._fetch_tools_from_mcp_server()
            
            if new_tools and len(new_tools) > len(self.available_tools):
                self.available_tools = new_tools
                logger.info(f"Successfully refreshed tools. Now have {len(self.available_tools)} tools available")
                return True
            elif new_tools:
                self.available_tools = new_tools
                logger.info(f"Tools refreshed. {len(self.available_tools)} tools available")
                return True
            else:
                logger.warning("Failed to refresh tools from MCP server")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing tools: {e}")
            return False
    
    def get_tool_count(self) -> int:
        """Get the current number of available tools."""
        return len(self.available_tools)
    
    def get_tools_description(self) -> str:
        """Get a formatted description of available tools for AI context."""
        tools_desc = "Available Power BI and Fabric tools:\n\n"
        for tool_name, tool_info in self.available_tools.items():
            tools_desc += f"🔧 {tool_name}:\n"
            tools_desc += f"   Description: {tool_info['description']}\n"
            if tool_info.get('parameters'):
                tools_desc += "   Parameters:\n"
                for param, desc in tool_info['parameters'].items():
                    tools_desc += f"     - {param}: {desc}\n"
            tools_desc += "\n"
        return tools_desc
    
    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        """Format tool result for presentation to the user."""
        try:
            if isinstance(result, (dict, list)):
                formatted_result = f"## Results from {tool_name}\n\n```json\n{json.dumps(result, indent=2, default=str)}\n```"
            elif isinstance(result, str) and (result.startswith('{') or result.startswith('[')):
                # Try to parse as JSON for better formatting
                try:
                    json_data = json.loads(result)
                    formatted_result = f"## Results from {tool_name}\n\n```json\n{json.dumps(json_data, indent=2, default=str)}\n```"
                except:
                    formatted_result = f"## Results from {tool_name}\n\n{result}"
            else:
                formatted_result = f"## Results from {tool_name}\n\n{result}"
            
            return formatted_result
        except Exception as e:
            logger.error(f"Error formatting tool result: {e}")
            return f"Results from {tool_name}: {str(result)}"
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific MCP tool with parameters through the PowerBIMCPServer."""
        try:
            logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
            
            # Special case for connect_dataset: check if already connected and disconnect first
            if tool_name == "tabulareditor_connect_dataset" and self.connected_dataset:
                logger.info("Already connected to a dataset, disconnecting first...")
                try:
                    # Try to disconnect first using the tabular_editor directly
                    if hasattr(self.tabular_editor, 'disconnect_dataset'):
                        self.tabular_editor.disconnect_dataset()
                        logger.info("Successfully disconnected from previous dataset")
                except Exception as disconnect_error:
                    logger.warning(f"Error during disconnect: {str(disconnect_error)}")
                
                # Reset connection state
                self.connected_workspace = None
                self.connected_dataset = None
            
            if tool_name not in self.available_tools:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found. Available tools: {list(self.available_tools.keys())}"
                }
            
            # Check if we have the MCP server available and try that first
            if self.mcp_server and hasattr(self.mcp_server, 'server') and hasattr(self.mcp_server.server, 'request_handlers'):
                try:
                    # Use the MCP server's call_tool handler
                    from mcp.types import CallToolRequest
                    
                    if CallToolRequest in self.mcp_server.server.request_handlers:
                        handler = self.mcp_server.server.request_handlers[CallToolRequest]
                        
                        # Create a CallToolRequest object
                        import asyncio
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # Create a proper request object with required fields
                            request = CallToolRequest(
                                method="tools/call",
                                params={
                                    "name": tool_name,
                                    "arguments": parameters
                                }
                            )
                            
                            # Call the handler
                            result = loop.run_until_complete(handler(request))
                            
                            # Extract the actual result
                            actual_result = result.root if hasattr(result, 'root') else result
                            
                            # Extract text from TextContent objects
                            if hasattr(actual_result, 'content') and actual_result.content:
                                result_text = actual_result.content[0].text
                                return {"success": True, "result": result_text}
                        finally:
                            loop.close()
                except Exception as mcp_error:
                    logger.warning(f"MCP server execution failed, falling back to direct method call: {mcp_error}")
                    # Continue to fallback method
            
            # Fallback to direct method call
            tool_info = self.available_tools[tool_name]
            
            # Check for required parameters
            missing_params = [param for param in tool_info.get('required_params', []) 
                             if param not in parameters]
            
            if missing_params:
                return {
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing_params)}"
                }
            
            # If the tool has a function property (from class method extraction), use that
            if 'function' in tool_info:
                result = tool_info['function'](**parameters)
                
                # Update connection state if this was a connect_dataset call
                if tool_name == "tabulareditor_connect_dataset" and isinstance(result, bool) and result:
                    self.connected_dataset = parameters.get("database_name")
                    self.connected_workspace = parameters.get("workspace_identifier")
                    logger.info(f"Updated connection state: workspace={self.connected_workspace}, dataset={self.connected_dataset}")
                
                return {
                    "success": True,
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": "Tool function not available"
                }
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def auto_detect_and_execute_tools(self, user_message: str) -> List[Dict[str, Any]]:
        """
        Automatically detect and execute appropriate tools based on user message.
        Uses GPT-4o to understand the user's intent and extract relevant parameters.
        Returns a list of executed tool results.
        """
        results = []
        
        try:
            # Create a structured list of available tools for GPT-4o
            tools_info = []
            for tool_name, tool_info in self.available_tools.items():
                tool_data = {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "required_params": tool_info["required_params"],
                    "optional_params": tool_info["optional_params"],
                    "source_class": tool_info["source_class"]
                }
                tools_info.append(tool_data)
            
            # Current connection state
            connection_state = {
                "connected_workspace": self.connected_workspace,
                "connected_dataset": self.connected_dataset
            }
            
            # Create the prompt for GPT-4o to analyze the user message
            messages = [
                {"role": "system", "content": f"""You are an assistant that helps identify Power BI and Fabric operations in user messages.
Your task is to analyze the user's message and determine which tool(s) should be executed and with what parameters.
Do not engage in conversation. Just return a structured JSON response with your analysis.

Available tools: {json.dumps(tools_info, indent=2)}

Current connection state: {json.dumps(connection_state, indent=2)}

SMART TOOL MAPPING:
When users mention these concepts, map them to specific tools:
1. SQL Endpoint, SQL connection, SQL database, connect to SQL, initialize SQL:
   - Map to "sqlendpoint_initialize_sql_connection" tool
   - Extract server/endpoint and database names from the message

2. Power BI dataset, semantic model, PBI model, dataset connection, connect to Power BI:
   - Map to "tabulareditor_connect_dataset" tool
   - Extract workspace and dataset names from the message

3. Show tables, list tables, get tables:
   - If in SQL context: map to "sqlendpoint_get_sql_tables"
   - If in Power BI context: map to "tabulareditor_list_tables"

4. Run query, execute query, execute DAX:
   - If DAX mentioned: map to "tabulareditor_execute_dax_query"
   - If SQL mentioned: map to "sqlendpoint_execute_sql_query"

EXAMPLES OF MAPPING:
1. "Connect to SQL endpoint example.sql.azuresynapse.net with database myDB" 
   → sqlendpoint_initialize_sql_connection with parameters {{"sql_endpoint": "example.sql.azuresynapse.net", "sql_database": "myDB"}}

2. "Connect to Power BI dataset Sales Analysis in workspace Marketing" 
   → tabulareditor_connect_dataset with parameters {{"workspace_identifier": "Marketing", "database_name": "Sales Analysis"}}

3. "Show me all tables in the current SQL connection" 
   → sqlendpoint_get_sql_tables

4. "Run this DAX query: EVALUATE VALUES(DimProduct)" 
   → tabulareditor_execute_dax_query with parameters {{"dax_query": "EVALUATE VALUES(DimProduct)"}}

5. "Can you initialize the SQL endpoint at myserver.database.windows.net with the AdventureWorks database?"
   → sqlendpoint_initialize_sql_connection with parameters {{"sql_endpoint": "myserver.database.windows.net", "sql_database": "AdventureWorks"}}

Be aggressive in parameter extraction - look for names, IDs, and contextual clues even if they're not explicitly labeled.
Infer parameter values from context when possible. For example:
- A GUID-like string is likely a workspace_identifier
- Words that sound like database or dataset names should be used for database_name
- SQL endpoint URLs often contain "sql.azuresynapse.net" or similar patterns

Return your response in the following JSON format:
{{
  "tools_to_execute": [
    {{
      "tool_name": "name_of_tool_to_execute",
      "params": {{
        "param1": "value1",
        "param2": "value2"
      }},
      "confidence": 0.9,
      "explanation": "Why this tool should be executed"
    }}
  ]
}}

If no tools should be executed, return an empty list for tools_to_execute.
Do not include any tool that doesn't have all required parameters.
Focus on extracting explicit information from the message, like GUIDs, table names, etc.
For any GUID in the message, consider if it might be a workspace_identifier or database_name.
"""},
                {"role": "user", "content": user_message}
            ]
            
            # Call Azure OpenAI to analyze the message
            response = client.chat.completions.create(
                model=azure_deployment,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=500
            )
            
            tool_analysis = json.loads(response.choices[0].message.content)
            logger.info(f"Tool analysis: {tool_analysis}")
            
            # Execute the tools based on the analysis
            if "tools_to_execute" in tool_analysis and tool_analysis["tools_to_execute"]:
                for tool_execution in tool_analysis["tools_to_execute"]:
                    tool_name = tool_execution.get("tool_name")
                    params = tool_execution.get("params", {})
                    confidence = tool_execution.get("confidence", 0)
                    explanation = tool_execution.get("explanation", "")
                    
                    # Only execute if we have sufficient confidence (0.6 or higher)
                    if tool_name in self.available_tools and confidence >= 0.6:
                        logger.info(f"Auto-executing tool: {tool_name} with params: {params}")
                        logger.info(f"Reason: {explanation} (confidence: {confidence})")
                        
                        # Execute the tool
                        execution_result = self.execute_tool(tool_name, params)
                        
                        # Update connection state if this was a successful connect_dataset call
                        if tool_name == "tabulareditor_connect_dataset" and execution_result.get("success", False):
                            self.connected_workspace = params.get("workspace_identifier")
                            self.connected_dataset = params.get("database_name")
                        
                        # Add to results
                        results.append({
                            "tool_name": tool_name,
                            "params": params,
                            "confidence": confidence,
                            "explanation": explanation,
                            "result": execution_result
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in GPT-based tool detection: {e}")
            return []

# Initialize the MCP Tool Manager
tool_manager = MCPToolManager()

# Add helper method for enhanced chat
def get_enhanced_chat_response(message: str, user_id: str, conversation_id: str, conversation_history=None, progress_callback=None) -> str:
    """Get AI response with MCP tools context and conversation history."""
    try:
        # Use provided conversation history or start with empty list
        if conversation_history is None:
            conversation_history = []
        
        # Always provide tools context - let GPT-4o decide when to use them
        if progress_callback:
            progress_callback("Preparing tools context...")
        
        system_message = f"""You are a helpful AI assistant with access to Power BI and Microsoft Fabric tools. 
        You can help users with Power BI operations, DAX queries, semantic model management, and Fabric lakehouse integration.
        
        IMPORTANT: You have conversation history and should maintain context between messages. Reference previous operations and continue conversations naturally.

        LEVERAGE YOUR BUSINESS INTELLIGENCE:
        - Use your extensive knowledge of Power BI, DAX, and Microsoft Fabric to make intelligent decisions.
        - Apply your understanding of typical business data structures and naming patterns.
        - Recognize common scenarios: customer searches, sales analysis, account management, etc.
        - Use your knowledge of standard table relationships and data warehouse patterns.
        
        MICROSOFT BUSINESS DOMAIN EXPERTISE:
        - Apply your knowledge of Microsoft's business model: partners, opportunities, subsidiaries, sales processes.
        - Understand Microsoft business terminology: CSP partners, ISVs, OEMs, enterprise customers.
        - Recognize Microsoft sales funnel: leads → opportunities → deals → revenue.
        - Know Microsoft organizational structure: subsidiaries, business units, geographical divisions.
        - Understand partner ecosystem relationships and revenue attribution models.
        - Apply knowledge of Microsoft finance processes: billing, revenue recognition, subsidiary reporting.

        {tool_manager.get_tools_description()}

        SMART TOOL DETECTION:
        - When users ask to "connect to SQL" or "initialize SQL endpoint" - automatically use the sqlendpoint_initialize_sql_connection tool.
        - When users mention "connect to Power BI dataset", "connect to semantic model", or similar - use the tabulareditor_connect_dataset tool.
        - For "list tables" or "show tables" requests - use sqlendpoint_get_sql_tables for SQL or tabulareditor_list_tables for Power BI.
        - For "run query" or "execute query" - select the appropriate tool based on context (SQL or DAX).
        - Be proactive in suggesting tools when users describe what they want to do, even if they don't know the exact tool name.
        
        EXECUTION STYLE:
        - Be DIRECT and ACTION-ORIENTED. When users ask for something, DO IT immediately.
        - Use your intelligence to determine the best approach based on the user's intent.
        - If you need information to answer a question, get it. Don't ask permission.
        - Execute tools in the most logical sequence to fulfill the user's request.

        PROBLEM-SOLVING APPROACH:
        - Analyze the user's intent and choose the appropriate tools to achieve their goal.
        - If one approach fails, automatically try alternative approaches.
        - Use context from conversation history to make better decisions.
        - Apply your business knowledge to understand the user's needs and provide relevant insights.
        
        TOOL USAGE GUIDELINES:
        - For data searches: Get the relevant table structure first, then query appropriately.
        - For explorations: Use the most direct path to get the information requested.
        - For errors: Automatically troubleshoot and retry with corrected approaches.
        - Use your DAX and Power BI expertise to write effective queries.

        When a user asks for Power BI operations that require tool execution, respond ONLY with:

        TOOL_EXECUTION:
        {{
            "tool": "tool_name",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}

        Do NOT include any explanatory text before the TOOL_EXECUTION block. The tool results will be automatically formatted and presented to the user.
        
        Use your intelligence to select the right tools and sequence to fulfill any request efficiently.
        """
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        for msg in conversation_history:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add the current user message
        messages.append({"role": "user", "content": message})
        
        if progress_callback:
            progress_callback("Generating AI response...")
        
        # Response is generated with Azure OpenAI client
        try:
            response = client.chat.completions.create(
                model=azure_deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            ).choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error calling OpenAI API: {e}"
        
        # Check if the response contains a tool execution request
        if "TOOL_EXECUTION:" in response:
            if progress_callback:
                progress_callback("Tool execution requested...")
            
            # Remove any text before TOOL_EXECUTION and only keep the clean result
            tool_part = response.split("TOOL_EXECUTION:")[1].strip()
            try:
                # Extract JSON from the response
                json_start = tool_part.find('{')
                json_end = tool_part.rfind('}') + 1
                tool_json = tool_part[json_start:json_end]
                
                tool_request = json.loads(tool_json)
                tool_name = tool_request.get("tool")
                parameters = tool_request.get("parameters", {})
                
                if progress_callback:
                    progress_callback(f"Executing {tool_name}...")
                
                # Execute the tool
                tool_result = tool_manager.execute_tool(tool_name, parameters)
                
                # Replace the entire response with just the clean tool results
                if tool_result.get("success"):
                    # Format the result
                    result = tool_result.get("result", "")
                    if isinstance(result, (dict, list)):
                        formatted_result = f"## Results from {tool_name}\n\n```json\n{json.dumps(result, indent=2, default=str)}\n```"
                    elif isinstance(result, str) and (result.startswith('{') or result.startswith('[')):
                        # Try to parse as JSON for better formatting
                        try:
                            json_data = json.loads(result)
                            formatted_result = f"## Results from {tool_name}\n\n```json\n{json.dumps(json_data, indent=2, default=str)}\n```"
                        except:
                            formatted_result = f"## Results from {tool_name}\n\n{result}"
                    else:
                        formatted_result = f"## Results from {tool_name}\n\n{result}"
                    
                    # Return only the formatted result, no explanatory text
                    response = formatted_result
                    if progress_callback:
                        progress_callback("Tool execution completed successfully")
                else:
                    response = f"❌ Error: {tool_result.get('error', 'Unknown error')}"
                    if progress_callback:
                        progress_callback("Tool execution failed")
                    
            except json.JSONDecodeError:
                response += "\n\n⚠️ Could not parse tool execution request. Please check the format."
                if progress_callback:
                    progress_callback("Tool parsing failed")
            except Exception as e:
                response += f"\n\n❌ Tool execution error: {str(e)}"
                if progress_callback:
                    progress_callback(f"Tool execution error: {str(e)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced chat response: {e}")
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return f"I encountered an error: {str(e)}"

# Use the tools from the tool manager
MCP_TOOLS = {}

# Convert the tool manager's tools to the format expected by the app
for name, info in tool_manager.available_tools.items():
    MCP_TOOLS[name] = {
        "description": info["description"],
        "required_params": info["required_params"],
        "optional_params": info["optional_params"],
        "function": info.get("function")
    }

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html', tools=MCP_TOOLS)

# Global dictionary to store conversation history by session_id
conversation_sessions = {}

@app.route('/api/chat', methods=['POST', 'GET'])
def chat():
    """API endpoint for chat"""
    if request.method == 'GET':
        return jsonify({
            'message': 'This is the chat API endpoint. Please use POST method with a JSON body containing a "message" field.',
            'example': {
                'message': 'Your question or command here',
                'tool_name': '(Optional) Specific tool to execute',
                'tool_params': {},
                'session_id': '(Optional) Session identifier for conversation continuity'
            }
        })
    
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or create session_id
        session_id = data.get('session_id', 'default_session')
        logger.info(f"Processing message for session: {session_id}")
        logger.info(f"User message: {user_message}")
        
        # Create a progress tracker
        progress_updates = []
        def track_progress(message):
            progress_updates.append(message)
            logger.info(f"Progress: {message}")
        
        # Initialize empty lists for tool results
        tool_results = []  # Store all tool results to return to the client
        executed_tool_names = []
        
        # Retrieve existing conversation history or create a new one
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
            logger.info(f"Created new conversation session: {session_id}")
        
        conversation_history = conversation_sessions[session_id]
        logger.info(f"Current conversation history length: {len(conversation_history)}")
        
        # Auto-detect and execute tools based on the user message
        auto_results = tool_manager.auto_detect_and_execute_tools(user_message)
        
        # Add any auto-executed tool results to the chat history
        if auto_results:
            for tool_execution in auto_results:
                tool_name = tool_execution.get("tool_name")
                params = tool_execution.get("params", {})
                explanation = tool_execution.get("explanation", "")
                confidence = tool_execution.get("confidence", 0)
                result = tool_execution.get("result", {})
                
                if result and result.get("success", False):
                    tool_result = result.get("result")  # The actual result of execution
                    executed_tool_names.append(tool_name)
                    
                    # Format the result to be more readable
                    formatted_result = ""
                    try:
                        if isinstance(tool_result, (dict, list)):
                            formatted_result = json.dumps(tool_result, indent=2, default=str)
                        else:
                            formatted_result = str(tool_result)
                    except:
                        formatted_result = str(tool_result)
                    
                    # Add formatted result to the list of results
                    tool_results.append({
                        "tool_name": tool_name,
                        "params": params,
                        "result": formatted_result,
                        "explanation": explanation,
                        "confidence": confidence
                    })
                    
                    # Add the tool execution and result to the chat history
                    conversation_history.append({
                        "role": "assistant", 
                        "content": f"I've automatically executed the '{tool_name}' tool with parameters: {json.dumps(params)}.\n\nReason: {explanation}"
                    })
                    
                    conversation_history.append({
                        "role": "system", 
                        "content": f"Tool result:\n\n{formatted_result}"
                    })
        # Check if we need to execute a manually selected tool
        tool_name = data.get('tool_name')
        tool_params = data.get('tool_params', {})
        
        if tool_name and tool_name in MCP_TOOLS and tool_name not in executed_tool_names:
            tool_info = MCP_TOOLS[tool_name]
            
            # Check if all required parameters are provided
            missing_params = [param for param in tool_info.get('required_params', []) 
                             if param not in tool_params or not tool_params[param]]
            
            if missing_params:
                return jsonify({
                    'error': f"Missing required parameters: {', '.join(missing_params)}",
                    'missing_params': missing_params
                }), 400
            
            try:
                # Execute the tool using the tool manager
                logger.info(f"Executing manually selected tool: {tool_name} with params: {tool_params}")
                result = tool_manager.execute_tool(tool_name, tool_params)
                
                if result["success"]:
                    tool_result = result["result"]
                    executed_tool_names.append(tool_name)
                    
                    # Format the result to be more readable
                    formatted_result = ""
                    try:
                        if isinstance(tool_result, (dict, list)):
                            formatted_result = json.dumps(tool_result, indent=2, default=str)
                        else:
                            formatted_result = str(tool_result)
                    except:
                        formatted_result = str(tool_result)
                    
                    # Add formatted result to the list of results
                    tool_results.append({
                        "tool_name": tool_name,
                        "params": tool_params,
                        "result": formatted_result,
                        "explanation": "Manually selected tool execution",
                        "confidence": 1.0
                    })
                    
                    # Add the tool execution and result to the chat history
                    conversation_history.append({
                        "role": "assistant", 
                        "content": f"I'm executing the '{tool_name}' tool with parameters: {json.dumps(tool_params)}"
                    })
                    
                    conversation_history.append({
                        "role": "system", 
                        "content": f"Tool result:\n\n{formatted_result}"
                    })
                else:
                    logger.error(f"Tool execution failed: {result['error']}")
                    return jsonify({'error': result['error']}), 400
            
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                return jsonify({'error': f"Tool execution error: {str(e)}"}), 500
        
        # Add the current user message to history
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Call get_enhanced_chat_response to generate the response
        try:
            # Use the enhanced chat response function with MCP tool integration
            assistant_message = get_enhanced_chat_response(
                message=user_message,
                user_id="web_user",
                conversation_id=session_id,
                conversation_history=conversation_history,
                progress_callback=track_progress
            )
            
            # Add the assistant's response to the conversation history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Save the updated conversation history
            conversation_sessions[session_id] = conversation_history
            
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {str(e)}")
            assistant_message = f"I couldn't generate a response using Azure OpenAI API. Error: {str(e)}"
        
        # Return the assistant's response along with any tool results
        return jsonify({
            'message': assistant_message,
            'tool_results': tool_results,
            'auto_executed_tools': executed_tool_names,
            'progress': progress_updates,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

@app.route('/api/tools', methods=['GET'])
def get_tools():
    """API endpoint to get the list of available tools"""
    tools_info = {}
    for name, info in tool_manager.available_tools.items():
        tools_info[name] = {
            "description": info["description"],
            "required_params": info["required_params"],
            "optional_params": info["optional_params"],
            "source_class": info["source_class"]
        }
    return jsonify(tools_info)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check and tool status"""
    status = {
        "status": "healthy",
        "tool_count": len(tool_manager.available_tools),
        "components": {
            "auth_manager": tool_manager.auth_manager is not None,
            "sql_endpoint": tool_manager.sql_endpoint is not None,
            "fabric": tool_manager.fabric is not None,
            "tabular_editor": tool_manager.tabular_editor is not None,
            "copilot_evaluator": tool_manager.copilot_evaluator is not None,
            "azure_openai": client is not None
        },
        "azure_openai": {
            "endpoint_configured": azure_endpoint is not None,
            "deployment": azure_deployment,
            "api_version": api_version
        },
        "tools_by_class": {}
    }
    
    # Count tools by class
    class_counts = {}
    for name, info in tool_manager.available_tools.items():
        class_name = info["source_class"]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    status["tools_by_class"] = class_counts
    
    return jsonify(status)

@app.route('/api/sessions/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Get conversation history for a specific session"""
    if session_id not in conversation_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    history = conversation_sessions[session_id]
    return jsonify({
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    })

if __name__ == '__main__':
    print("Starting Power BI Assistant on http://localhost:5000")
    app.run(debug=True, port=5000)
