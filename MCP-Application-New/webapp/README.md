# Power BI Assistant

A Flask web application with a chatbot interface that uses OpenAI's GPT-4o model to help users interact with Power BI datasets using MCP tools.

## Project Structure

```
webapp/
├── app.py                 # Main Flask application with MCP integration
├── simple_app.py          # Simplified version with mock MCP tools
├── run.bat                # Batch script to run the full application
├── run_simple.bat         # Batch script to run the simplified application
├── static/
│   └── style.css          # Additional CSS styles
└── templates/
    └── index.html         # HTML template for the chat interface
```

## Features

- Interactive chat interface for querying Power BI datasets
- Integration with Power BI tabular models through MCP tools
- Powered by OpenAI's GPT-4o model
- Direct execution of MCP tools from the chat interface

## Prerequisites

- Python 3.8 or higher
- Flask
- OpenAI Python client
- Power BI datasets with proper credentials
- Azure Key Vault access for secure credential storage

## Setup

1. Make sure you have all the necessary dependencies installed:
   ```bash
   pip install flask flask-cors openai python-dotenv
   ```

2. The application uses the `.env` file in the root folder which contains:
   ```
   OPENAI_API_KEY=your_openai_api_key_here (if needed)
   AZURE_TENANT_ID=your_tenant_id_here
   AZURE_CLIENT_ID=your_client_id_here
   Key_Vault_Name=your_key_vault_name_here
   Secret_Name=your_secret_name_here
   ```

## Running the Application

### Full Version (requires Azure Authentication)

1. Configure the `.env` file with your Azure credentials
2. Run the application using:
   ```bash
   python app.py
   ```
   or double-click the `run.bat` file

### Simplified Version (with Mock MCP Tools)

For testing without Azure authentication:

1. Run the simplified version:
   ```bash
   python simple_app.py
   ```
   or double-click the `run_simple.bat` file

2. Open your browser and go to `http://localhost:5000`

## Usage

1. Type your question or request in the chat box
2. The assistant will respond with information or suggest using specific MCP tools
3. You can also directly use tools by clicking on them in the sidebar
4. Enter the required parameters for the tool and execute it

## Available MCP Tools

In the full version, these tools connect to actual Power BI datasets:

- **connect_dataset**: Connect to a Power BI dataset
- **disconnect_dataset**: Disconnect from the current dataset
- **list_tables**: List all tables in the connected model
- **execute_dax_query**: Execute a DAX query on the connected dataset
- **get_table_properties**: Get properties of a specific table
- **get_column_properties**: Get properties of a specific column
- **get_measure_properties**: Get properties of a specific measure
- **check_date_table_exists**: Check if a date table exists in the model
- **list_all_relationships**: List all relationships in the model

The simplified version includes mock implementations of these tools for demonstration purposes.

## Example Queries

- "Connect to my Power BI dataset"
- "List all the tables in my dataset"
- "What relationships exist between tables?"
- "Execute a DAX query to count rows in the Sales table"
- "Tell me about the properties of the Customer table"

## Notes

- You need a valid OpenAI API key to use the GPT-4o functionality
- The full version requires Azure authentication for accessing Power BI datasets
- This application is for demonstration purposes and not intended for production use
