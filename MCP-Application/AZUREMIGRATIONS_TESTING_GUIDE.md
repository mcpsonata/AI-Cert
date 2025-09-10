# AzureMigrations Lakehouse Testing Guide: Real-World Query Scenarios

## Overview
This guide provides specific, realistic questions based on the **AzureMigrations lakehouse** in **MSXI-Azure02-Dev** workspace. These questions are designed to test your Power BI MCP Application's ability to handle real Azure migration analysis scenarios using actual data structures.

## 🎯 Testing Context

### **Target Environment**:
- **Workspace**: MSXI-Azure02-Dev  
- **Lakehouse**: AzureMigrations
- **Data Focus**: Azure migration assessment and planning data
- **Use Case**: Enterprise cloud migration analysis

### **Expected Data Domain**:
Based on Azure Migrations context, the lakehouse likely contains:
- Server inventory and assessment data
- Application dependency mappings  
- Migration readiness scores
- Cost analysis and projections
- Performance metrics and sizing recommendations

## 🔍 Category 1: Migration Assessment & Planning

### **Migration Readiness Analysis Questions**

**Question 1: Server Inventory Assessment**
> "I need to analyze our current server inventory in the AzureMigrations lakehouse. Show me all servers grouped by operating system type, include their current resource utilization levels, and identify which servers have the highest migration readiness scores. I want to prioritize our migration waves based on this data."

**Expected Chatbot Behavior:**
- Connects to MSXI-Azure02-Dev workspace and AzureMigrations lakehouse
- Identifies server inventory tables
- Performs OS-based grouping analysis
- Handles resource utilization calculations
- Implements readiness score ranking

**Question 2: Application Dependency Mapping**
> "Help me understand the application dependencies for our critical business applications. I need to see which servers host interdependent applications, map their communication patterns, and identify potential migration blockers due to complex dependencies. Focus on applications marked as 'business critical' in our assessment data."

**Expected Chatbot Behavior:**
- Identifies application and dependency tables
- Performs complex join operations for dependency mapping
- Filters for business-critical applications
- Analyzes communication patterns between servers
- Identifies migration complexity factors

**Question 3: Migration Wave Planning**
> "I want to create migration waves based on application dependencies and server readiness. Group servers into logical migration waves where dependent systems are migrated together, prioritize by business impact, and ensure each wave doesn't exceed 50 servers. Show me the recommended timeline based on our assessment data."

**Expected Chatbot Behavior:**
- Performs complex dependency analysis
- Implements grouping logic for wave planning
- Applies business impact prioritization
- Handles capacity constraints (50 servers per wave)
- Suggests timeline based on data patterns

## 📊 Category 2: Cost Analysis & Optimization

### **Azure Cost Projection Questions**

**Question 4: Current vs. Projected Cost Analysis**
> "Compare our current on-premises infrastructure costs with projected Azure costs for all servers in the AzureMigrations data. Break this down by server type, include different Azure VM SKU recommendations, and show me potential cost savings by region. I need this analysis for our business case presentation."

**Expected Chatbot Behavior:**
- Accesses cost analysis tables in the lakehouse
- Performs current vs. projected cost calculations  
- Groups analysis by server types and regions
- Handles multiple VM SKU comparisons
- Calculates potential savings percentages

**Question 5: Right-Sizing Recommendations**
> "Analyze the performance data for all our servers and identify right-sizing opportunities. Show me servers that are over-provisioned based on their actual CPU, memory, and storage utilization patterns over the last 6 months. Calculate potential cost savings from right-sizing these resources in Azure."

**Expected Chatbot Behavior:**
- Analyzes historical performance metrics
- Identifies over-provisioned resources
- Performs utilization pattern analysis over time periods
- Calculates right-sizing opportunities
- Quantifies cost reduction potential

**Question 6: Total Cost of Ownership (TCO) Analysis**
> "I need a comprehensive TCO analysis comparing our 3-year on-premises costs versus Azure migration costs. Include infrastructure, licensing, maintenance, and operational costs. Factor in our existing Azure credits and enterprise agreements. Show me the break-even point and long-term savings projection."

**Expected Chatbot Behavior:**
- Performs multi-dimensional cost analysis
- Handles 3-year projection calculations
- Incorporates licensing and operational costs
- Factors in Azure credits and enterprise agreements
- Calculates break-even analysis

## 🛠️ Category 3: Technical Migration Planning

### **Infrastructure Assessment Questions**

**Question 7: Network and Storage Migration Planning**
> "Analyze the network connectivity requirements and storage patterns for our servers in the migration data. Identify servers that require ExpressRoute connectivity, calculate total storage migration requirements by storage type, and recommend Azure storage solutions based on current IOPS and throughput patterns."

**Expected Chatbot Behavior:**
- Analyzes network connectivity requirements
- Identifies ExpressRoute candidates
- Performs storage analysis by type and performance
- Maps current patterns to Azure storage solutions
- Provides IOPS and throughput recommendations

**Question 8: Security and Compliance Assessment**
> "Review our server inventory for security and compliance requirements. Identify servers handling sensitive data that need additional security controls in Azure, map compliance requirements to Azure security services, and highlight any servers that might need special migration considerations due to regulatory requirements."

**Expected Chatbot Behavior:**
- Identifies security-sensitive servers
- Maps compliance requirements to Azure services
- Highlights regulatory considerations
- Suggests appropriate Azure security controls
- Identifies special migration requirements

**Question 9: Backup and Disaster Recovery Planning**
> "Analyze our current backup and DR configurations from the assessment data. Design Azure backup strategies based on current RTO/RPO requirements, identify servers needing Azure Site Recovery, and calculate the infrastructure requirements for maintaining our DR capabilities post-migration."

**Expected Chatbot Behavior:**
- Analyzes current backup configurations
- Maps RTO/RPO requirements to Azure solutions
- Identifies Site Recovery candidates
- Calculates DR infrastructure needs
- Designs backup strategy recommendations

## 📈 Category 4: Performance and Monitoring

### **Performance Analysis Questions**

**Question 10: Performance Trend Analysis**
> "Examine the performance trends for our top 20 resource-intensive servers over the past year. Identify seasonal patterns, peak usage periods, and performance bottlenecks. Use this analysis to recommend appropriate Azure VM sizes and auto-scaling configurations for these workloads."

**Expected Chatbot Behavior:**
- Identifies top resource-intensive servers
- Performs year-long trend analysis
- Detects seasonal and peak usage patterns
- Identifies performance bottlenecks
- Recommends Azure VM sizing and auto-scaling

**Question 11: Application Performance Impact Assessment**
> "Analyze how migration to Azure might impact application performance for our critical workloads. Compare current response times and throughput with projected Azure performance, identify applications that might benefit from Azure-native services, and flag any that might experience performance degradation."

**Expected Chatbot Behavior:**
- Analyzes current application performance metrics
- Performs migration impact assessment
- Compares current vs. projected Azure performance
- Identifies Azure-native service opportunities
- Flags potential performance risks

**Question 12: Resource Optimization Opportunities**
> "Identify optimization opportunities in our current environment that we should implement before migration. Look for unused resources, oversized instances, inefficient storage allocations, and applications that could benefit from modernization. Quantify the impact of addressing these issues pre-migration."

**Expected Chatbot Behavior:**
- Identifies unused and oversized resources
- Analyzes storage allocation efficiency
- Suggests modernization opportunities
- Quantifies pre-migration optimization impact
- Prioritizes optimization activities

## 🔄 Category 5: Migration Execution & Tracking

### **Migration Progress Questions**

**Question 13: Migration Status Dashboard Data**
> "Create a comprehensive view of our migration progress using the latest data in AzureMigrations. Show me completion rates by migration wave, identify any servers experiencing migration issues, track actual vs. planned migration timelines, and highlight upcoming migration milestones."

**Expected Chatbot Behavior:**
- Analyzes migration progress data
- Calculates completion rates by wave
- Identifies migration issues and delays
- Compares actual vs. planned timelines
- Highlights upcoming milestones

**Question 14: Post-Migration Validation Analysis**
> "For servers that have already been migrated, validate their performance in Azure against pre-migration baselines. Identify any performance regressions, validate that applications are functioning correctly, and measure actual vs. projected Azure costs for migrated workloads."

**Expected Chatbot Behavior:**
- Compares post-migration performance data
- Identifies performance regressions
- Validates application functionality metrics
- Compares actual vs. projected costs
- Provides migration success validation

**Question 15: Risk and Issue Tracking**
> "Analyze migration risks and issues from our tracking data. Identify servers with repeated migration failures, categorize issues by root cause, track resolution times, and predict which upcoming migrations might face similar challenges based on historical patterns."

**Expected Chatbot Behavior:**
- Analyzes migration failure patterns
- Categorizes issues by root cause
- Tracks issue resolution metrics
- Performs predictive risk analysis
- Identifies potential future challenges

## 🧪 Context Retention & Advanced Scenarios

### **Multi-Turn Conversation Tests**

**Question 16: Building on Previous Analysis**
> "Based on our earlier analysis of server migration readiness, now I want to focus specifically on the servers you identified as 'high priority' for migration. Can you dive deeper into their dependencies and create a detailed migration plan just for these servers?"

**Question 17: Cross-Analysis Integration**
> "Combine the cost analysis we did earlier with the performance trends you showed me. I need to identify servers where the cost savings justify potential performance trade-offs, and vice versa - where performance requirements justify higher Azure costs."

**Question 18: Dynamic Filtering and Refinement**
> "Take the migration wave plan we created and now filter it to only include Windows servers, adjust the timeline to account for a 2-week holiday freeze in December, and recalculate the resource requirements for each wave."

## 📊 Testing Protocol for AzureMigrations Data

### **Pre-Test Setup**
```powershell
# Initialize testing session
$testSession = "azuremigrations-test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Write-Host "Starting AzureMigrations testing session: $testSession"

# Baseline connection test
$connectionTest = @{
    message = "Connect to the AzureMigrations lakehouse in MSXI-Azure02-Dev workspace and show me what tables are available."
    session_id = $testSession
} | ConvertTo-Json

$baselineResponse = Invoke-RestMethod -Uri "http://localhost:5000/api/chat-optimized" -Method POST -ContentType "application/json" -Body $connectionTest

Write-Host "Connection established. Available tables detected: $($baselineResponse.response -match 'table|schema' ? 'Yes' : 'No')"
```

### **Systematic Testing Approach**

```powershell
function Test-AzureMigrationsQuery {
    param(
        [string]$QuestionText,
        [string]$QuestionNumber,
        [string]$ExpectedBehavior
    )
    
    # Pre-query metrics
    $preTokens = (Invoke-RestMethod -Uri "http://localhost:5000/api/token-usage").total_usage.total_tokens
    
    # Execute query
    $queryBody = @{
        message = $QuestionText
        session_id = $testSession
    } | ConvertTo-Json
    
    $startTime = Get-Date
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/chat-optimized" -Method POST -ContentType "application/json" -Body $queryBody
    $endTime = Get-Date
    
    # Post-query metrics
    $postTokens = (Invoke-RestMethod -Uri "http://localhost:5000/api/token-usage").total_usage.total_tokens
    $tokensUsed = $postTokens - $preTokens
    $responseTime = ($endTime - $startTime).TotalSeconds
    
    # Analyze response quality
    $qualityIndicators = @{
        MentionsLakehouse = $response.response -match "AzureMigrations|lakehouse" ? $true : $false
        MentionsWorkspace = $response.response -match "MSXI-Azure02-Dev" ? $true : $false
        IdentifiesTools = $response.response -match "mcp_" ? $true : $false
        ShowsSteps = $response.response -match "1\.|2\.|first|then|next" ? $true : $false
        HandlesComplexity = $response.response -match "analyze|calculate|identify|compare" ? $true : $false
    }
    
    Write-Host "`n=== Question $QuestionNumber Results ==="
    Write-Host "Tokens Used: $tokensUsed"
    Write-Host "Response Time: $responseTime seconds"
    Write-Host "Quality Indicators:"
    $qualityIndicators.GetEnumerator() | ForEach-Object { Write-Host "  $($_.Key): $($_.Value)" }
    
    return @{
        question_number = $QuestionNumber
        tokens_used = $tokensUsed
        response_time = $responseTime
        quality_score = ($qualityIndicators.Values | Where-Object { $_ -eq $true }).Count
        response_text = $response.response
        expected_behavior = $ExpectedBehavior
    }
}
```

### **Expected Performance Benchmarks**

For AzureMigrations-specific queries:

```json
{
  "complexity_expectations": {
    "migration_assessment_queries": {
      "token_range": "2500-4000",
      "response_time": "< 5 seconds",
      "quality_score": "> 4/5 indicators"
    },
    "cost_analysis_queries": {
      "token_range": "3000-4500", 
      "response_time": "< 6 seconds",
      "quality_score": "> 4/5 indicators"
    },
    "technical_planning_queries": {
      "token_range": "3500-5000",
      "response_time": "< 7 seconds", 
      "quality_score": "> 3/5 indicators"
    },
    "context_retention_queries": {
      "token_range": "1500-3000",
      "response_time": "< 4 seconds",
      "quality_score": "> 4/5 indicators"
    }
  }
}
```

## 🎯 Success Criteria for AzureMigrations Testing

### **Domain-Specific Expectations**:

1. **Workspace & Lakehouse Recognition**: 100% accuracy in connecting to MSXI-Azure02-Dev and AzureMigrations
2. **Migration Context Understanding**: Demonstrates understanding of Azure migration concepts and terminology
3. **Data Relationship Mapping**: Correctly identifies relationships between servers, applications, and dependencies  
4. **Business Logic Application**: Applies migration planning logic (waves, priorities, constraints)
5. **Cost Calculation Accuracy**: Handles financial analysis and TCO calculations appropriately

### **Technical Performance Targets**:

- **Average Tokens per Query**: < 3500 for migration assessment questions
- **Complex Query Handling**: < 5000 tokens for multi-table cost analysis
- **Context Retention**: Successfully references previous analysis in follow-up questions
- **Tool Selection**: 95%+ accuracy in selecting appropriate MCP tools for lakehouse operations

## 📋 Results Documentation Template

```markdown
## AzureMigrations Lakehouse Testing Results

### Environment Details:
- **Workspace**: MSXI-Azure02-Dev
- **Lakehouse**: AzureMigrations  
- **Test Session**: [Session ID]
- **Test Date**: [Date and Time]

### Question Performance Summary:
| Category | Avg Tokens | Avg Response Time | Quality Score | Notes |
|----------|------------|------------------|---------------|-------|
| Migration Assessment (Q1-3) | [X] | [X]s | [X]/5 | [Comments] |
| Cost Analysis (Q4-6) | [X] | [X]s | [X]/5 | [Comments] |
| Technical Planning (Q7-9) | [X] | [X]s | [X]/5 | [Comments] |
| Performance Analysis (Q10-12) | [X] | [X]s | [X]/5 | [Comments] |
| Migration Execution (Q13-15) | [X] | [X]s | [X]/5 | [Comments] |
| Context Retention (Q16-18) | [X] | [X]s | [X]/5 | [Comments] |

### Key Observations:
- **Strengths**: [Domain understanding, tool selection, etc.]
- **Improvement Areas**: [Token efficiency, response accuracy, etc.]
- **Unexpected Behaviors**: [Any surprises or issues]

### Recommendations:
- [Specific improvements for AzureMigrations scenarios]
- [Optimization opportunities for migration-specific queries]
- [Additional testing scenarios to consider]
```

This guide provides realistic, context-specific questions that will thoroughly test your Power BI MCP Application's ability to handle actual Azure migration scenarios using the real AzureMigrations lakehouse data in your MSXI-Azure02-Dev workspace.
