# SQL Endpoint Testing Guide: Complex Query Generation & Performance Evaluation

## Overview
This guide provides structured instructions for generating complex user questions to test the Power BI MCP Application's SQL endpoint capabilities. The goal is to evaluate the chatbot's performance in translating natural language queries into effective SQL operations without directly providing SQL solutions.

## 🎯 Testing Objectives

### Primary Goals:
1. **Query Complexity Handling**: Test the chatbot's ability to understand and translate complex analytical requirements
2. **Tool Selection Accuracy**: Evaluate correct identification of MCP tools for different query types
3. **Parameter Extraction**: Assess accuracy in extracting connection parameters and query specifications
4. **Error Handling**: Test resilience when dealing with ambiguous or incomplete requests
5. **Performance Optimization**: Monitor token usage and response efficiency during complex interactions

## 📋 Pre-Testing Setup Instructions

### 1. Environment Preparation
```powershell
# Ensure application is running
cd "C:\Users\v-vare\Downloads\New folder\AI-Cert\MCP-Application"
python webapp/app.py

# Open separate PowerShell for testing
# Navigate to application directory for API calls
```

### 2. Baseline Metrics Collection
Before starting complex testing, collect baseline metrics:

```powershell
# Cache status
$cacheStatus = Invoke-RestMethod -Uri "http://localhost:5000/api/cache/status" -Method GET
Write-Host "Cache Hit Rate: $($cacheStatus.cache_benefits.cache_hit_rate)"

# Token usage baseline
$tokenUsage = Invoke-RestMethod -Uri "http://localhost:5000/api/token-usage" -Method GET
Write-Host "Current Token Usage: $($tokenUsage.total_usage.total_tokens)"
```

### 3. Test Session Initialization
```powershell
# Start fresh conversation for consistent testing
$sessionId = "sql-test-session-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Write-Host "Test Session ID: $sessionId"
```

## 🔍 Complex Query Categories for Testing

### Category 1: Multi-Table Analysis Questions
**Complexity Level**: High  
**Focus**: Join operations, relationship navigation, aggregation across tables

#### Sample Questions to Ask Your Chatbot:

1. **Cross-Departmental Analysis**:
   > "I need to analyze sales performance across different regions and compare it with marketing spend by department. Show me which regions have the highest ROI when considering both sales revenue and marketing investment."

2. **Temporal Trend Analysis**:
   > "Can you help me understand the quarterly growth patterns for our top 5 products over the last 2 years, and how they correlate with seasonal customer behavior trends?"

3. **Hierarchical Data Exploration**:
   > "I want to drill down from country-level sales data to city-level performance, but only for countries where sales exceeded the global average. Include population density as a factor."

4. **Complex Aggregation Scenarios**:
   > "Show me the average deal size for each sales representative, but group them by their manager and region. I also need to see how this compares to their quota achievement percentage."

#### Expected Evaluation Criteria:
- [ ] Correctly identifies need for multiple table joins
- [ ] Recognizes temporal analysis requirements
- [ ] Suggests appropriate MCP tools for lakehouse connection
- [ ] Handles hierarchical data concepts appropriately
- [ ] Maintains conversation context for follow-up questions

### Category 2: Analytical Function Challenges
**Complexity Level**: Very High  
**Focus**: Window functions, statistical analysis, advanced calculations

#### Sample Questions to Ask Your Chatbot:

5. **Running Calculations**:
   > "I need a running total of monthly revenue with a 3-month moving average, but reset the calculation for each product category. Also show the percentage change from the previous month."

6. **Ranking and Percentile Analysis**:
   > "Rank our customers by their lifetime value, but I want to see their percentile ranking within their industry segment. Include customers who made purchases in the last 6 months only."

7. **Statistical Correlation Queries**:
   > "Help me find correlations between customer satisfaction scores and various factors like response time, product type, and support ticket volume. I need statistical significance indicators."

8. **Cohort Analysis Requirements**:
   > "I want to perform a cohort analysis on customer retention rates. Group customers by their first purchase month and track their purchasing behavior over the next 12 months."

#### Expected Evaluation Criteria:
- [ ] Recognizes need for advanced analytical functions
- [ ] Understands statistical concepts in natural language
- [ ] Suggests appropriate data preparation steps
- [ ] Identifies potential data quality considerations
- [ ] Proposes suitable visualization approaches

### Category 3: Data Quality & Validation Queries
**Complexity Level**: Medium-High  
**Focus**: Data integrity checks, outlier detection, validation rules

#### Sample Questions to Ask Your Chatbot:

9. **Data Completeness Assessment**:
   > "I suspect we have data quality issues in our customer database. Can you help me identify records with missing email addresses, invalid phone numbers, or customers with duplicate entries based on similar names and addresses?"

10. **Outlier Detection**:
    > "Find unusual patterns in our sales data - transactions that are significantly higher or lower than typical for each sales representative, and flag any that might indicate data entry errors or exceptional deals."

11. **Cross-System Consistency Checks**:
    > "Compare customer information between our CRM system and billing system tables. Identify discrepancies in customer names, addresses, or contact information that need reconciliation."

12. **Business Rule Validation**:
    > "Verify that all orders follow our business rules: orders above $10,000 should have manager approval, international orders should have shipping addresses validated, and bulk discounts should only apply to eligible customer tiers."

#### Expected Evaluation Criteria:
- [ ] Understands data quality concepts
- [ ] Proposes multi-step validation approaches
- [ ] Suggests appropriate comparison methodologies
- [ ] Identifies potential business rule violations
- [ ] Recommends data cleansing strategies

### Category 4: Performance & Optimization Scenarios
**Complexity Level**: High  
**Focus**: Large dataset handling, query optimization awareness

#### Sample Questions to Ask Your Chatbot:

13. **Large Dataset Analysis**:
    > "We have 50 million transaction records. I need to analyze monthly spending patterns by customer segment, but the query needs to run efficiently. What's the best approach for this analysis?"

14. **Real-Time Dashboard Requirements**:
    > "I'm building a dashboard that needs to refresh every 15 minutes with current sales metrics, top-performing products, and alert conditions. How should I structure the data retrieval for optimal performance?"

15. **Historical Data Archiving Query**:
    > "Help me identify records older than 7 years that can be archived, but I need to ensure we retain records that are referenced by active customers or ongoing legal cases."

16. **Aggregation Optimization**:
    > "I need daily, weekly, and monthly sales summaries for the executive dashboard. The data should be pre-aggregated for fast loading, but I need to understand what tables and relationships are optimal."

#### Expected Evaluation Criteria:
- [ ] Recognizes performance considerations
- [ ] Suggests appropriate aggregation strategies
- [ ] Understands refresh and caching concepts
- [ ] Proposes efficient data access patterns
- [ ] Considers scalability implications

## 📊 Performance Evaluation Framework

### 1. Response Quality Assessment

#### Scoring Criteria (1-5 scale):

**Tool Selection Accuracy (Weight: 25%)**
- 5: Perfect tool identification with correct sequence
- 4: Correct primary tool with minor sequence issues
- 3: Mostly correct tools with some confusion
- 2: Partial correctness, missing key tools
- 1: Incorrect or no tool identification

**Parameter Extraction (Weight: 20%)**
- 5: All parameters correctly identified and formatted
- 4: Most parameters correct with minor formatting issues
- 3: Key parameters identified, some missing details
- 2: Partial parameter extraction, some errors
- 1: Poor or no parameter extraction

**Query Understanding (Weight: 25%)**
- 5: Complete comprehension of complex requirements
- 4: Good understanding with minor gaps
- 3: Adequate understanding of main concepts
- 2: Partial understanding, missing key aspects
- 1: Poor comprehension of requirements

**Solution Approach (Weight: 20%)**
- 5: Optimal approach with clear step-by-step plan
- 4: Good approach with minor inefficiencies
- 3: Reasonable approach, some improvements possible
- 2: Workable but suboptimal approach
- 1: Poor or unrealistic approach

**Error Handling (Weight: 10%)**
- 5: Proactive error identification and prevention
- 4: Good error anticipation with solutions
- 3: Basic error awareness
- 2: Limited error consideration
- 1: No error awareness

### 2. Token Efficiency Monitoring

```powershell
# Monitor token usage during complex query testing
function Test-ComplexQuery {
    param($Question, $TestNumber)
    
    # Get baseline tokens
    $preTokens = (Invoke-RestMethod -Uri "http://localhost:5000/api/token-usage").total_usage.total_tokens
    
    # Send complex question
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/chat-optimized" -Method POST -ContentType "application/json" -Body (@{
        message = $Question
        session_id = $sessionId
    } | ConvertTo-Json)
    
    # Get post-request tokens
    $postTokens = (Invoke-RestMethod -Uri "http://localhost:5000/api/token-usage").total_usage.total_tokens
    $tokensUsed = $postTokens - $preTokens
    
    Write-Host "Test $TestNumber - Tokens Used: $tokensUsed"
    Write-Host "Response Quality Indicators:"
    Write-Host "- Tools Mentioned: $($response.response -match 'mcp_' ? 'Yes' : 'No')"
    Write-Host "- Parameters Extracted: $($response.response -match 'workspace_identifier|lakehouse_identifier' ? 'Yes' : 'No')"
    Write-Host "- Step-by-step Approach: $($response.response -match '1\.|2\.|step|first|then' ? 'Yes' : 'No')"
    
    return @{
        tokens_used = $tokensUsed
        response = $response.response
        test_number = $TestNumber
    }
}
```

### 3. Conversation Context Evaluation

#### Context Retention Tests:
After asking 3-4 complex queries, test context retention:

**Follow-up Questions**:
17. "Based on the previous analysis we discussed, can you modify the approach to include quarterly comparisons?"
18. "Using the same dataset from our earlier conversation, now filter for only enterprise customers."
19. "Can you combine the techniques we used in the first two queries for a comprehensive report?"

#### Expected Behaviors:
- [ ] References previous query context appropriately
- [ ] Maintains lakehouse connection details
- [ ] Builds upon established analytical approaches
- [ ] Avoids repeating basic setup instructions

## 🧪 Systematic Testing Protocol

### Phase 1: Individual Query Testing (Questions 1-16)
```powershell
# Execute each question individually and collect metrics
$testResults = @()
for ($i = 1; $i -le 16; $i++) {
    Write-Host "`n=== Testing Question $i ==="
    $question = Read-Host "Enter Question $i"
    $result = Test-ComplexQuery -Question $question -TestNumber $i
    $testResults += $result
    
    # Brief pause between tests
    Start-Sleep -Seconds 5
}
```

### Phase 2: Context Retention Testing (Questions 17-19)
```powershell
# Test conversation memory and context building
Write-Host "`n=== Context Retention Testing ==="
foreach ($followUp in @("Question 17", "Question 18", "Question 19")) {
    $question = Read-Host "Enter $followUp"
    $result = Test-ComplexQuery -Question $question -TestNumber $followUp
    $testResults += $result
}
```

### Phase 3: Stress Testing
```powershell
# Rapid-fire questions to test optimization under load
Write-Host "`n=== Stress Testing ==="
$stressQuestions = @(
    "Quick analysis of top 10 customers by revenue",
    "Monthly sales trend for last year", 
    "Product performance comparison by region",
    "Customer churn rate analysis",
    "Inventory turnover metrics"
)

foreach ($sq in $stressQuestions) {
    $result = Test-ComplexQuery -Question $sq -TestNumber "Stress"
    $testResults += $result
    Start-Sleep -Seconds 2  # Minimal pause for stress testing
}
```

## 📈 Results Analysis & Reporting

### 1. Performance Summary Generation
```powershell
# Generate test summary
$totalTokens = ($testResults | Measure-Object -Property tokens_used -Sum).Sum
$avgTokensPerQuery = $totalTokens / $testResults.Count
$maxTokens = ($testResults | Measure-Object -Property tokens_used -Maximum).Maximum

Write-Host "`n=== TEST SUMMARY ==="
Write-Host "Total Queries Tested: $($testResults.Count)"
Write-Host "Total Tokens Consumed: $totalTokens"
Write-Host "Average Tokens per Query: $avgTokensPerQuery"
Write-Host "Maximum Tokens in Single Query: $maxTokens"

# Final cache performance
$finalCache = Invoke-RestMethod -Uri "http://localhost:5000/api/cache/status"
Write-Host "Final Cache Hit Rate: $($finalCache.cache_benefits.cache_hit_rate)"
```

### 2. Quality Assessment Template
```markdown
## SQL Endpoint Testing Results - [Date]

### Test Configuration:
- Session ID: [Generated Session ID]
- Test Duration: [Start Time] - [End Time]
- Queries Tested: [Total Number]
- Complexity Categories: Multi-table, Analytical, Data Quality, Performance

### Quantitative Results:
- **Token Efficiency**: 
  - Total Tokens: [X] 
  - Avg per Query: [X]
  - Optimization Savings: [X]%
- **Cache Performance**: Hit Rate [X]%
- **Response Time**: Avg [X] seconds

### Qualitative Assessment:
- **Tool Selection**: [Score 1-5] - [Comments]
- **Parameter Extraction**: [Score 1-5] - [Comments] 
- **Query Understanding**: [Score 1-5] - [Comments]
- **Solution Approach**: [Score 1-5] - [Comments]
- **Error Handling**: [Score 1-5] - [Comments]

### Observations:
- **Strengths**: [Key positive observations]
- **Improvement Areas**: [Areas needing enhancement]
- **Unexpected Behaviors**: [Any surprises or anomalies]

### Recommendations:
- **Immediate Actions**: [Priority improvements]
- **Optimization Opportunities**: [Performance enhancements]
- **Future Testing**: [Additional test scenarios to consider]
```

## 🎯 Success Criteria & Benchmarks

### Minimum Acceptable Performance:
- **Tool Selection Accuracy**: > 90%
- **Parameter Extraction**: > 85%
- **Query Understanding**: > 80%
- **Token Efficiency**: < 4000 tokens per complex query
- **Cache Hit Rate**: > 85%
- **Context Retention**: 3+ conversation turns

### Excellent Performance Targets:
- **Tool Selection Accuracy**: > 95%
- **Parameter Extraction**: > 95%
- **Query Understanding**: > 90%
- **Token Efficiency**: < 3000 tokens per complex query
- **Cache Hit Rate**: > 95%
- **Context Retention**: 5+ conversation turns with complex referencing

## 🔄 Iterative Testing & Improvement

### 1. Weekly Testing Cycles
- Run subset of complex queries (4-6 questions)
- Monitor performance trends
- Identify degradation or improvement patterns
- Update baseline expectations

### 2. Monthly Comprehensive Testing
- Full 19-question test suite
- Performance comparison with previous months
- Quality assessment evolution
- Optimization strategy adjustments

### 3. Continuous Monitoring Integration
```powershell
# Add to daily monitoring routine
function Daily-ComplexityCheck {
    $testQuestion = "Analyze sales trends for top customers in the Northeast region with year-over-year comparison"
    $result = Test-ComplexQuery -Question $testQuestion -TestNumber "Daily"
    
    if ($result.tokens_used -gt 4000) {
        Write-Warning "Daily complexity test exceeded token threshold: $($result.tokens_used)"
    }
    
    return $result
}
```

This comprehensive testing guide provides structured evaluation of your Power BI MCP Application's SQL endpoint capabilities while maintaining objectivity through third-person assessment methodology. Use it to systematically evaluate and improve the chatbot's performance with complex analytical requirements.
