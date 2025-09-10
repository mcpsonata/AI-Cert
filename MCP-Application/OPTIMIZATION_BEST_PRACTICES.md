# Best Practices Guide: Power BI MCP Application Optimization

## Overview
This document provides actionable best practices derived from comprehensive testing of the Power BI MCP Application's prompt optimization system and lakehouse integration capabilities.

## 🎯 Core Optimization Principles

### 1. Token Efficiency Management
```python
# Recommended Token Limits
CONSERVATIVE_LIMIT = 2000        # For production stability
AGGRESSIVE_LIMIT = 1500          # For cost optimization  
EMERGENCY_LIMIT = 3000           # Absolute maximum before forced optimization

# Optimization Triggers
PROACTIVE_THRESHOLD = 0.7        # Trigger optimization at 70% of limit
EMERGENCY_THRESHOLD = 0.9        # Force immediate optimization at 90%
```

### 2. Conversation Context Strategy
```python
# Sliding Window Configuration
RECENT_MESSAGE_RATIO = 1/3       # Keep only 1/3 of tokens for recent messages
SUMMARY_RATIO = 1/4              # Use 1/4 for conversation summaries
BUFFER_RATIO = 5/12              # Reserve remaining for new content

# Auto-Optimization Rules
AUTO_OPTIMIZE_EVERY = 5          # Messages before automatic optimization
MAX_CONVERSATION_AGE = 50        # Messages before forcing new session
```

### 3. Cache Management Strategy
```python
# Cache Optimization
CACHE_EVERYTHING = True          # Cache all prompts aggressively
INVALIDATE_ON_TOOL_CHANGE = True # Regenerate when tools change
PREEMPTIVE_WARMING = True        # Warm cache on application start
CACHE_MONITORING = True          # Monitor hit rates continuously
```

## 📊 Monitoring and Alerting Framework

### Essential Metrics to Track
```json
{
  "token_metrics": {
    "average_per_request": "< 3000 tokens",
    "growth_rate_per_message": "< 30%",
    "optimization_trigger_frequency": "> 80% proactive"
  },
  "performance_metrics": {
    "cache_hit_rate": "> 90%",
    "tool_detection_accuracy": "> 95%", 
    "api_call_reduction": "> 45%"
  },
  "quality_metrics": {
    "parameter_extraction_success": "> 98%",
    "conversation_coherence": "High",
    "error_recovery_rate": "> 85%"
  }
}
```

### Automated Health Checks
```bash
# Daily Health Check Script
curl -s http://localhost:5000/api/cache/status | jq '.cache_benefits.cache_hit_rate'
curl -s http://localhost:5000/api/token-usage | jq '.total_usage.total_tokens'
curl -s http://localhost:5000/api/optimization/comparison | jq '.comparison.savings.percentage_reduction'
```

### Alert Thresholds
```yaml
alerts:
  token_growth:
    warning: "> 50% increase per message"
    critical: "> 100% increase per message"
  cache_performance:
    warning: "< 85% hit rate"
    critical: "< 70% hit rate"
  response_quality:
    warning: "< 90% tool detection accuracy"
    critical: "< 80% tool detection accuracy"
```

## 🔧 Implementation Best Practices

### 1. Endpoint Usage Strategy
```python
# Production Deployment Recommendations
PRIMARY_ENDPOINT = "/api/chat-optimized"     # Use for all new implementations
FALLBACK_ENDPOINT = "/api/chat"              # Keep for legacy compatibility
MONITORING_ENDPOINTS = [
    "/api/cache/status",                     # Cache performance monitoring
    "/api/optimization/comparison",          # Efficiency comparison
    "/api/token-usage"                       # Cost tracking
]
```

### 2. Session Management
```python
# Session Lifecycle Management
def manage_session_lifecycle(session_id, message_count):
    if message_count % 5 == 0:
        trigger_optimization(session_id)
    if message_count > 50:
        suggest_new_session(session_id)
    if token_usage > EMERGENCY_LIMIT:
        force_optimization(session_id)
```

### 3. Error Handling Protocol
```python
# Robust Error Handling
class OptimizedErrorHandler:
    def handle_tool_failure(self, error, context):
        # 1. Log error with full context
        logger.error(f"Tool failure: {error}", extra=context)
        
        # 2. Trigger autonomous debugging
        debug_result = self.autonomous_debug(error, context)
        
        # 3. Attempt recovery with alternative tools
        if not debug_result.success:
            return self.try_alternative_tools(context)
        
        # 4. Escalate to user with helpful suggestions
        return self.escalate_with_suggestions(error, context)
```

## 🚀 Performance Optimization Strategies

### 1. Proactive Cache Warming
```python
# Application Startup Cache Warming
def warm_caches_on_startup():
    tools_description = get_tools_description()
    
    # Pre-generate and cache main prompts
    PromptManager.get_cached_main_assistant_prompt(tools_description)
    
    # Pre-generate common tool detection scenarios
    common_scenarios = [
        ("lakehouse connection", "workspace info"),
        ("table listing", "sql tables"),
        ("query execution", "dax queries")
    ]
    
    for scenario in common_scenarios:
        PromptManager.get_cached_tool_detection_prompt(*scenario)
```

### 2. Intelligent Context Prioritization
```python
# Context Importance Scoring
def score_message_importance(message):
    importance_indicators = {
        "tool_execution": 0.9,
        "error_message": 0.8,
        "connection_info": 0.7,
        "query_results": 0.6,
        "general_chat": 0.3
    }
    
    # Prioritize keeping high-importance messages
    return max([score for keyword, score in importance_indicators.items() 
               if keyword in message.lower()])
```

### 3. Adaptive Optimization
```python
# Dynamic Optimization Based on Usage Patterns
class AdaptiveOptimizer:
    def optimize_based_on_pattern(self, session_history):
        if self.is_analytical_session(session_history):
            # Keep more context for analytical work
            return self.optimize_for_analysis(max_tokens=3000)
        elif self.is_exploratory_session(session_history):
            # More aggressive pruning for exploration
            return self.optimize_for_exploration(max_tokens=1500)
        else:
            # Standard optimization for mixed usage
            return self.standard_optimization(max_tokens=2000)
```

## 📋 Quality Assurance Guidelines

### 1. Pre-Deployment Testing Checklist
```markdown
- [ ] All optimization endpoints respond correctly
- [ ] Cache system populates and retrieves properly
- [ ] Token usage stays within acceptable ranges
- [ ] Tool detection accuracy > 95%
- [ ] Parameter extraction success > 98%
- [ ] Conversation optimization reduces token growth
- [ ] Error scenarios handled gracefully
- [ ] Performance targets met under load
```

### 2. Regression Testing Suite
```python
# Automated Regression Tests
class OptimizationRegressionTests:
    def test_token_efficiency(self):
        # Ensure token usage doesn't regress
        assert conversation_token_growth() < 0.3
        
    def test_cache_performance(self):
        # Verify cache hit rates maintained
        assert cache_hit_rate() > 0.90
        
    def test_tool_accuracy(self):
        # Confirm tool selection accuracy
        assert tool_detection_accuracy() > 0.95
        
    def test_optimization_effectiveness(self):
        # Validate optimization reduces consumption
        assert optimization_savings() > 0.6
```

### 3. Load Testing Parameters
```yaml
load_testing:
  concurrent_users: 50
  messages_per_user: 20
  test_duration: "30 minutes"
  acceptable_response_time: "< 3 seconds"
  memory_usage_limit: "< 2GB"
  token_budget_per_test: 500000
```

## 🎯 Production Deployment Recommendations

### 1. Infrastructure Setup
```yaml
# Production Infrastructure
resources:
  cpu: "4 cores minimum"
  memory: "8GB minimum" 
  storage: "50GB for cache and logs"
  
monitoring:
  metrics_collection: "prometheus"
  log_aggregation: "elasticsearch"
  alerting: "grafana + pagerduty"
  
scaling:
  horizontal_scaling: "kubernetes"
  load_balancer: "nginx"
  cache_layer: "redis"
```

### 2. Configuration Management
```python
# Production Configuration
PRODUCTION_CONFIG = {
    "optimization": {
        "max_history_tokens": 2000,
        "optimization_trigger": 0.7,
        "auto_optimize_interval": 5,
        "cache_strategy": "aggressive"
    },
    "monitoring": {
        "token_usage_alerts": True,
        "cache_performance_tracking": True,
        "quality_metrics_collection": True
    },
    "security": {
        "rate_limiting": True,
        "input_validation": True,
        "output_sanitization": True
    }
}
```

### 3. Maintenance Procedures
```bash
# Daily Maintenance Tasks
0 2 * * * /scripts/clear_old_sessions.sh      # Clean up old sessions
0 3 * * * /scripts/optimize_cache.sh          # Optimize cache storage
0 4 * * * /scripts/generate_metrics.sh        # Generate daily metrics report

# Weekly Maintenance Tasks  
0 1 * * 0 /scripts/cache_performance_report.sh # Weekly cache analysis
0 2 * * 0 /scripts/token_usage_analysis.sh     # Weekly cost analysis
```

## 🔍 Troubleshooting Guide

### Common Issues and Solutions

#### High Token Usage
```python
# Diagnostic Steps
1. Check conversation length: GET /api/sessions/{id}/history
2. Verify optimization settings: GET /api/cache/status
3. Trigger manual optimization: POST /api/conversations/{id}/optimize
4. Monitor token growth pattern: Multiple test requests
5. Adjust optimization thresholds if needed
```

#### Poor Cache Performance  
```python
# Diagnostic Steps
1. Verify cache population: GET /api/cache/status
2. Check tool change frequency: Monitor cache invalidations
3. Clear and regenerate cache: POST /api/cache/clear
4. Validate cache hit patterns: Review logs
5. Consider cache warming strategies
```

#### Tool Selection Issues
```python
# Diagnostic Steps
1. Review tool detection confidence scores
2. Validate parameter extraction accuracy  
3. Check for tool definition changes
4. Test with known-good scenarios
5. Review prompt engineering for tool selection
```

## 📈 Continuous Improvement Framework

### 1. Performance Baseline Updates
```python
# Monthly Baseline Review
def update_performance_baselines():
    current_metrics = collect_monthly_metrics()
    
    # Update baselines if consistent improvement observed
    if current_metrics.token_efficiency > baseline.token_efficiency * 1.1:
        baseline.update_token_efficiency(current_metrics.token_efficiency)
    
    # Flag regressions for investigation
    if current_metrics.cache_hit_rate < baseline.cache_hit_rate * 0.95:
        alert.regression_detected("cache_performance")
```

### 2. Optimization Strategy Evolution
```python
# Quarterly Strategy Review
class StrategyEvolution:
    def review_and_adapt(self):
        usage_patterns = analyze_quarterly_usage()
        
        if usage_patterns.show_more_analytical_work():
            self.adapt_for_analytical_workloads()
        elif usage_patterns.show_more_exploratory_queries():
            self.adapt_for_exploratory_workloads()
            
        return self.generate_optimization_recommendations()
```

### 3. User Feedback Integration
```python
# User Experience Optimization
def integrate_user_feedback():
    feedback = collect_user_satisfaction_metrics()
    
    # Adjust optimization aggressiveness based on user experience
    if feedback.response_quality_satisfaction < 0.85:
        reduce_optimization_aggressiveness()
    elif feedback.cost_satisfaction > 0.9:
        increase_optimization_aggressiveness()
```

## 🏆 Success Criteria and KPIs

### Production Success Metrics
```json
{
  "efficiency_kpis": {
    "token_reduction": "> 60% vs baseline",
    "api_call_reduction": "> 50% vs baseline", 
    "cache_hit_rate": "> 90%",
    "conversation_optimization": "> 80% proactive"
  },
  "quality_kpis": {
    "tool_detection_accuracy": "> 95%",
    "parameter_extraction_success": "> 98%",
    "user_satisfaction": "> 4.5/5",
    "error_rate": "< 2%"
  },
  "business_kpis": {
    "cost_reduction": "> 65% vs unoptimized",
    "response_time_improvement": "> 40%",
    "system_reliability": "> 99.5% uptime",
    "maintenance_overhead": "< 5% of operational time"
  }
}
```

This comprehensive best practices guide provides the foundation for maintaining and improving the optimized Power BI MCP Application, ensuring continued efficiency gains and high-quality user experience.
