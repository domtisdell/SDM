# WSA Steel Use Metrics Hierarchy - 2015

This diagram shows the relationship between different steel consumption measures as defined by WSA.

```mermaid
graph TD
    %% Define styles
    classDef metrics fill:#9370DB,stroke:#333,stroke-width:2px,color:#fff
    classDef apparent_crude fill:#4682B4,stroke:#333,stroke-width:2px,color:#fff
    classDef apparent_finished fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    classDef true_steel fill:#FF6347,stroke:#333,stroke-width:2px,color:#fff
    classDef calculation fill:#FFB84D,stroke:#333,stroke-width:2px,color:#fff
    
    %% WSA Steel Use Metrics Structure
    A[Steel Consumption Metrics<br/>WSA Standard Measures]
    
    B[Apparent Steel Use<br/>crude steel equivalent<br/>6967 kt]
    C[Apparent Steel Use<br/>finished steel products<br/>6291 kt]
    D[True Steel Use<br/>finished steel equivalent<br/>11228 kt]
    
    E[Calculated from<br/>Production + Imports - Exports<br/>Standard Trade Balance]
    F[Finished Products Basis<br/>Direct Product Consumption]
    G[Adjusted for Indirect Trade<br/>Embedded Steel in Goods]
    
    %% Steel Use Metrics Flow
    A --> B
    A --> C
    A --> D
    
    B --> E
    C --> F
    D --> G
    
    %% Apply Styles
    class A metrics
    class B apparent_crude
    class C apparent_finished
    class D true_steel
    class E,F,G calculation
```

## Steel Use Metrics Comparison - 2015

| WSA Consumption Metric | Volume (kt) | Calculation Method | Purpose |
|------------------------|-------------|-------------------|---------|
| **Apparent Steel Use (crude steel equivalent)** | 6967 | Production + Imports - Exports | Standard trade balance |
| **Apparent Steel Use (finished steel products)** | 6291 | Finished products basis | Direct consumption |
| **True Steel Use (finished steel equivalent)** | 11228 | Adjusted for indirect trade | Comprehensive consumption |

### Key Differences:
- **Crude Steel Equivalent**: Raw steel production accounting
- **Finished Steel Products**: End-product consumption focus  
- **Finished Steel Equivalent**: Most comprehensive measure including indirect trade

*Based on official WSA Steel Use Metrics Hierarchy diagram*

