# WSA Crude Steel Production Methods - 2050

This diagram shows the different production pathways for crude steel based on WSA methodology.

```mermaid
graph TD
    %% Define styles
    classDef crude fill:#4682B4,stroke:#333,stroke-width:2px,color:#fff
    classDef eaf fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
    classDef bof fill:#4ECDC4,stroke:#333,stroke-width:2px,color:#fff
    classDef combined fill:#96CEB4,stroke:#333,stroke-width:2px,color:#fff
    
    %% WSA Crude Steel Production Methods Structure
    A[Total Production of Crude Steel<br/>5505 kt]
    
    B[Production of Crude Steel in Electric Furnaces<br/>EAF Route<br/>Estimated: 1652 kt]
    C[Production of Crude Steel in Oxygen-blown Converters<br/>BOF Route<br/>Estimated: 3854 kt]
    
    D[Electric Arc Furnace Route<br/>Scrap-based Production]
    E[Basic Oxygen Furnace Route<br/>Pig Iron-based Production]
    
    F[Production Methods Combined<br/>Total: 5505 kt]
    
    %% Production Method Flow
    A --> B
    A --> C
    B --> D
    C --> E
    D --> F
    E --> F
    
    %% Apply Styles
    class A crude
    class B eaf
    class C bof
    class D eaf
    class E bof
    class F combined
```

## Crude Steel Production Methods - 2050

| Production Method | Estimated Volume (kt) | Share (%) |
|-------------------|----------------------|-----------|
| Electric Arc Furnace (EAF) | 1652 | 30% |
| Basic Oxygen Furnace (BOF) | 3854 | 70% |
| **Total Crude Steel** | **5505** | **100%** |

*Note: EAF/BOF split estimated using typical Australian steel industry proportions*
*Based on official WSA Crude Steel Production Methods diagram*

