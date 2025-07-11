# WSA Production Flow Hierarchy - 2025

This diagram replicates the official WSA Production Flow Hierarchy with 2025 forecast volumes.

```mermaid
graph TD
    %% Define styles
    classDef level0 fill:#8B4513,stroke:#333,stroke-width:2px,color:#fff
    classDef level1 fill:#CD853F,stroke:#333,stroke-width:2px,color:#fff
    classDef level2 fill:#4682B4,stroke:#333,stroke-width:2px,color:#fff
    classDef level3 fill:#5F9EA0,stroke:#333,stroke-width:2px,color:#fff
    classDef level4 fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    classDef level5 fill:#228B22,stroke:#333,stroke-width:2px,color:#fff
    classDef tubular fill:#FF6347,stroke:#333,stroke-width:2px,color:#fff
    
    %% WSA Production Flow Structure (Levels 0-5 + Tubular)
    A[Production of Iron Ore<br/>Level 0<br/>939497 kt]
    B[Production of Pig Iron<br/>Level 1<br/>3501 kt]
    C[Total Production of Crude Steel<br/>Level 2<br/>5516 kt]
    
    D[Production of Ingots<br/>Level 3<br/>27 kt]
    E[Production of Continuously-cast Steel<br/>Level 3<br/>5506 kt]
    
    F[Production of Hot Rolled Products<br/>Level 4<br/>5004 kt total]
    
    G[Production of Hot Rolled Flat Products<br/>Level 4<br/>3232 kt]
    H[Production of Hot Rolled Long Products<br/>Level 4<br/>1772 kt]
    
    I[Production of Hot Rolled Coil, Sheet, and Strip (&lt;3mm)<br/>Level 5<br/>1646 kt]
    J[Production of Non-metallic Coated Sheet and Strip<br/>Level 5<br/>535 kt]
    K[Production of Other Metal Coated Sheet and Strip<br/>Level 5<br/>1051 kt]
    
    L[Production of Wire Rod<br/>Level 5<br/>1557 kt]
    M[Production of Railway Track Material<br/>Level 5<br/>215 kt]
    
    N[Total Production of Tubular Products<br/>Parallel Branch<br/>156 kt]
    
    %% Main Production Flow (exact WSA structure)
    A --> B
    B --> C
    C --> D
    C --> E
    D --> F
    E --> F
    F --> G
    F --> H
    G --> I
    G --> J
    G --> K
    H --> L
    H --> M
    
    %% Tubular Branch (parallel from crude steel)
    C --> N
    
    %% Apply Styles
    class A level0
    class B level1
    class C level2
    class D level3
    class E level3
    class F level4
    class G level4
    class H level4
    class I level5
    class J level5
    class K level5
    class L level5
    class M level5
    class N tubular
```

## Production Flow Summary - 2025

| Category | Volume (kt) | Share (%) |
|----------|-------------|-----------|\n| Production of Iron Ore | 939497 | 95.1% |\n| True Steel Use (finished steel equivalent) | 10533 | 1.1% |\n| Apparent Steel Use (crude steel equivalent) | 6969 | 0.7% |\n| Apparent Steel Use (finished steel products) | 6290 | 0.6% |\n| Total Production of Crude Steel | 5516 | 0.6% |\n| Production of Continuously-cast Steel | 5506 | 0.6% |\n| Production of Pig Iron | 3501 | 0.4% |\n| Production of Hot Rolled Flat Products | 3232 | 0.3% |\n| Production of Hot Rolled Long Products | 1772 | 0.2% |\n| Production of Hot Rolled Coil, Sheet, and Strip (<3mm) | 1646 | 0.2% |\n| Production of Wire Rod | 1557 | 0.2% |\n| Production of Other Metal Coated Sheet and Strip | 1051 | 0.1% |\n| Production of Non-metallic Coated Sheet and Strip | 535 | 0.1% |\n| Production of Railway Track Material | 215 | 0.0% |\n| Total Production of Tubular Products | 156 | 0.0% |\n| Production of Ingots | 27 | 0.0% |\n| **Total** | **988002** | **100.0%** |

*Based on official WSA Production Flow Hierarchy diagram*
*Volumes represent Track A forecasts mapped to WSA categories*

