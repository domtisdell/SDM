# WSA Production Flow Hierarchy - 2015

This diagram replicates the official WSA Production Flow Hierarchy with 2015 forecast volumes.

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
    A[Production of Iron Ore<br/>Level 0<br/>809900 kt]
    B[Production of Pig Iron<br/>Level 1<br/>3594 kt]
    C[Total Production of Crude Steel<br/>Level 2<br/>4925 kt]
    
    D[Production of Ingots<br/>Level 3<br/>25 kt]
    E[Production of Continuously-cast Steel<br/>Level 3<br/>4900 kt]
    
    F[Production of Hot Rolled Products<br/>Level 4<br/>4307 kt total]
    
    G[Production of Hot Rolled Flat Products<br/>Level 4<br/>2628 kt]
    H[Production of Hot Rolled Long Products<br/>Level 4<br/>1679 kt]
    
    I[Production of Hot Rolled Coil, Sheet, and Strip - less than 3mm<br/>Level 5<br/>2415 kt]
    J[Production of Non-metallic Coated Sheet and Strip<br/>Level 5<br/>656 kt]
    K[Production of Other Metal Coated Sheet and Strip<br/>Level 5<br/>1367 kt]
    
    L[Production of Wire Rod<br/>Level 5<br/>720 kt]
    M[Production of Railway Track Material<br/>Level 5<br/>89 kt]
    
    N[Total Production of Tubular Products<br/>Parallel Branch<br/>155 kt]
    
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

## Production Flow Summary - 2015

| Category | Volume (kt) | Share (%) |
|----------|-------------|-----------|\n| Production of Iron Ore | 809900 | 94.4% |\n| True Steel Use (finished steel equivalent) | 11228 | 1.3% |\n| Apparent Steel Use (crude steel equivalent) | 6967 | 0.8% |\n| Apparent Steel Use (finished steel products) | 6291 | 0.7% |\n| Total Production of Crude Steel | 4925 | 0.6% |\n| Production of Continuously-cast Steel | 4900 | 0.6% |\n| Production of Pig Iron | 3594 | 0.4% |\n| Production of Hot Rolled Flat Products | 2628 | 0.3% |\n| Production of Hot Rolled Coil, Sheet, and Strip (<3mm) | 2415 | 0.3% |\n| Production of Hot Rolled Long Products | 1679 | 0.2% |\n| Production of Other Metal Coated Sheet and Strip | 1367 | 0.2% |\n| Production of Wire Rod | 720 | 0.1% |\n| Production of Non-metallic Coated Sheet and Strip | 656 | 0.1% |\n| Total Production of Tubular Products | 155 | 0.0% |\n| Production of Railway Track Material | 89 | 0.0% |\n| Production of Ingots | 25 | 0.0% |\n| **Total** | **857539** | **100.0%** |

*Based on official WSA Production Flow Hierarchy diagram*
*Volumes represent Track A forecasts mapped to WSA categories*

