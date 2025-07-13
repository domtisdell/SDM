# WSA Production Flow Hierarchy - 2020

This diagram replicates the official WSA Production Flow Hierarchy with 2020 forecast volumes.

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
    A[Production of Iron Ore<br/>Level 0<br/>918063 kt]
    B[Production of Pig Iron<br/>Level 1<br/>3723 kt]
    C[Total Production of Crude Steel<br/>Level 2<br/>5490 kt]
    
    D[Production of Ingots<br/>Level 3<br/>nan kt]
    E[Production of Continuously-cast Steel<br/>Level 3<br/>5490 kt]
    
    F[Production of Hot Rolled Products<br/>Level 4<br/>4915 kt total]
    
    G[Production of Hot Rolled Flat Products<br/>Level 4<br/>2951 kt]
    H[Production of Hot Rolled Long Products<br/>Level 4<br/>1964 kt]
    
    I[Production of Hot Rolled Coil, Sheet, and Strip - less than 3mm<br/>Level 5<br/>2612 kt]
    J[Production of Non-metallic Coated Sheet and Strip<br/>Level 5<br/>797 kt]
    K[Production of Other Metal Coated Sheet and Strip<br/>Level 5<br/>1589 kt]
    
    L[Production of Wire Rod<br/>Level 5<br/>578 kt]
    M[Production of Railway Track Material<br/>Level 5<br/>90 kt]
    
    N[Total Production of Tubular Products<br/>Parallel Branch<br/>nan kt]
    
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

## Production Flow Summary - 2020

| Category | Volume (kt) | Share (%) |
|----------|-------------|-----------|\n| Apparent Steel Use (crude steel equivalent) | 5648 | 0.0% |\n| Production of Continuously-cast Steel | 5490 | 0.0% |\n| Production of Ingots | nan | 0.0% |\n| Production of Iron Ore | 918063 | 0.0% |\n| Total Production of Crude Steel | 5490 | 0.0% |\n| Apparent Steel Use (finished steel products) | 5100 | 0.0% |\n| Production of Pig Iron | 3723 | 0.0% |\n| Production of Hot Rolled Flat Products | 2951 | 0.0% |\n| Production of Hot Rolled Coil, Sheet, and Strip (<3mm) | 2612 | 0.0% |\n| Production of Hot Rolled Long Products | 1964 | 0.0% |\n| Production of Other Metal Coated Sheet and Strip | 1589 | 0.0% |\n| Production of Non-metallic Coated Sheet and Strip | 797 | 0.0% |\n| Production of Wire Rod | 578 | 0.0% |\n| Production of Railway Track Material | 90 | 0.0% |\n| Total Production of Tubular Products | nan | 0.0% |\n| True Steel Use (finished steel equivalent) | nan | 0.0% |\n| **Total** | **nan** | **100.0%** |

*Based on official WSA Production Flow Hierarchy diagram*
*Volumes represent Track A forecasts mapped to WSA categories*

