# WSA Product Categories Relationship - 2015

This diagram shows how different product categories relate to each other in the WSA framework.

```mermaid
graph LR
    %% Define styles
    classDef level2 fill:#4682B4,stroke:#333,stroke-width:2px,color:#fff
    classDef products fill:#9370DB,stroke:#333,stroke-width:2px,color:#fff
    classDef semi fill:#5F9EA0,stroke:#333,stroke-width:2px,color:#fff
    classDef finished fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    classDef flat fill:#4ECDC4,stroke:#333,stroke-width:2px,color:#fff
    classDef long fill:#FFB84D,stroke:#333,stroke-width:2px,color:#fff
    classDef tubular fill:#FF6347,stroke:#333,stroke-width:2px,color:#fff
    classDef specific fill:#96CEB4,stroke:#333,stroke-width:2px,color:#fff
    
    %% WSA Product Categories Structure - CORRECTED HIERARCHY
    A[Crude Steel Production<br/>4925 kt]
    
    B[Semi-finished Products<br/>Intermediate Forms<br/>4925 kt]
    C[Steel Products<br/>Final Output<br/>4462 kt]
    
    D[Ingots and Semis<br/>Cast Forms<br/>25 kt]
    E[Continuously-cast Steel<br/>Modern Casting<br/>4900 kt]
    
    F[Flat Products<br/>2628 kt]
    G[Long Products<br/>1679 kt]
    H[Tubular Products<br/>155 kt]
    
    I[Sheets/Strips/Coils<br/>2415 kt]
    J[Coated Products<br/>2023 kt]
    
    K[Wire Rod<br/>720 kt]
    L[Railway Track Material<br/>89 kt]
    M[Other Long Products<br/>870 kt]
    
    %% CORRECTED Product Category Relationships - Sequential Flow
    A --> B
    B --> C
    
    B --> D
    B --> E
    
    C --> F
    C --> G
    C --> H
    
    F --> I
    F --> J
    
    G --> K
    G --> L
    G --> M
    
    %% Apply Styles
    class A level2
    class B,D,E semi
    class C products
    class F,I,J flat
    class G,K,L,M long
    class H tubular
```

## Product Categories Summary - 2015 (CORRECTED HIERARCHY)

| Production Stage | Category | Total Volume (kt) | Track A Coverage |
|-----------------|----------|-------------------|------------------|
| **Crude Steel** | Total Production | 4925 | ✅ Full |
| **Semi-finished** | Ingots + Continuously-cast | 4925 | ✅ Full |
| **Final Products** | Flat + Long + Tubular | 4462 | ✅ Full |

### Product Categories Breakdown:
| Category | Sub-categories | Total Volume (kt) | Track A Coverage |
|----------|----------------|-------------------|------------------|
| **Flat Products** | Sheets, Strips, Coils, Coated | 2628 | ✅ Full |
| **Long Products** | Wire Rod, Rails, Other Sections | 1679 | ✅ Full |
| **Tubular Products** | Pipes, Tubes | 155 | ✅ Full |

### Production Flow Logic:
- **Crude Steel** (4925 kt) → **Semi-finished** (4925 kt) → **Final Products** (4462 kt)
- **Yield Loss**: 9.4% processing loss from semi-finished to final products

### Product Specialization:
- **Hot Rolled Coil, Sheet, Strip (<3mm)**: 2415 kt
- **Non-metallic Coated Products**: 656 kt  
- **Metal Coated Products**: 1367 kt

*Based on official WSA Product Categories Relationship diagram*

