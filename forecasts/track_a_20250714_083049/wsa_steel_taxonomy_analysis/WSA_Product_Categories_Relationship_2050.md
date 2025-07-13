# WSA Product Categories Relationship - 2050

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
    A[Crude Steel Production<br/>nan kt]
    
    B[Semi-finished Products<br/>Intermediate Forms<br/>nan kt]
    C[Steel Products<br/>Final Output<br/>nan kt]
    
    D[Ingots and Semis<br/>Cast Forms<br/>nan kt]
    E[Continuously-cast Steel<br/>Modern Casting<br/>nan kt]
    
    F[Flat Products<br/>nan kt]
    G[Long Products<br/>nan kt]
    H[Tubular Products<br/>nan kt]
    
    I[Sheets/Strips/Coils<br/>nan kt]
    J[Coated Products<br/>nan kt]
    
    K[Wire Rod<br/>nan kt]
    L[Railway Track Material<br/>nan kt]
    M[Other Long Products<br/>0 kt]
    
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

## Product Categories Summary - 2050 (CORRECTED HIERARCHY)

| Production Stage | Category | Total Volume (kt) | Track A Coverage |
|-----------------|----------|-------------------|------------------|
| **Crude Steel** | Total Production | nan | ✅ Full |
| **Semi-finished** | Ingots + Continuously-cast | nan | ✅ Full |
| **Final Products** | Flat + Long + Tubular | nan | ✅ Full |

### Product Categories Breakdown:
| Category | Sub-categories | Total Volume (kt) | Track A Coverage |
|----------|----------------|-------------------|------------------|
| **Flat Products** | Sheets, Strips, Coils, Coated | nan | ✅ Full |
| **Long Products** | Wire Rod, Rails, Other Sections | nan | ✅ Full |
| **Tubular Products** | Pipes, Tubes | nan | ✅ Full |

### Production Flow Logic:
- **Crude Steel** (nan kt) → **Semi-finished** (nan kt) → **Final Products** (nan kt)
- **Yield Loss**: 0.0% processing loss from semi-finished to final products

### Product Specialization:
- **Hot Rolled Coil, Sheet, Strip (<3mm)**: nan kt
- **Non-metallic Coated Products**: nan kt  
- **Metal Coated Products**: nan kt

*Based on official WSA Product Categories Relationship diagram*

