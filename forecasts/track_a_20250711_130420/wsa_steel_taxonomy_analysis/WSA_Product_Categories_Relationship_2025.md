# WSA Product Categories Relationship - 2025

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
    A[Crude Steel Production<br/>5516 kt]
    
    B[Semi-finished Products<br/>Intermediate Forms<br/>5533 kt]
    C[Steel Products<br/>Final Output<br/>5160 kt]
    
    D[Ingots and Semis<br/>Cast Forms<br/>27 kt]
    E[Continuously-cast Steel<br/>Modern Casting<br/>5506 kt]
    
    F[Flat Products<br/>3232 kt]
    G[Long Products<br/>1772 kt]
    H[Tubular Products<br/>156 kt]
    
    I[Sheets/Strips/Coils<br/>1646 kt]
    J[Coated Products<br/>1586 kt]
    
    K[Wire Rod<br/>1557 kt]
    L[Railway Track Material<br/>215 kt]
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

## Product Categories Summary - 2025 (CORRECTED HIERARCHY)

| Production Stage | Category | Total Volume (kt) | Track A Coverage |
|-----------------|----------|-------------------|------------------|
| **Crude Steel** | Total Production | 5516 | ✅ Full |
| **Semi-finished** | Ingots + Continuously-cast | 5533 | ✅ Full |
| **Final Products** | Flat + Long + Tubular | 5160 | ✅ Full |

### Product Categories Breakdown:
| Category | Sub-categories | Total Volume (kt) | Track A Coverage |
|----------|----------------|-------------------|------------------|
| **Flat Products** | Sheets, Strips, Coils, Coated | 3232 | ✅ Full |
| **Long Products** | Wire Rod, Rails, Other Sections | 1772 | ✅ Full |
| **Tubular Products** | Pipes, Tubes | 156 | ✅ Full |

### Production Flow Logic:
- **Crude Steel** (5516 kt) → **Semi-finished** (5533 kt) → **Final Products** (5160 kt)
- **Yield Loss**: 6.7% processing loss from semi-finished to final products

### Product Specialization:
- **Hot Rolled Coil, Sheet, Strip (<3mm)**: 1646 kt
- **Non-metallic Coated Products**: 535 kt  
- **Metal Coated Products**: 1051 kt

*Based on official WSA Product Categories Relationship diagram*

