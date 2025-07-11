# World Steel Association Statistics Hierarchy Diagrams

## 1. Production Flow Hierarchy
This diagram shows the transformation from raw materials to finished steel products:

```mermaid
graph TD
    A[Production of Iron Ore] --> B[Production of Pig Iron]
    B --> C[Total Production of Crude Steel]
    
    C --> D[Production of Ingots]
    C --> E[Production of Continuously-cast Steel]
    
    D --> F[Production of Hot Rolled Products]
    E --> F
    
    F --> G[Production of Hot Rolled Flat Products]
    F --> H[Production of Hot Rolled Long Products]
    
    G --> I[Production of Hot Rolled Coil, Sheet, and Strip <3mm]
    G --> J[Production of Non-metallic Coated Sheet and Strip]
    G --> K[Production of Other Metal Coated Sheet and Strip]
    
    H --> L[Production of Wire Rod]
    H --> M[Production of Railway Track Material]
    
    C --> N[Total Production of Tubular Products]
```

## 2. Crude Steel Production Methods
This diagram shows the different production pathways for crude steel:

```mermaid
graph TD
    A[Total Production of Crude Steel] --> B[Production of Crude Steel in Electric Furnaces]
    A --> C[Production of Crude Steel in Oxygen-blown Converters]
    
    B --> D[Electric Arc Furnace Route]
    C --> E[Basic Oxygen Furnace Route]
    
    D --> F[Production Methods Combined]
    E --> F
```

## 3. Trade Flow Hierarchy
This diagram illustrates the import/export structure for steel products:

```mermaid
graph TD
    A[Trade in Steel Products]
    
    A --> B[Exports]
    A --> C[Imports]
    
    B --> D[Exports of Iron Ore]
    B --> E[Exports of Pig Iron]
    B --> F[Exports of Scrap]
    B --> G[Exports of Ingots and Semis]
    B --> H[Exports of Flat Products]
    B --> I[Exports of Long Products]
    B --> J[Exports of Semi-finished and Finished Steel Products]
    B --> K[Exports of Tubular Products]
    
    C --> L[Imports of Iron Ore]
    C --> M[Imports of Pig Iron]
    C --> N[Imports of Scrap]
    C --> O[Imports of Direct Reduced Iron]
    C --> P[Imports of Ingots and Semis]
    C --> Q[Imports of Flat Products]
    C --> R[Imports of Long Products]
    C --> S[Imports of Semi-finished and Finished Steel Products]
    C --> T[Imports of Tubular Products]
```

## 4. Steel Use Metrics Hierarchy
This diagram shows the relationship between different steel consumption measures:

```mermaid
graph TD
    A[Steel Consumption Metrics]
    
    A --> B[Apparent Steel Use crude steel equivalent]
    A --> C[Apparent Steel Use finished steel products]
    A --> D[True Steel Use finished steel equivalent]
    
    B --> E[Calculated from Production + Imports - Exports]
    C --> F[Finished Products Basis]
    D --> G[Adjusted for Indirect Trade]
```

## 5. Product Categories Relationship
This diagram shows how different product categories relate to each other:

```mermaid
graph LR
    A[Steel Products] --> B[Semi-finished Products]
    A --> C[Finished Products]
    
    B --> D[Ingots and Semis]
    B --> E[Continuously-cast Steel]
    
    C --> F[Flat Products]
    C --> G[Long Products]
    C --> H[Tubular Products]
    
    F --> I[Sheets/Strips/Coils]
    F --> J[Coated Products]
    
    G --> K[Wire Rod]
    G --> L[Railway Track Material]
    G --> M[Other Long Products]
```

## Key Relationships Explained

### Raw Material to Product Flow:
- **Level 0**: Iron Ore (raw material)
- **Level 1**: Pig Iron (primary processing)
- **Level 2**: Crude Steel (intermediate product)
- **Level 3**: Semi-finished products (Ingots, Continuously-cast steel)
- **Level 4**: Hot Rolled Products (primary finished products)
- **Level 5**: Specialized finished products (Coated sheets, Wire rod, etc.)

### Production Process Branches:
1. **Primary Route**: Iron Ore → Pig Iron → Crude Steel (via oxygen converters)
2. **Secondary Route**: Scrap → Crude Steel (via electric furnaces)
3. **Direct Reduction**: Iron Ore → Direct Reduced Iron → Crude Steel

### Trade Balance Components:
- Each product category has corresponding export and import volumes
- Net trade position = Exports - Imports
- Apparent consumption = Production + Imports - Exports

### Product Categorization:
- **Flat Products**: Sheets, strips, plates, coils
- **Long Products**: Bars, rods, sections, rails
- **Tubular Products**: Pipes and tubes
- **Semi-finished**: Ingots, billets, slabs, blooms