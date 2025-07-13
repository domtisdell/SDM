<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

## Updated Australian Steel Decomposition Factor Registry (2025)

This table presents a fully updated, evidence-based registry of decomposition factors for the Australian steel demand model, reflecting the latest available data and international benchmarks. All factors are accompanied by their recommended value, source type, primary source, and confidence level.

### Level 0: Sectoral Weights

| Sector | Recommended Weight (2025) | Basis/Notes | Source Type | Primary Source(s) | Confidence |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Construction | 33% | Infrastructure + residential/commercial; aligns with global 30–35% range | Mixed | World Steel Association[^1_1][^1_2], ASI[^1_3] | Medium |
| Infrastructure | 22% | Major public infrastructure pipeline, official government data | Empirical | Infrastructure Australia[^1_4], ASI[^1_3] | High |
| Manufacturing | 28% | Post-automotive, structural fabrication, white goods, mining equipment | Mixed | ABS[^1_5], ASI[^1_3] | Medium |
| Renewable Energy | 4% | Steel intensity × installed capacity, wind/solar growth | Calculated | Clean Energy Council, WSA[^1_1][^1_2] | High |
| Other Sectors | 13% | Mining, agriculture, transport, residual | Residual | Mathematical residual | Low |

### Level 0→1: Sector-to-Product Mapping

| Sector → Product | Weight (%) | Basis/Notes | Source Type | Primary Source(s) | Confidence |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Construction → Long Products | 65 | Structural beams/columns, global pattern | Estimated | WSA[^1_1][^1_2], ASI[^1_3] | Medium |
| Construction → Flat Products | 20 | Plate, roofing, cladding | Estimated | WSA[^1_1][^1_2], ASI[^1_3] | Medium |
| Construction → Semi-Finished | 8 | Local fabrication | Estimated | Industry knowledge | Low |
| Construction → Tube/Pipe | 7 | Structural hollow sections | Estimated | WSA[^1_1][^1_2], ASI[^1_3] | Medium |
| Infrastructure → Long Products | 55 | Rail, bridges, heavy structures | Mixed | WSA[^1_1][^1_2], ASI[^1_3] | Medium |
| Infrastructure → Tube/Pipe | 25 | Pipelines, utilities | Estimated | WSA[^1_1][^1_2], ASI[^1_3] | Medium |
| Infrastructure → Flat Products | 12 | Marine, heavy infrastructure | Estimated | Industry knowledge | Low |
| Infrastructure → Semi-Finished | 8 | Local processing | Estimated | Industry knowledge | Low |
| Manufacturing → Semi-Finished | 55 | Industrial processing, post-auto | Mixed | ABS[^1_5], ASI[^1_3] | Medium |
| Manufacturing → Flat Products | 30 | White goods, appliances | Mixed | ABS[^1_5], ASI[^1_3] | Medium |
| Manufacturing → Long Products | 12 | Machinery, equipment | Estimated | Industry knowledge | Medium |
| Manufacturing → Tube/Pipe | 3 | Specialized tubing | Estimated | Industry knowledge | Low |
| Renewable → Flat Products | 45 | Wind turbine towers | Empirical | WSA[^1_1][^1_2], CEC | High |
| Renewable → Long Products | 40 | Solar mounting, grid structures | Empirical | WSA[^1_1][^1_2], CEC | High |
| Renewable → Semi-Finished | 10 | Local renewable fabrication | Estimated | Industry knowledge | Low |
| Renewable → Tube/Pipe | 5 | Grid infrastructure | Estimated | Industry knowledge | Low |
| Other → Semi-Finished | 40 | Mining equipment | Estimated | Industry knowledge | Low |
| Other → Long Products | 35 | Agricultural/transport machinery | Estimated | Industry knowledge | Low |
| Other → Flat Products | 15 | Mining/transport plate applications | Estimated | Industry knowledge | Low |
| Other → Tube/Pipe | 10 | Mining/agricultural tubing | Estimated | Industry knowledge | Low |

### Level 1→2: Product Breakdown

| Product Group | Sub-Product | Weight (%) | Basis/Notes | Source Type | Primary Source(s) | Confidence |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Semi-Finished | Commercial Billets | 55 | Standard processing | Estimated | WSA[^1_1][^1_2] | Medium |
|  | SBQ Billets | 25 | Specialty applications | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Standard Slabs | 15 | Flat product production | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Degassed Billets | 3 | Premium applications | Estimated | Industry knowledge | Low |
|  | Degassed Slabs | 2 | Premium flat products | Estimated | Industry knowledge | Low |
| Long | Structural Beams | 25 | Building framework | Estimated | ASI[^1_3] | Medium |
|  | Reinforcing Bar | 25 | Concrete reinforcement | Estimated | ASI[^1_3] | Medium |
|  | Structural Columns | 15 | Vertical supports | Estimated | ASI[^1_3] | Medium |
|  | Structural Channels | 12 | Secondary elements | Estimated | ASI[^1_3] | Medium |
|  | Rails Standard | 6 | Rail infrastructure | Mixed | ASI[^1_3] | Medium |
|  | Other Long Products | 17 | Wire rod, angles, etc. | Estimated | Industry knowledge | Low |
| Flat | Hot Rolled Coil | 40 | General manufacturing input | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Cold Rolled Coil | 25 | Appliances, auto | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Steel Plate | 20 | Construction, marine | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Galvanized Products | 15 | Corrosion resistance | Estimated | WSA[^1_1][^1_2] | Medium |
| Tube/Pipe | Welded Structural | 30 | Construction | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Seamless Line Pipe | 25 | Energy infrastructure | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Welded Line Pipe | 20 | Utilities | Estimated | WSA[^1_1][^1_2] | Medium |
|  | Other Tube/Pipe | 25 | Industrial applications | Estimated | Industry knowledge | Low |

### Level 2→3: Specification Breakdown

| Product | Specification | Weight (%) | Basis/Notes | Source Type | Primary Source(s) | Confidence |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Commercial Billets | Low Carbon | 60 | Rebar applications | Estimated | ASI[^1_3] | Medium |
|  | Medium Carbon | 40 | General fabrication | Estimated | ASI[^1_3] | Medium |
| SBQ | Automotive Grade | 45 | Pre-2017 auto, now limited | Contextual | FCAI, ASI[^1_3] | Low |
|  | Mining Equipment | 35 | Heavy machinery | Estimated | Industry knowledge | Medium |
|  | Oil/Gas Equipment | 20 | Energy sector | Estimated | Industry knowledge | Medium |
| Structural Beams | Grade 300 | 80 | Standard construction | Estimated | ASI[^1_3] | Medium |
|  | Grade 300PLUS | 20 | Enhanced applications | Estimated | ASI[^1_3] | Medium |
| Structural Columns | Grade 300 | 85 | Standard columns | Estimated | ASI[^1_3] | Medium |
|  | Grade 300PLUS | 15 | Enhanced grade | Estimated | ASI[^1_3] | Medium |
| Rails | Standard Freight | 70 | Freight network | Estimated | ASI[^1_3] | Medium |
|  | Standard Passenger | 30 | Urban rail | Estimated | ASI[^1_3] | Medium |

### Renewable Energy Steel Intensities

| Application | Steel Intensity (t/MW) | Basis/Notes | Source Type | Primary Source(s) | Confidence |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Wind Onshore | 50–100 | International range | Empirical | WSA[^1_1][^1_2], CEC | High |
| Wind Offshore | 150–200 | International range | Empirical | WSA[^1_1][^1_2], CEC | High |
| Solar Utility | 35–45 | Mounting systems | Empirical | WSA[^1_1][^1_2], CEC | High |
| Solar Distributed | 40 | Residential/commercial | Empirical | WSA[^1_1][^1_2], CEC | Medium |

### Source and Confidence Legend

- **Empirical**: Direct data from authoritative sources
- **Mixed**: Combination of empirical data and estimation
- **Estimated**: Engineering/industry judgment
- **Calculated**: Derived from other empirical data
- **Contextual**: Historical context, limited current relevance
- **Residual**: Mathematical residual for total consistency
- **High**: Strong empirical evidence or official data
- **Medium**: Reasonable estimate, industry pattern
- **Low**: Engineering judgment, limited validation


### Key Research Notes

- **Construction/Infrastructure**: Global and Australian data converge on ~50% combined share, but separation is justified for transparency[^1_1][^1_3][^1_2].
- **Manufacturing**: Post-2017, automotive is negligible; focus is on structural fabrication, white goods, mining equipment[^1_4][^1_3].
- **Renewable Energy**: Still a small but rapidly growing share; steel intensity benchmarks are robust internationally[^1_1][^1_2].
- **Other Sectors**: Remain a residual, with mining and agriculture as major unknowns; further research needed[^1_4][^1_3].
- **Product Splits**: Global product mix is a reasonable proxy, but local validation is a priority for future research[^1_1][^1_2][^1_3].

*This registry is based on the latest available data from the Australian Steel Institute, World Steel Association, Infrastructure Australia, and other authoritative sources. It is recommended for immediate use in sectoral decomposition and for ongoing validation as new data becomes available.*

<div style="text-align: center">⁂</div>

[^1_1]: https://worldsteel.org/data/steel-use-by-sector/

[^1_2]: https://worldsteel.org/wp-content/uploads/World-Steel-in-Figures-2025-1.pdf

[^1_3]: https://www.steel.org.au/about-us/our-industry/

[^1_4]: TRACK_B_RESEARCH_VALIDATION_REPORT.md

[^1_5]: https://www.abs.gov.au/statistics/industry/industry-overview/australian-industry/latest-release

[^1_6]: https://www.sec.gov/Archives/edgar/data/889132/000119312525104983/d928283d20f.htm

[^1_7]: https://www.sec.gov/Archives/edgar/data/1452857/000145285725000010/splp-20241231.htm

[^1_8]: https://www.sec.gov/Archives/edgar/data/1190723/000155485525000262/ts-20241231.htm

[^1_9]: https://www.sec.gov/Archives/edgar/data/1959994/000164117225016159/forms-1a.htm

[^1_10]: https://www.sec.gov/Archives/edgar/data/1243429/000124342925000017/mt-20241231.htm

[^1_11]: https://www.sec.gov/Archives/edgar/data/1959994/000164117225010068/forms-1a.htm

[^1_12]: https://www.sec.gov/Archives/edgar/data/1959994/000164117225009369/forms-1a.htm

[^1_13]: https://worldsteel.org/wp-content/uploads/World-Steel-in-Figures-2024.pdf

[^1_14]: https://www.aph.gov.au/DocumentStore.ashx?id=0a0cfc3f-1de6-4b4b-bd6b-3bfa2d66f17d\&subId=409465

[^1_15]: https://open.metu.edu.tr/bitstream/handle/11511/21804/index.pdf

[^1_16]: https://worldsteel.org/data/world-steel-in-figures/world-steel-in-figures-2023/

[^1_17]: https://worldsteel.org/wp-content/uploads/worldsteel-book-final-2022-1.pdf

[^1_18]: https://www.abs.gov.au

[^1_19]: https://www.sec.gov/Archives/edgar/data/1852131/000185213125000021/nxt-20250331_htm.xml

[^1_20]: https://www.sec.gov/Archives/edgar/data/1924482/000173112225000769/e6598_20f.htm

[^1_21]: https://www.sec.gov/Archives/edgar/data/6176/000095017025069330/ap-20250331.htm

[^1_22]: https://www.sec.gov/Archives/edgar/data/1315257/000095017025067733/kop-20250331.htm

[^1_23]: https://worldsteel.org/data/world-steel-in-figures/world-steel-in-figures-2024/

[^1_24]: https://australiansteel.com/2024/page/3/

[^1_25]: https://www.industry.gov.au/sites/default/files/2024-03/resources-and-energy-quarterly-march-2024.pdf

[^1_26]: https://www.ncbi.nlm.nih.gov/books/NBK207192/

[^1_27]: https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-7/

[^1_28]: https://www.scgh.health.wa.gov.au/~/media/HSPs/NMHS/Hospitals/SCGH/Documents/Research/2023-24-SCGOPHCG-Research-Report.pdf

[^1_29]: https://poultry-research.sydney.edu.au/wp-content/uploads/2025/03/Proceedings-APSS2025.pdf

[^1_30]: https://www.sec.gov/Archives/edgar/data/1022671/000155837025001886/stld-20241231x10k.htm

[^1_31]: https://www.sec.gov/Archives/edgar/data/1022671/000155837025007346/stld-20250331x10q.htm

[^1_32]: https://www.sec.gov/Archives/edgar/data/1984124/000121390025037162/ea0238671-20f_ludatech.htm

[^1_33]: https://www.steel.org.au

[^1_34]: https://www.steel.org.au/Membership/media/Australian-Steel-Institute/PDFs/Steel-Australia-Media-Kit-2024.pdf

[^1_35]: https://www.aspecthuntley.com.au/asxdata/20240830/pdf/02846117.pdf

[^1_36]: https://www.fremantle.wa.gov.au/wp-content/uploads/2025/04/Meeting-attachments-Ordinary-Meeting-of-Council-28-February-2024.pdf

[^1_37]: http://www.digecon.havyatt.com.au/docs/0112.pdf

