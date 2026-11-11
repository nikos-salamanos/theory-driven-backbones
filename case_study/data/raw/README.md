# RAAD population- and transport-enriched dataset

This document describes the columns in `RAAD_population_transport_enriched.csv`. The dataset contains amphora assemblage records from the RAAD dataset, alongside additional site population estimates, ORBIS-related information, and bibliographic metadata.


| Column | Description | Provenance |
|---|---|---|
| *[unnamed first column]* | Unnamed first field in the supplied CSV; its meaning is not specified in the column documentation. | Not specified |
| `site_number` | Site identifier; not described in the supplied column documentation. | Not specified |
| `RAAD_type_number` | Unique RAAD amphora-form identifier. | Original Raad dataset |
| `RAAD_form` | Commonly used typologies associated with the RAAD type number. | Original Raad dataset |
| `origin` | Region where an amphora was produced. | Original Raad dataset |
| `contents` | Product carried in an amphora. | Original Raad dataset |
| `southampton_type_number` | Corresponding entry in *Southampton Roman Amphorae: a Digital Resource*, where available. | Original Raad dataset |
| `frequency` | Recorded amphora frequency. | Original Raad dataset |
| `origin_h1` | Region where an amphora was produced. | Original Raad dataset |
| `origin_h2` | Subregion within a production region where an amphora was made. | Original Raad dataset |
| `raad_type_start_date` | Start date of amphora-type production. | Original Raad dataset |
| `raad_type_end_date` | End date of amphora-type production. | Original Raad dataset |
| `site_name_modern` | Modern name of the archaeological site from which an amphora assemblage came. | Original Raad dataset |
| `site_name_ancient` | Roman name of the site, where known. | Original Raad dataset |
| `ORBIS_alt_site_name` | Alternative site name used to calculate transport time when `site_name_ancient` was not found in ORBIS; identified using latitude/longitude. | Added |
| `modern_country` | Modern country associated with a site (interpretation of the field name; not described in the supplied documentation). | Not specified |
| `roman_province` | Ancient provincial location of the site. | Original Raad dataset |
| `population_estimate` | Estimate of the site's population. | Added |
| `source_population_estimate` | Reference supporting the population estimate. | Added |
| `Population_unknown(mean)` | Where no site population could be assigned, an estimate derived from the mean of all known population estimates. | Added |
| `transport_time` | Transport time in days, obtained from ORBIS using `origin` as the origin and `site_name_ancient` as the destination, as described in the supplied documentation. | Added |
| `size (ha)` | Site area in hectares, collected as a reference when population was difficult to estimate. | Added |
| `size_source` | Source documenting the site's area. | Added |
| `Barrington Rank` | Barrington rank, on a scale from 1 to 5. | Added |
| `Rank Source` | Reference supporting the Barrington rank. | Added |
| `pleiades` | Site geographical coordinates according to the Pleiades Atlas. | Original Raad dataset |
| `dare` | Site geographical coordinates according to the Digital Atlas of the Roman Empire. | Original Raad dataset |
| `vici` | Site geographical coordinates according to the Archaeological Atlas of Antiquity. | Original Raad dataset |
| `lat` | Latitude of the archaeological site. | Original Raad dataset |
| `long` | Longitude of the archaeological site. | Original Raad dataset |
| `major_site_type` | Site category: 1 = military; 2 = settlement; 3 = oppidum; 4 = villa. | Original Raad dataset |
| `minor_site_type` | More specific site type, where available. | Original Raad dataset |
| `site_start_date` | Start date of amphora consumption at the site. | Original Raad dataset |
| `site_end_date` | End date of amphora consumption at the site. | Original Raad dataset |
| `quantification_method` | Method by which the assemblage was quantified. | Original Raad dataset |
| `quantification_abbreviation` | Ceramic quantification method: 1 = `total` (total sherd count); 2 = `mni` (minimum number of individuals); 3 = `rbh` (diagnostic rim, base and handle count); 4 = `max` (maximum number of individuals). | Original Raad dataset |
| `total_assemblage_size` | Total number of amphorae or sherds per site. | Original Raad dataset |
| `reference` | Bibliographic reference for publication of the assemblage. | Original Raad dataset |
| `NB` | Additional notes, citations or other relevant information. | Added |

## Transport-data coverage

The transport-time information in this RAAD-derived file is **not a complete matrix for all potential production-origin–consumption-site pairs**. The separate `ORBIS_transport_matrix_24x79.csv` provides transport inputs for the model, including pairs with no observed amphora occurrence. The two files have different roles and should not be conflated.
