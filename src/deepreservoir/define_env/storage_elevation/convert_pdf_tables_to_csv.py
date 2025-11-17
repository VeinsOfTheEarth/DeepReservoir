# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:07:45 2024
"""
import pandas as pd
import pdfplumber

from deepreservoir.data.loader import DataPaths

# Define the PDF path
pdf_path = DataPaths().elev_area_storage_pdf

## Capacity
# Open and process the PDF
all_capacities = []
all_elevations = []
with pdfplumber.open(pdf_path) as pdf:
    # Extract data from pages 133 to 228
    for page_num in range(138, 234):  # Zero-based indexing for pages
        print(page_num)
        page = pdf.pages[page_num]
        text = page.extract_text()
        
        # Process each line to identify and store table rows
        for line in text.split('\n'):
            elements = line.split()
            if len(elements) == 11:  # Expecting data rows with at least 11 elements
                these_capacities = [float(f) for f in elements[1:]]
                these_elevations = [float(elements[0]) + i/100 for i in range(len(elements)-1)]
               
                all_capacities.extend(these_capacities)
                all_elevations.extend(these_elevations)
                
capacity_df = pd.DataFrame(data={'Elevation (ft)' : all_elevations,
                                 'Capacity (ac-ft)' : all_capacities})
capacity_df.sort_values(by='Elevation (ft)', inplace=True)

                
## Area
# Open and process the PDF
all_areas = []
all_elevations = []
with pdfplumber.open(pdf_path) as pdf:
    # Extract data from pages 133 to 228
    for page_num in range(42, 137):  # Zero-based indexing for pages
        print(page_num)
        page = pdf.pages[page_num]
        text = page.extract_text()
        
        # Process each line to identify and store table rows
        for line in text.split('\n'):
            elements = line.split()
            if len(elements) == 11:  # Expecting data rows with at least 11 elements
                these_areas = [float(f) for f in elements[1:]]
                these_elevations = [float(elements[0]) + i/100 for i in range(len(elements)-1)]
               
                all_areas.extend(these_areas)
                all_elevations.extend(these_elevations)

area_df = pd.DataFrame(data={'Elevation (ft)' : all_elevations,
                                 'Area (ac)' : all_areas})
area_df.sort_values(by='Elevation (ft)', inplace=True)


## Combine into single dataframe and save
df = capacity_df.merge(area_df, on='Elevation (ft)')
df.to_csv(DataPaths.elev_area_storage_csv')