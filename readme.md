## MCS PDF Scraper

This project scrapes mines production and reserves tables from the Mineral Commodity Summaries (MCS) PDF file and processes the data to generate a final Excel file.

## Prerequisites

Required Python libraries (install using pip install -r requirements.txt)

## Usage

Follow these steps to use the `main.ipynb` notebook:

1. **Prepare the PDF**
   - Place the target MCS PDF file (e.g., `mcs2024.pdf`) in the project root directory.

2. **Run the Notebook**
   - Open `main.ipynb` in Jupyter Notebook.
   - Run the cells in order, following the instructions in each section.

3. **Extract Material Lists**
   - The notebook will extract the list of materials from the PDF.
   - Review the extracted list in the output.

4. **Match Materials with Pages**
   - The notebook will match each material with its corresponding pages in the PDF.
   - Check the output for any materials with empty or unusual page counts.

5. **Extract Tables and Remarks**
   - The notebook will extract tables and remarks for each material.
   - This step generates `scraping_base_with_tables.json`.

6. **Generate Initial Excel File**
   - The notebook creates `production_reserve_tables1.xlsx`.

7. **Manual Correction**
   - Open `production_reserve_tables1.xlsx` and make any necessary corrections.
   - Save the corrected file as `production_reserve_tables1_updated_manually.xlsx`.

8. **Process Remarks**
   - Run the cells that process the remarks and chemical compositions.
   - This step generates `produc_reserve_remarkinterpre.xlsx`.

9. **Extract Production Data**
   - The notebook will extract and process production data for each material.
   - This generates `production_each_material.xlsx`.

10. **Final Data Processing**
    - Run the remaining cells to process and combine the data.
    - The final output will be saved as `final_table.xlsx`.

## Output

The main output file is `final_table.xlsx`, which contains two sheets:
- `2022 Data`: Production and reserves data for 2022
- `2023 Data`: Production and reserves data for 2023

## Notes

- Make sure to review the output at each step for any anomalies or errors.
- The manual correction step is crucial for ensuring data accuracy.
- If you encounter any issues, check the error messages and make sure all prerequisites are installed correctly.

## Troubleshooting

If you encounter any problems:
- Ensure all required libraries are installed.
- Check that the PDF file is in the correct location and format.
- Review any error messages in the notebook output.
